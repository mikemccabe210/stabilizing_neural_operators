import os
import time
import numpy as np
import argparse
import h5py
import torch
import cProfile
import re
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from utils.opt_utils import AdamW, Shampoo
from networks.afnonet import AFNONet, PrecipNet, FourierInterpFilter
from networks.FNO import fno
from utils.img_utils import vis_precip, vis_swe
import wandb
from utils.weighted_acc_rmse import weighted_acc, weighted_rmse, weighted_rmse_torch,  unlog_tp_torch, weighted_rmse_torch_channels
from apex import optimizers
from utils.darcy_loss import LpLoss
import matplotlib.pyplot as plt
from collections import OrderedDict
import pickle
DECORRELATION_TIME = 36 # 9 days
import json
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
import multiprocessing as mp
from einops import rearrange

era5_var_key_dict = {0: 'u10' \
              , 1: 'v10' \
              , 2: 't2m' \
              , 3: 'sp'  \
              , 4: 'msl' \
              , 5: 't850'\
              , 6: 'u1000'\
              , 7: 'v1000'\
              , 8: 'z1000'\
              , 9: 'u850' \
              , 10: 'v850'\
              , 11: 'z850'\
              , 12: 'u500'\
              , 13: 'v500'\
              , 14: 'z500'\
              , 15: 't500'\
              , 16: 'z50' \
              , 17: 'r500'\
              , 18: 'r850'\
              , 19: 'tcwv'}

swe_var_key_dict = {0: 'u',
        1: 'v',
        2: 'h'}

def grad_clip(grad):
    max_norm = 1 #TODO generalize this so we can alter the "trust region"
    total_norm  = torch.norm(grad)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    return clip_coef_clamped * grad
class DoubleLoss(nn.Module):
    def forward(self, x):
        return x, x

class AliasLevelSplit(nn.Module):
    def forward(self, x):
        x=  torch.fft.rfft2(x)
        outs = []
        x = torch.fft.fftshift(x, dim=2)
        for level in range(3, -1, -1):
            res = torch.zeros_like(x)
            res[:, :, 1440//2 - (1440//2**(level+1)):1440//2+(1440//2**(level+1)), :1440//2**level //2 +1] = x[:, :, 1440//2 - (1440//2**(level+1)):1440//2+(1440//2**(level+1)), :1440//2**level //2 +1]
            for out in outs:
                res = res - out
            outs.append(res)
        return tuple(torch.fft.irfft2(torch.fft.ifftshift(res, dim=2)) for res in outs)

def rebalance_losses(m, grad_inputs, grad_outputs):
    grad_norm1 = torch.linalg.norm(grad_outputs[0])
    rescaled_gn1 = grad_outputs[0] / grad_norm1
    out = rescaled_gn1.clone()
    for i, grad in enumerate(grad_outputs[1:]):
        grad = grad / torch.linalg.norm(grad)
        proj = (out*grad).sum()
        print(i, proj)
        if proj < 0:
            grad = grad - proj*out
        out += grad
        out /= torch.linalg.norm(out)
    return (grad_norm1*out,)

class RescaledMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
        self.splitter = AliasLevelSplit()
        self.splitter.register_full_backward_hook(rebalance_losses)
        self.y_splitter = AliasLevelSplit()
    def forward(self, x, y):
        xs = self.splitter(x)
        ys = self.y_splitter(y)
        return sum([self.loss(x, y) for x, y in zip(xs, ys)])



def scale_and_ortho(m, grad_inputs, grad_outputs):
    grad1, grad2 = grad_outputs
    #print('ginputs', [g.shape for g in grad_inputs])
    #print(grad1.shape, grad2.shape)
    g1_norm = torch.linalg.norm(grad1)
    g2_norm = torch.linalg.norm(grad2)
    grad1_new = grad1 / g1_norm
    grad2_new = grad2 / g2_norm
    proj = (grad1_new*grad2_new).sum()
    print('proj!', proj, g1_norm, g2_norm)
    if proj < 0:
        grad2_new = grad2_new - proj*grad1_new
    else:
        grad2_new = grad2_new
    grad2_new = grad2_new / torch.linalg.norm(grad2_new)
    #print('out', grad1_new.shape, grad2_new.shape)
    return (g1_norm*(grad1_new + grad2_new),)

class RSpectralLoss():
    def __call__(self, x, y):
        yfft = torch.fft.rfft2(y, norm='ortho')
        xfft = torch.fft.rfft2(x, norm='ortho')
        return ((yfft - xfft).abs()**2 / (yfft.abs()+1e-10)**2).mean()
        
class Trainer():
  def count_parameters(self):
    return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

  @torch.no_grad()
  def downsample_data(self, x, y, scale_factor=2.):
      return F.interpolate(x, mode='bicubic', scale_factor=scale_factor), F.interpolate(y, mode='bicubic', scale_factor=scale_factor)

  def add_weight_decay(self, weight_decay=1e-5, inner_lr=1e-3, skip_list=()):
    """ From Ross Wightman at:
        
        https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3 """
    decay = []
    no_decay = []
    spectral = []
    for name, param in self.model.named_parameters():
        if not param.requires_grad:
            continue
        # Hacking this together to make things easier
        if (len(param.squeeze().shape) <= 1 or name in skip_list):
            no_decay.append(param)
        elif 'mags' in name or 'phases' in name:
            spectral.append(param)
        else:
            decay.append(param)
    return [
            {'params': no_decay, 'weight_decay': 0.,},
            {'params': decay, 'weight_decay': weight_decay, 'lr': inner_lr},
            {'params': spectral, 'weight_decay':0, 'lr': inner_lr*10}]

  def __init__(self, params, world_rank):
    self.sweep_id = args.sweep_id
    self.params = params
    self.rollout_milestones = params.rollout_milestones[:]
    self.world_rank = world_rank
    self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    #if params.log_to_wandb:
    #  if self.sweep_id:
    #      wandb.init(dir=params.experiment_dir)
    #      print('what about here')
    #      hpo_config = wandb.config
    #      self.params.update_params(hpo_config)
    #      print('GOT HEREEEEEEEEEEEEEEEEEE')
    #  else:
    #    wandb.init(dir=params.experiment_dir, config=params, name=params.name, group=params.group, project=params.project, entity=params.entity, resume=True)

    logging.info('rank %d, begin data loader init'%world_rank)
    self.loss_obj = nn.MSELoss()# LpLoss()
    self.valid_obj = LpLoss()
    self.freq_obj = RSpectralLoss()
    self.doubler = DoubleLoss()
    self.doubler.register_full_backward_hook(scale_and_ortho)
    # precip models
    self.precip = True if "precip" in params else False
  
  def setup_model(self, params): 
    print('beginning of setup')
    if self.precip:
      if 'model_wind_path' not in params:
        raise Exception("no backbone model weights specified")
      # load a wind model 
      # the wind model has out channels = in channels
      out_channels = np.array(params['in_channels'])
      params['N_out_channels'] = len(out_channels)

      if params.nettype_wind == 'afno':
        self.model_wind = AFNONet(params).to(self.device)
      else:
        raise Exception("not implemented")

      if dist.is_initialized():
        self.model_wind = DistributedDataParallel(self.model_wind,
                                            device_ids=[params.local_rank],
                                            output_device=[params.local_rank],find_unused_parameters=True)
      self.load_model_wind(params.model_wind_path)
      self.switch_off_grad(self.model_wind) # no backprop through the wind model

    # reset out_channels for precip models
    if self.precip:
      params['N_out_channels'] = len(params['out_channels'])
    print('pre_model!')
    if params.nettype == 'afno':
      self.model = AFNONet(params).to(self.device) 
    elif params.nettype == 'fno':
      self.model = fno(params).to(self.device)
    else:
      raise Exception("not implemented")
     
    # precip model
    if self.precip:
      self.model = PrecipNet(params, backbone=self.model).to(self.device)

    if self.params.enable_nhwc:
      # NHWC: Convert model to channels_last memory format
      self.model = self.model.to(memory_format=torch.channels_last)

    #if params.log_to_wandb:
    #  wandb.watch(self.model)
    if params.weight_decay >= 0:
        if 'FS' in params.optimizer_type:
            model_params = self.add_weight_decay(params.weight_decay, params.fixed_step_norm)
        else:
            model_params = self.add_weight_decay(params.weight_decay, params.lr)
    else:
        model_params = self.model.parameters()

    if params.optimizer_type == 'FusedAdam':
        max_gnorm = (.1*sum([p.numel() for p in self.model.parameters()]))**.5 # Want the constraint, but don't want the numerical issues
        print(max_gnorm)
        self.optimizer = optimizers.FusedLAMB(model_params, lr = params.lr, 
                #betas=(.5, .6),
                weight_decay=params.weight_decay, max_grad_norm=max_gnorm, eps=1e-14)
    else:
      self.optimizer = torch.optim.Adam(model_params, lr = params.lr, weight_decay=params.weight_decay)
      #self.optimizer = AdamW(model_params, lr=params.lr, weight_decay=params.weight_decay)
    if params.enable_amp == True:
      self.gscaler = amp.GradScaler()

#    self.hook_handles = []
#    for name, p in self.model.named_parameters():
#        print(name)
#        h = p.register_hook(lambda x: grad_clip(x))
#        self.hook_handles.append(h)
    self.iters = 0
    self.startEpoch = 0
    if params.resuming:
      logging.info("Loading checkpoint %s"%params.checkpoint_path)
      self.restore_checkpoint(params.checkpoint_path)
    if params.resuming == False and params.pretrained == True:
      logging.info("Starting from pretrained one-step afno model at %s"%params.pretrained_ckpt_path)
      self.restore_checkpoint(params.pretrained_ckpt_path)
      self.iters = 0
      self.startEpoch = 0
        #logging.info("Pretrained checkpoint was trained for %d epochs"%self.startEpoch)
        #logging.info("Adding %d epochs specified in config file for refining pretrained model"%self.params.max_epochs)
        #self.params.max_epochs += self.startEpoch
    if params.upscale and params.rescale_factor > 1:
        if params.resuming is False:
            logging.info("Starting from pretrained one-step afno model at %s" % params.old_checkpoint_path)
            self.restore_checkpoint(params.old_checkpoint_path)
            self.iters = 0
            self.startEpoch = 0
            self.model.resize((self.model.img_size[0]*params.rescale_factor, self.model.img_size[1]*params.rescale_factor))
    if dist.is_initialized():
      self.model = DistributedDataParallel(self.model,
                                           device_ids=[params.local_rank],
                                           output_device=[params.local_rank],find_unused_parameters=True,
                                           broadcast_buffers=False)
    # Bisection would be faster, but i don't think this is long enough to matter?
    for milestone in self.params.rollout_milestones:
      if self.startEpoch >= milestone:
        self.params['rollout_length'] += 1
        self.rollout_milestones.pop(0)
    logging.info('rank %d, begin data loader init'%world_rank)
    self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(params, params.train_data_path, dist.is_initialized(), train=True)
    self.valid_data_loader, self.valid_dataset = get_data_loader(params, params.valid_data_path, dist.is_initialized(), train=False)
    self.long_valid_data_loader, self.long_valid_dataset = get_data_loader(params, params.valid_data_path, dist.is_initialized(), train=False, long=True)
    logging.info('rank %d, data loader initialized'%world_rank)
    num_batches = len(self.train_data_loader)
    params.crop_size_x = self.valid_dataset.crop_size_ns
    params.crop_size_y = self.valid_dataset.crop_size_ew
    params.img_shape_x = self.valid_dataset.img_shape_ns
    params.img_shape_y = self.valid_dataset.img_shape_ew
            
    self.epoch = self.startEpoch

    if params.scheduler == 'ReduceLROnPlateau':
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
    elif params.scheduler == 'CosineAnnealingLR':
      # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.max_epochs, last_epoch=self.startEpoch-1)
      print('start epoch', self.startEpoch)
      k = params.warmup_steps
      if self.startEpoch < k:
        warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=1.0, total_iters=k*num_batches)
        decay = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.max_epochs*num_batches)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, [warmup, decay], [k*num_batches], last_epoch=(num_batches*self.startEpoch)-1)
      else:
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.max_epochs*num_batches, last_epoch=(self.startEpoch-k)*num_batches)
    else:
      self.scheduler = None

    '''if params.log_to_screen:
      logging.info(self.model)'''
    if params.log_to_screen:
      logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))

  def switch_off_grad(self, model):
    for param in model.parameters():
      param.requires_grad = False

  def train(self):
    params = self.params
    if self.params.log_to_wandb:
      if self.sweep_id:
          wandb.init(dir=params.experiment_dir)
          print('what about here')
          hpo_config = wandb.config.as_dict()
          self.params.update_params(hpo_config)
          params = self.params
          print('GOT HEREEEEEEEEEEEEEEEEEE')
      else:
        wandb.init(dir=params.experiment_dir, config=params, name=params.name, group=params.group, project=params.project, entity=params.entity, resume=True)
    if self.sweep_id and dist.is_initialized():
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        assert self.world_rank == rank
        if rank != 0: 
            hpo_config = None
        hpo_config = comm.bcast(hpo_config, root=0)
        self.params.update_params(hpo_config)
        params = self.params
        #self.params = comm.bcast(self.params, root=0)
        #self.params.device = self.device # dont broadcast 0s device
        #self.params.log_to_wandb = (self.params.log_to_wandb and self.world_rank == 0)

    self.setup_model(self.params)
    if self.params.log_to_wandb:
      wandb.watch(self.model)

    if self.params.log_to_screen:
      logging.info("Starting Training Loop...")
    #for ln in self.model.module.lns:
    #    ln.nu2.requires_grad = False
    #    ln.nu.requires_grad = False
    self.model.module.diffusion.nu.requires_grad = False
    self.model.module.diffusion.nu2.requires_grad = False

    best_valid_loss = 1.e6
    for epoch in range(self.startEpoch, self.params.max_epochs):
      if epoch >= 5:
      #  for ln in self.model.module.lns:
       #     ln.nu2.requires_grad = True
        #    ln.nu.requires_grad = True
        self.model.module.diffusion.nu.requires_grad = True
        self.model.module.diffusion.nu2.requires_grad = True

      # Check if we need to increase rollout length based on provided milestones
      if len(self.rollout_milestones) > 0:
        if epoch >= self.rollout_milestones[0]:
          self.params['rollout_length'] += 1
          self.rollout_milestones.pop(0)
          logging.info('rank %d, Refreshing data loader' % world_rank)
          self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(self.params, self.params.train_data_path,
                                                                                           dist.is_initialized(),
                                                                                           train=True)
      if dist.is_initialized():
        self.train_sampler.set_epoch(epoch)
#        self.valid_sampler.set_epoch(epoch)

      start = time.time()
      tr_time, data_time, train_logs = self.train_one_epoch()
      valid_start = time.time()
      valid_time, valid_logs = self.validate_one_epoch()
      if epoch==self.params.max_epochs-1 and self.params.prediction_type == 'direct':
        valid_weighted_rmse = self.validate_final()


      post_start = time.time()
#      if self.params.scheduler == 'ReduceLROnPlateau':
#        self.scheduler.step(valid_logs['valid_loss'])
#      elif self.params.scheduler == 'CosineAnnealingLR':
#        self.scheduler.step()
#        if self.epoch >= self.params.max_epochs:
#          logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
#          exit()
      if self.params.log_to_wandb:
        for pg in self.optimizer.param_groups:
          lr = pg['lr']
        wandb.log({'lr': lr})
      
      if self.world_rank == 0:
        if self.params.save_checkpoint:
          #checkpoint at the end of every epoch
          self.save_checkpoint(self.params.checkpoint_path)
          if valid_logs['valid_loss'] <= best_valid_loss:
            #logging.info('Val loss improved from {} to {}'.format(best_valid_loss, valid_logs['valid_loss']))
            self.save_checkpoint(self.params.best_checkpoint_path)
            best_valid_loss = valid_logs['valid_loss']
      
      cur_time = time.time()
      print('Time for train {}. For valid: {}. For postprocessing:{}'.format(valid_start-start, post_start-valid_start, cur_time-post_start))
      if self.params.log_to_screen:
        logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        #logging.info('train data time={}, train step time={}, valid step time={}'.format(data_time, tr_time, valid_time))
        logging.info('Train loss: {}. Valid loss: {}'.format(train_logs['loss'], valid_logs['valid_loss']))
#        if epoch==self.params.max_epochs-1 and self.params.prediction_type == 'direct':
#          logging.info('Final Valid RMSE: Z500- {}. T850- {}, 2m_T- {}'.format(valid_weighted_rmse[0], valid_weighted_rmse[1], valid_weighted_rmse[2]))



  def train_one_epoch(self):
    self.epoch += 1
    tr_time = 0
    data_time = 0
    last_step_time = None
    last_step_end = None
    self.model.train()
    last_step = None
    last_step_mag = None
    step_logs = []
    last_mags = [None]*4
    dtype = torch.float
    self.model = self.model.to(dtype)
    torch.set_printoptions(edgeitems=5)
    seed = torch.Generator() # This is on CPU 
    seed.manual_seed(1337*self.epoch) # Doesn't really matter since batches are random, but want all devices using same rollout for efficiency 
    for i, data in enumerate(self.train_data_loader, 0):
      #if self.world_rank==0:
         # for j, conv in enumerate(self.model.module.sp_convs):
          #    if last_mags[j] is not None:
           #     diff = (conv.ew_mags - last_mags[j]).abs()
            #    diff = rearrange(diff, 'b h w c -> h w (b c)')
             #   print('conv diff',  j,torch.median(diff, dim=-1).values[50:60])
             # last_mags[j] = conv.ew_mags.clone()

      with torch.no_grad():
        if self.params.rollout_length > self.params.max_grad_steps:
          ps = torch.softmax(torch.arange(self.params.rollout_length, 0, -1) / (self.epoch+1), dim = -1)
          bp_steps = torch.multinomial(ps, self.params.max_grad_steps, replacement=False, generator=seed)
        else:
          ps = torch.ones(self.params.rollout_length)
          bp_steps = torch.arange(self.params.rollout_length, device=self.device)
      self.iters += 1
      # adjust_LR(optimizer, params, iters)
      data_start = time.time()

      augs = data[-1]
      augs = None # Just avoid augs entirely for now
      data = data[:-1]

      inp, tar, year_idx, local_idx = map(lambda x: x.to(self.device, dtype = dtype), data)      
      
      #if self.params.grid_res != 'full':
      #    inp = self.clip(inp)
      #    tar = self.clip(tar)
      
      if self.params.orography and self.params.rollout_length > 1:
        orog = inp[:,-2:-1]

      #if params.scale_factor != 1:
      #  inp, tar = self.downsample_data(inp, tar, scale_factor=self.params.scale_factor * self.params.rescale_factor)

      if self.params.enable_nhwc:
        inp = inp.to(memory_format=torch.channels_last)
        tar = tar.to(memory_format=torch.channels_last)

      # TODO Pretty sure this doesn't actually work unless rollout is len 1, but not fixing it now since
      # we don't use it
      if 'residual_field' in self.params.target:
        tar -= inp[:, 0:tar.size()[1]]
      data_time += time.time() - data_start
      if last_step_time is not None:
          print('data time', time.time() - last_step_time, time.time()-data_start)
      tr_start = time.time()

      self.model.zero_grad()
      with amp.autocast(self.params.enable_amp):
        if self.precip: # use a wind model to predict 17(+n) channels at t+dt
          with torch.no_grad():
            inp = self.model_wind(inp).to(self.device, dtype = torch.float)
          gen = self.model(inp.detach(), local_idx, augs).to(self.device, dtype = torch.float)
          loss = self.loss_obj(gen, tar)
        else:
          # Check if we're bping through the first example
          with torch.set_grad_enabled(torch.isin(0, bp_steps, assume_unique=True).item()):
            gen = self.model(inp, local_idx, augs, extra_out=True).to(self.device, dtype = dtype)
            #gen, gen2 = self.doubler(gen)
            first_loss = self.loss_obj(gen, tar[:,0:self.params.N_out_channels])
            #freq_loss = self.freq_obj(gen, tar[:,0:self.params.N_out_channels])
            #print('Losses', first_loss, freq_loss)
            # first_loss += freq_loss
          with torch.no_grad():
            l1_loss = nn.functional.l1_loss(gen, tar[:,0:self.params.N_out_channels])
          loss = first_loss
          loss_by_step = [first_loss]
          for step in range(1, self.params.rollout_length):
            # Remember to remove if re-implementing longer rollouts
            if torch.isin(0, bp_steps, assume_unique=True).item() and self.params.max_grad_steps==1:
                break
            local_idx = local_idx + 1
            # Check if we're bping past this step - we might do some extra compute on long rollouts
            # due to sampling but that's because break and local computation don/self.'t play nice.
            with torch.set_grad_enabled(torch.isin(step, bp_steps, assume_unique=True).item()):
              if self.params.orography:
                gen = torch.cat((gen, orog), axis = 1)
              gen = self.model(gen, local_idx, augs, extra_out=True).to(self.device, dtype = torch.float)
              new_loss = self.loss_obj(gen, tar[:,step*self.params.N_out_channels:(step+1)*self.params.N_out_channels])
            # Note - in tests, it looks like this does correctly BP through only a couple examples
            if torch.isin(step, bp_steps, assume_unique=True).item(): # Shouldn't be needed, but just incase...
                loss += new_loss #/ min(self.params.rollout_length, self.params.max_grad_steps)
            loss_by_step.append(new_loss)
      loss = loss / min(self.params.rollout_length, self.params.max_grad_steps)
      #print('step', i, loss, year_idx, local_idx)
      if self.params.enable_amp:
        self.gscaler.scale(loss).backward()
        self.gscaler.step(self.optimizer)
        #self.gscaler.unscale_(self.optimizer)
        """ # Some debugging tools for the optimizer - not really needed anymore
        old_params = []
        cur_grad_norm = 0
        step_mag = 0
        with torch.no_grad():
            for k, g in enumerate(self.optimizer.param_groups):
                op = []
                for j, p in enumerate(g['params']):
                    if p.grad is not None:
                        op.append(p.detach().clone())
                        cur_grad_norm += (p.grad**2).sum()
                    else:
                        op.append(None)
                old_params.append(op)
            cur_grad_norm = cur_grad_norm**.5
        self.gscaler.step(self.optimizer)
        last_step = []
        with torch.no_grad():
            for k, g in enumerate(self.optimizer.param_groups):
                ls = []
                for j, p in enumerate(g['params']):
                    if p.grad is not None:
                        diff =  p.detach() - old_params[k][j]
                        step_mag += (diff**2).sum()
                        last_step.append(diff)
                    else:
                        last_step.append(None)
            step_mag = step_mag ** .5
            last_step_mag = step_mag.clone()
            log_things = {'train_grad_norm': cur_grad_norm, 'train_laststep_size': step_mag}
        
           # for key in log_things:
            #    dist.all_reduce(log_things[key].detach(), op=dist.ReduceOp.AVG)
                # log_things[key] = log_things[key].detach().cpu().item()
        """
        print('Epoch', self.epoch, i, year_idx[0].item(), local_idx[0].item(), local_idx[0].item() % 4, 'loss', loss.item(), bp_steps)
      # wandb.log(log_things)
      else:
        loss.backward()
        self.optimizer.step()
            #1_pen += conv.mags.abs().mean()
      # L1 prox on spectral mags
      for j, (name, p) in enumerate(self.model.module.named_parameters()):
        with torch.no_grad():
            if 'mags' in name:
                pass#p.copy_(F.softshrink(p, self.optimizer.param_groups[2]['lr']*1e-3))
      print('Epoch', self.epoch, i, year_idx[0].item(), local_idx[0].item(), local_idx[0].item() % 4, 'loss', loss.item(), bp_steps)
      if self.params.enable_amp:
        self.gscaler.update()

      if self.params.scheduler == 'CosineAnnealingLR':
        self.scheduler.step()
        print('lr', self.optimizer.state_dict()['param_groups'][0]['lr'])
      tr_time += time.time() - tr_start
      if last_step_time is not None:
          print('full step', time.time() - last_step_time)
          print('model time', time.time() - tr_start)
      last_step_time = time.time()
      #if i > 10:
      #  break
    try:
        logs = {'loss': loss / min(params.rollout_length, params.max_grad_steps),
                'train_l1': l1_loss}
    except:
        logs = {'loss': loss / self.params.rollout_length,
                'train_l1': l1_loss}

    if dist.is_initialized():
      for key in sorted(logs.keys()):
        dist.all_reduce(logs[key].detach())
        logs[key] = float(logs[key]/dist.get_world_size())
    if self.params.log_to_wandb:
      wandb.log(logs, step=self.epoch)

    return tr_time, data_time, logs

  def validate_one_epoch(self):
      self.model = self.model.float()
      if self.epoch % self.params.long_valid_rollout_interval == 0:
          valid_data_loader = self.long_valid_data_loader
          valid_rollout_length = self.params.long_valid_rollout_length
          n_valid_batches = 1
          prefix = 'long'
      else:
          valid_data_loader = self.valid_data_loader
          valid_rollout_length = self.params.valid_rollout_length
          n_valid_batches = 20
          prefix = ''


      if 'swe' in params.config:
          var_key_dict = swe_var_key_dict
      else:
          var_key_dict = era5_var_key_dict

      self.model.eval()
      #n_valid_batches = 20  # do validation on first 20 images, just for LR scheduler
      if self.params.normalization == 'minmax':
          mult = torch.as_tensor(
              np.load(self.params.max_path)[0, self.params.out_channels, 0, 0] - np.load(self.params.min_path)[
                  0, self.params.out_channels, 0, 0]).to(self.device)
      elif self.params.normalization == 'zscore':
          mult = torch.as_tensor(np.load(self.params.global_stds_path)[0, self.params.out_channels, 0, 0]).to(
              self.device)

      valid_buff = torch.zeros((3), dtype=torch.float32, device=self.device)
      valid_loss = valid_buff[0].view(-1)
      valid_l1 = valid_buff[1].view(-1)
      valid_steps = valid_buff[2].view(-1)
      step_losses = torch.zeros((valid_rollout_length,), dtype=torch.float32, device=self.device)
      valid_weighted_rmse = torch.zeros((self.params.N_out_channels), dtype=torch.float32, device=self.device)
      valid_weighted_acc = torch.zeros((self.params.N_out_channels), dtype=torch.float32, device=self.device)

      valid_start = time.time()

      sample_idx = np.random.randint(n_valid_batches)
      with torch.no_grad():
          for i, data in enumerate(valid_data_loader, 0):
              if (not self.precip) and i >= n_valid_batches:
                  break
              augs = data[-1]
              augs = None # Broke augs to speed things up since it didnt seem to help anyway
              data = data[:-1]
              inp, tar, year_idx, local_idx = map(lambda x: x.to(self.device, dtype=torch.float), data)
              if self.params.orography and self.params.rollout_length > 1: # this looks wrong, but not using orography, so not debugging atm
                  orog = inp[:, -2:-1]

              inp, dtar = self.downsample_data(inp, tar,
                                               scale_factor=self.params.scale_factor * self.params.rescale_factor)
              if self.precip:
                  with torch.no_grad():
                      inp = self.model_wind(inp).to(self.device, dtype=torch.float)
                  gen = self.model(inp.detach(), local_idx, augs)
                  valid_loss += self.valid_obj(gen, tar)
                  valid_l1 += nn.functional.l1_loss(gen, tar)
              else:
                  gen = self.model(inp, local_idx, augs).to(self.device, dtype=torch.float)
                  gen, _ = self.downsample_data(gen, dtar, scale_factor=1 / (
                              self.params.scale_factor * self.params.rescale_factor))
                  sample_loss = self.valid_obj(gen, tar[:, 0:self.params.N_out_channels])
                  valid_loss += sample_loss
                  step_losses[0] += sample_loss
                  valid_l1 += nn.functional.l1_loss(gen, tar[:, 0:self.params.N_out_channels])
                  valid_weighted_rmse += weighted_rmse_torch(gen, tar[:, 0 * self.params.N_out_channels:(0 + 1) * self.params.N_out_channels])
                  valid_steps += 1.
                  for step in range(1, valid_rollout_length):
                      if self.params.orography:
                          gen = torch.cat((gen, orog), axis=1)
                      local_idx = local_idx + 1
                      gen = self.model(gen, local_idx, augs).to(self.device, dtype=torch.float)
                      new_loss = self.valid_obj(gen, tar[:, step * self.params.N_out_channels:(
                                                                                                         step + 1) * self.params.N_out_channels])
                      step_losses[step] += new_loss
                      valid_loss += new_loss
                      valid_l1 += nn.functional.l1_loss(gen, tar[:, step * self.params.N_out_channels:(step + 1) * self.params.N_out_channels])
                      valid_weighted_rmse += weighted_rmse_torch(gen, tar[:, step * self.params.N_out_channels:(step + 1) * self.params.N_out_channels])
                      valid_steps += 1.
              # save fields for vis before log norm
              if (i == sample_idx):
                  if (self.precip and self.params.log_to_wandb):
                    fields = [gen[0, 0].detach().cpu().numpy(), tar[0, 0].detach().cpu().numpy()]
                  elif len(prefix) > 0:
                      fields = [gen[0].detach().cpu().numpy(), tar[0].detach().cpu().numpy()]
              if self.precip:
                  gen = unlog_tp_torch(gen, self.params.precip_eps)
                  tar = unlog_tp_torch(tar, self.params.precip_eps)



              # if not self.precip:
              #     try:
              #         os.mkdir(params['experiment_dir'] + "/" + str(i))
              #     except:
              #         pass
              #     # save first channel of last image
              #     save_image(torch.cat((gen[0, 0], torch.zeros((self.valid_dataset.img_shape_x, 4)).to(self.device,
              #                                                                                          dtype=torch.float),
              #                           tar[0, 0]), axis=1),
              #                params['experiment_dir'] + "/" + str(i) + "/" + str(self.epoch) + ".png")

          # Do one batch power iteration on noise
          noise = torch.randn_like(inp)
          power_iter = self.model(noise, local_idx, augs)
          noise_op_est = torch.linalg.norm(power_iter) / torch.linalg.norm(noise)

          indist_power_iter = self.model(inp, local_idx, augs)
          indist_op_est = torch.linalg.norm(indist_power_iter) / torch.linalg.norm(inp)
          #twostep_op_est = torch.linalg.norm(self.model(indist_power_iter, local_idx, augs)) / torch.linalg.norm(indist_power_iter)

      if dist.is_initialized():
          dist.all_reduce(valid_buff)
          dist.all_reduce(valid_weighted_rmse)
          dist.all_reduce(step_losses)
          dist.all_reduce(noise_op_est, op=dist.ReduceOp.AVG)
          dist.all_reduce(indist_op_est, op=dist.ReduceOp.AVG)
       #   dist.all_reduce(twostep_op_est, op=dist.ReduceOp.AVG)
      # divide by number of steps
      valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]
      valid_weighted_rmse = valid_weighted_rmse / valid_buff[2]
      step_losses = step_losses / valid_buff[2]

      if not self.precip:
          valid_weighted_rmse *= mult

      # download buffers
      valid_buff_cpu = valid_buff.detach().cpu().numpy()
      valid_weighted_rmse_cpu = valid_weighted_rmse.detach().cpu().numpy()
      step_losses_cpu = step_losses.detach().cpu().numpy()
      if valid_rollout_length > 1:
          # Let's just log this periodically so we don't have 100 extra valids - should probably parameterize the logging
          step_logs = {prefix+'valid_step_%s_loss' % i: step_losses_cpu[i] for i in range(0, valid_rollout_length, valid_rollout_length//5)}
      else:
          step_logs = {}
      valid_time = time.time() - valid_start
      loss_logs = {(prefix+'valid_rmse_' + k): valid_weighted_rmse_cpu[i] for i, k in var_key_dict.items()}
      # valid_weighted_rmse = mult*torch.mean(valid_weighted_rmse, axis = 0)
      if self.precip:
          logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0],
                  'valid_rmse_tp': valid_weighted_rmse_cpu[0],
                  'noise_op_est': noise_op_est, 'data_op_est': indist_op_est, }
      else:
          try:
              logs = {prefix+'valid_l1': valid_buff_cpu[1] ,
                      prefix+'valid_loss': valid_buff_cpu[0] ,
                      'noise_op_est': noise_op_est, 'data_op_est': indist_op_est, }
          except:
              logs = {prefix+'valid_l1': valid_buff_cpu[1], prefix+'valid_loss': valid_buff_cpu[0],
                      'noise_op_est': noise_op_est, 'data_op_est': indist_op_est, }
      logs.update(step_logs)
      logs.update(loss_logs)
      if self.params.log_to_wandb:
          if self.precip:
              fig = vis_precip(fields)
              logs['vis'] = wandb.Image(fig)
              plt.close(fig)
          elif len(prefix)>0:
              fig = vis_swe(fields)
              logs['vis'] = wandb.Image(fig)
              plt.close(fig)

          wandb.log(logs, step=self.epoch)
      if len(prefix) > 0:
          logs['valid_loss'] = logs[prefix+'valid_loss']
      return valid_time, logs

  def validate_final(self):
    self.model.eval()
    n_valid_batches = int(self.valid_dataset.n_patches_total/self.valid_dataset.n_patches) #validate on whole dataset
    valid_weighted_rmse = torch.zeros(n_valid_batches, self.params.N_out_channels)
    if self.params.normalization == 'minmax':
        torch.as_tensor(mult = np.load(self.params.max_path)[0, self.params.out_channels, 0, 0] - np.load(self.params.min_path)[0, self.params.out_channels, 0, 0]).to(self.device)
    elif self.params.normalization == 'zscore':
        mult = torch.as_tensor(np.load(self.params.global_stds_path)[0, self.params.out_channels, 0, 0]).to(self.device)

    with torch.no_grad():
        for i, data in enumerate(self.valid_data_loader):
          if i>100:
            break
          inp, tar = map(lambda x: x.to(self.device, dtype = torch.float), data)
          if self.params.orography and self.params.two_step_training:
              orog = inp[:,-2:-1]
          if 'residual_field' in self.params.target:
            tar -= inp[:, 0:tar.size()[1]]

        if self.params.two_step_training:
            gen_step_one = self.model(inp).to(self.device, dtype = torch.float)
            loss_step_one = self.loss_obj(gen_step_one, tar[:,0:self.params.N_out_channels])

            if self.params.orography:
                gen_step_two = self.model(torch.cat( (gen_step_one, orog), axis = 1)  ).to(self.device, dtype = torch.float)
            else:
                gen_step_two = self.model(gen_step_one).to(self.device, dtype = torch.float)

            loss_step_two = self.loss_obj(gen_step_two, tar[:,self.params.N_out_channels:2*self.params.N_out_channels])
            valid_loss[i] = loss_step_one + loss_step_two
            valid_l1[i] = nn.functional.l1_loss(gen_step_one, tar[:,0:self.params.N_out_channels])
        else:
            gen = self.model(inp)
            valid_loss[i] += self.loss_obj(gen, tar) 
            valid_l1[i] += nn.functional.l1_loss(gen, tar)

        if self.params.two_step_training:
            for c in range(self.params.N_out_channels):
              if 'residual_field' in self.params.target:
                valid_weighted_rmse[i, c] = weighted_rmse_torch((gen_step_one[0,c] + inp[0,c]), (tar[0,c]+inp[0,c]), self.device)
              else:
                valid_weighted_rmse[i, c] = weighted_rmse_torch(gen_step_one[0,c], tar[0,c], self.device)
        else:
            for c in range(self.params.N_out_channels):
              if 'residual_field' in self.params.target:
                valid_weighted_rmse[i, c] = weighted_rmse_torch((gen[0,c] + inp[0,c]), (tar[0,c]+inp[0,c]), self.device)
              else:
                valid_weighted_rmse[i, c] = weighted_rmse_torch(gen[0,c], tar[0,c], self.device)
            
        #un-normalize
        valid_weighted_rmse = mult*torch.mean(valid_weighted_rmse[0:100], axis = 0).to(self.device)

    return valid_weighted_rmse


  def load_model_wind(self, model_path):
    if self.params.log_to_screen:
      logging.info('Loading the wind model weights from {}'.format(model_path))
    checkpoint = torch.load(model_path, map_location='cuda:{}'.format(self.params.local_rank))
    if dist.is_initialized():
      self.model_wind.load_state_dict(checkpoint['model_state'])
    else:
      new_model_state = OrderedDict()
      model_key = 'model_state' if 'model_state' in checkpoint else 'state_dict'
      for key in checkpoint[model_key].keys():
          if 'module.' in key: # model was stored using ddp which prepends module
              name = str(key[7:])
              new_model_state[name] = checkpoint[model_key][key]
          else:
              new_model_state[key] = checkpoint[model_key][key]
      self.model_wind.load_state_dict(new_model_state)
      self.model_wind.eval()

  def save_checkpoint(self, checkpoint_path, model=None):
    """ We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function """

    if not model:
      model = self.model

    torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                  'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)

  def restore_checkpoint(self, checkpoint_path):
    """ We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function """
    checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank))
    try:
        self.model.load_state_dict(checkpoint['model_state'])
    except:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            new_state_dict[name] = val
        new_state_dict.pop('cont_pos_embed.time_coords', None)
        new_state_dict.pop('cont_pos_embed.lats', None)
        new_state_dict.pop('cont_pos_embed.time_base', None)
        self.model.load_state_dict(new_state_dict)
    self.iters = checkpoint['iters']
    self.startEpoch = checkpoint['epoch']
    if self.params.resuming:  #restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--run_num", default='00', type=str)
  parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
  parser.add_argument("--config", default='default', type=str)
  parser.add_argument("--enable_amp", action='store_true')
  parser.add_argument("--epsilon_factor", default = 0, type = float)
  parser.add_argument("--sweep_id", default=None, type=str, help='sweep config from ./configs/sweeps.yaml')
  #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
  args = parser.parse_args()

  params = YParams(os.path.abspath(args.yaml_config), args.config)
  params['epsilon_factor'] = args.epsilon_factor
  params['config'] = args.config
  params['world_size'] = 1
  if 'WORLD_SIZE' in os.environ:
    params['world_size'] = int(os.environ['WORLD_SIZE'])

  world_rank = 0
  local_rank = 0
  if params['world_size'] > 1:
    dist.init_process_group(backend='nccl',
                            init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    args.gpu = local_rank
    world_rank = dist.get_rank()
    params['global_batch_size'] = params.batch_size
    params['batch_size'] = int(params.batch_size//params['world_size'])

  torch.cuda.set_device(local_rank)
  torch.backends.cudnn.benchmark = True
  #torch.backends.cudnn.enabled = False 
  # Set up directory
  params['upscale'] = 'rescale_factor' in params and params.rescale_factor != params.scale_factor
  
  if args.sweep_id:
    jid = os.environ['SLURM_JOBID'] # so different sweeps dont resume
    expDir = os.path.join(params.exp_dir, args.sweep_id, args.config, str(args.run_num), jid)
  else:
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))

  #expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
  params['old_exp_dir'] = expDir
  # I did this pretty much the worst way possible. Please fix it.
  if params.upscale:
      total_scale = params.scale_factor*params.rescale_factor
      expDir = os.path.join(expDir, 'DSx'+str(int(1/total_scale)))
      # Check to see if smaller rescale path exists
      candPath = os.path.join(expDir, 'DSx'+str(total_scale*2))
      if os.path.isdir(candPath):
          params['old_exp_dir'] = candPath

  if  world_rank==0:
    if not os.path.isdir(expDir):
      os.makedirs(expDir)
      os.makedirs(os.path.join(expDir, 'training_checkpoints/'))

  params['experiment_dir'] = os.path.abspath(expDir)
  params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
  params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
  params['old_checkpoint_path'] = os.path.join(params.old_exp_dir, 'training_checkpoints/best_ckpt.tar')
  # Do not comment this line out please:
  args.resuming = True if os.path.isfile(params.checkpoint_path) else False

  params['resuming'] = args.resuming
  params['local_rank'] = local_rank
  params['enable_amp'] = args.enable_amp

  # this will be the wandb name
#  params['name'] = args.config + '_' + str(args.run_num)
#  params['group'] = "era5_wind" + args.config
  params['name'] = args.config + '_' + str(args.run_num)
  params['group'] = "spectr_" + args.config
  params['project'] = "ssl"
  params['entity'] = "weatherbenching"
  if world_rank==0:
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
    logging_utils.log_versions()
    params.log()

  params['log_to_wandb'] = (world_rank==0) and params['log_to_wandb']
  params['log_to_screen'] = (world_rank==0) and params['log_to_screen']

  params['in_channels'] = np.array(params['in_channels'])
  params['out_channels'] = np.array(params['out_channels'])
  if params.orography:
    params['N_in_channels'] = len(params['in_channels']) +1
  else:
    params['N_in_channels'] = len(params['in_channels'])

  params['N_out_channels'] = len(params['out_channels'])

  if 'init_rollout_length' not in params:
    params['init_rollout_length'] = 1
  params['rollout_length'] = params['init_rollout_length']



  if world_rank == 0:
    hparams = ruamelDict()
    yaml = YAML()
    for key, value in params.params.items():
      hparams[str(key)] = str(value)
    with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
      yaml.dump(hparams,  hpfile )

  trainer = Trainer(params, world_rank)
  if args.sweep_id and trainer.world_rank==0:
    logging.disable(logging.CRITICAL)
    print(args.sweep_id, trainer.params.entity, trainer.params.project)
    wandb.agent(args.sweep_id, function=trainer.train, count=1, entity=trainer.params.entity, project=trainer.params.project) 
  else:
    trainer.train()
  logging.info('DONE ---- rank %d'%world_rank)


