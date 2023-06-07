import os
import time
import numpy as np
import argparse

from numpy.core.numeric import False_
import h5py
import torch
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
from utils.weighted_acc_rmse import weighted_rmse, weighted_acc, weighted_acc_masked
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from networks.afnonet import AFNONet 
import wandb
from utils.plots import plot
import matplotlib.pyplot as plt
import glob
#from plotting.animate import make_gif
from datetime import datetime
DECORRELATION_TIME = 36 # 9 days


def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return (x + noise)

def autoregressive_inference(params, ic):
    ic = int(ic)
    #get data loader
    valid_data_loader, valid_dataset = get_data_loader(params, params.valid_data_path, dist.is_initialized(), train=False)
    if dist.is_initialized():
        world_size = dist.get_world_size()
        print(world_size)
    else:
        world_size = 1

    #initialize global variables
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    exp_dir = params['experiment_dir'] 
    dt = int(params.dt)
    prediction_length = int(params.prediction_length/dt)
    n_history = params.n_history
    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    crop_size_x = valid_dataset.crop_size_x
    crop_size_y = valid_dataset.crop_size_y
    params.crop_size_x = crop_size_x
    params.crop_size_y = crop_size_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    if params["orography"]:
      params['N_in_channels'] = n_in_channels + 1
    else:
      params['N_in_channels'] = n_in_channels

    params['N_out_channels'] = n_out_channels
    mins = np.load(params.min_path)[0, out_channels]
    maxs = np.load(params.max_path)[0, out_channels]
    means = np.load(params.global_means_path)[0, out_channels]
    stds = np.load(params.global_stds_path)[0, out_channels]
    n_grid_channels = params.N_grid_channels
    orography = params.orography
    orography_path = params.orography_path
    #print(means.shape, stds.shape)
    
    train = False
    
    #load Model weights
    
    if params.log_to_screen:
      logging.info('Loading trained model checkpoint from {}'.format(params['best_checkpoint_path']))

    elif params.nettype == 'fno':
      model = FNO.Net2d(params).to(device)
    elif params.nettype == 'afno':
      model = AFNONet(params).to(device) 

    model.zero_grad()
    checkpoint_file  = params['best_checkpoint_path']
    checkpoint = torch.load(checkpoint_file)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key.replace("module.", "")
            if name != 'ged':
                new_state_dict[name] = val 
        
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    #initialize memory for image sequences and RMSE/ACC
    valid_loss = np.zeros((prediction_length, n_out_channels))
    acc = np.zeros((prediction_length, n_out_channels))
    acc_land = np.zeros((prediction_length, n_out_channels))
    acc_sea = np.zeros((prediction_length, n_out_channels))
    acc_unweighted = np.zeros((prediction_length, n_out_channels))
    seq_real = torch.zeros((prediction_length+ n_history, n_out_channels, img_shape_x, img_shape_y))
    seq_pred = torch.zeros((prediction_length+n_history, n_out_channels, img_shape_x, img_shape_y))


    if params.log_to_screen:
      logging.info('Loading validation data')
    files_paths = glob.glob(params.valid_data_path + "/*.h5")
    files_paths.sort()
    valid_data = h5py.File(files_paths[0], 'r')['fields']
    valid_data = valid_data[ic:(ic+prediction_length*dt+n_history*dt):dt, in_channels, 0:720] #extract valid data from first year
    logging.info(valid_data.shape)
    #normalize
    valid_data = (valid_data -means)/stds

    #autoregressive inference
    if params.log_to_screen:
      logging.info('Begin autoregressive inference')

    if orography:
      orog = torch.as_tensor(np.expand_dims(np.expand_dims(h5py.File(orography_path, 'r')['orog'][0:720], axis = 0), axis = 0)).to(device, dtype = torch.float)
      print("orography loaded; shape:", orog.shape)

    with torch.no_grad():
      for i in range(valid_data.shape[0]): 
        
        if i==0: #start of sequence
          first = torch.as_tensor(valid_data[0:n_history+1]).to(device, dtype = torch.float)
          future = valid_data[n_history+1]
          for h in range(n_history+1):
            seq_real[h] = first[h*n_in_channels : (h+1)*n_in_channels][0:n_out_channels] #extract history from 1st 
            seq_pred[h] = seq_real[h]
          
          if params.perturb:
              first = gaussian_perturb(first, level=params.n_level, device=device) # perturb the ic
          if orography:
              future_pred = model(torch.cat( (first, orog), axis = 1))
          else:
              future_pred = model(first)

        else:
          if i < prediction_length-1:
            future = valid_data[n_history+i+1]
          if orography:
            future_pred = model( torch.cat( (future_pred, orog), axis = 1 )) #autoregressive step
          else:
            future_pred = model(future_pred)

        if i < prediction_length-1: #not on the last step
          
          seq_pred[n_history + i+1] = future_pred
          seq_real[n_history + i+1] = torch.as_tensor(future)
          
          #collect history
          history_stack = seq_pred[i+1:i+2+n_history]

      
        future_pred = history_stack.to(device, dtype = torch.float)
      
        #Compute metrics 
        if i < prediction_length-1:
          
          #RMSE
          for c in range(n_out_channels):
            valid_loss[i+1, c] = weighted_rmse(seq_real[n_history+i+1, c].numpy(), seq_pred[n_history+i+1, c].numpy())

          #un-normalize  
          valid_loss[i+1] *= stds[:, 0, 0]

          #ACC
          #load time means
          tm = np.load(params.time_means_path)[0][out_channels] - means
          m = torch.as_tensor((np.load(params.time_means_path)[0][out_channels] - means)/stds)[:, 0:img_shape_x]

          if params.masked_acc:
              maskarray = np.load(params.maskpath)[0:720]

          for c in range(n_out_channels):
            acc[i+1, c] = weighted_acc((seq_real[n_history+i, c] - m[c]).numpy(), (seq_pred[n_history+i, c]-m[c]).numpy())

            if params.masked_acc:
                acc_land[i+1, c] = weighted_acc_masked((seq_real[n_history+i, c] - m[c]).numpy(), (seq_pred[n_history+i, c]-m[c]).numpy(), True,  maskarray)
                acc_sea[i+1, c] = weighted_acc_masked((seq_real[n_history+i, c] - m[c]).numpy(), (seq_pred[n_history+i, c]-m[c]).numpy(), True, 1-maskarray)

            acc_unweighted[i+1, c] = weighted_acc((seq_real[n_history+i, c] - m[c]).numpy(), (seq_pred[n_history+i, c]-m[c]).numpy(), weighted=False)


        if params.log_to_screen and (i+1) %5 ==0 and i < prediction_length-1:
          logging.info('Predicted timestep {} of {}. u10 RMS Error: {}, ACC: {}'.format((i+1), prediction_length, valid_loss[i+1, 0], acc[i+1, 0]))

        
    if params.masked_acc:
        return np.expand_dims(seq_real[n_history:], 0), np.expand_dims(seq_pred[n_history:], 0), np.expand_dims(valid_loss,0), np.expand_dims(acc, 0), np.expand_dims(acc_unweighted, 0),\
                np.expand_dims(acc_land, 0), np.expand_dims(acc_sea, 0)
    else:
        return np.expand_dims(seq_real[n_history:], 0), np.expand_dims(seq_pred[n_history:], 0), np.expand_dims(valid_loss,0), np.expand_dims(acc, 0), np.expand_dims(acc_unweighted, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--enable_amp", action='store_true')
    parser.add_argument("--epsilon_factor", default = 0, type = float)
    parser.add_argument("--override_dir", default = 'None', type = str)
    

    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['epsilon_factor'] = args.epsilon_factor

    params['world_size'] = 1
    if 'WORLD_SIZE' in os.environ:
      params['world_size'] = int(os.environ['WORLD_SIZE'])

    world_rank = 0
    if params['world_size'] > 1:
      torch.cuda.set_device(args.local_rank)
      dist.init_process_group(backend='nccl',
                              init_method='env://')
      args.gpu = args.local_rank
      world_rank = dist.get_rank()
      params['global_batch_size'] = params.batch_size
      params['batch_size'] = int(params.batch_size//params['world_size'])

    torch.backends.cudnn.benchmark = True

    # Set up directory
    if args.override_dir =='None':
      expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    else:
      expDir = args.override_dir

    if  world_rank==0:
      if not os.path.isdir(expDir):
        os.makedirs(expDir)
        os.makedirs(os.path.join(expDir, 'training_checkpoints/'))

    params['experiment_dir'] = os.path.abspath(expDir)
    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

    args.resuming = False

    params['resuming'] = args.resuming
    params['local_rank'] = args.local_rank
    params['enable_amp'] = args.enable_amp

    # this will be the wandb name
    params['name'] = args.config + '_' + str(args.run_num)
    params['group'] = args.config
    if world_rank==0:
      logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference_out.log'))
      logging_utils.log_versions()
      params.log()

    params['log_to_wandb'] = (world_rank==0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank==0) and params['log_to_screen']


    n_ics = params['n_initial_conditions']
#    ics = np.linspace(0, (DECORRELATION_TIME+ params['prediction_length'])*n_ics, n_ics)
    #t0 = int((276/365)*(1460))
    #ics = [t0-10, t0-5, t0, t0 + 5, t0+10]

    if params["ics_type"] == 'default':
#        ics = np.linspace(0, (DECORRELATION_TIME+ params['prediction_length'])*n_ics, n_ics)
        ics = np.linspace(0, (DECORRELATION_TIME)*n_ics, n_ics)
        n_ics = params['n_initial_conditions']
    elif params["ics_type"] == "datetime":
        date_strings = params["date_strings"]
        ics = []
        if params.perturb: #for perturbations use a single date and create n_ics perturbations
            n_ics = params["n_perturbations"]
            date = date_strings[0]
            date_obj = datetime.strptime(date,'%Y-%m-%d %H:%M:%S') 
            day_of_year = date_obj.timetuple().tm_yday - 1
            hour_of_day = date_obj.timetuple().tm_hour
            hours_since_jan_01_epoch = 24*day_of_year + hour_of_day
            for ii in range(n_ics):
                ics.append(int(hours_since_jan_01_epoch/6))
        else:
            for date in date_strings:
                date_obj = datetime.strptime(date,'%Y-%m-%d %H:%M:%S') 
                day_of_year = date_obj.timetuple().tm_yday - 1
                hour_of_day = date_obj.timetuple().tm_hour
                hours_since_jan_01_epoch = 24*day_of_year + hour_of_day
                ics.append(int(hours_since_jan_01_epoch/6))
        print(ics)
        n_ics = len(ics)

    try:
        autoregressive_inference_filetag = params["inference_file_tag"]
    except:
        autoregressive_inference_filetag = ""

    #initialize lists for image sequences and RMSE/ACC
    valid_loss = np.zeros
    acc_unweighted = []
    acc = []
    acc_land = []
    acc_sea = []
    seq_pred = []
    seq_real = []

    #run autoregressive inference for multiple initial conditions
    for i, ic in enumerate(ics):
      logging.info("Initial condition {} of {}".format(i+1, n_ics))
      if params.masked_acc:
        sr, sp, vl, a, au, aland, asea = autoregressive_inference(params, ic)
      else:
        sr, sp, vl, a, au = autoregressive_inference(params, ic)
      if i ==0:
        if params.save_channel:
          seq_real = sr[:,:,params.save_idx:params.save_idx+1]
          seq_pred = sp[:,:,params.save_idx:params.save_idx+1]
        else:
          seq_real = sr
          seq_pred = sp
        valid_loss = vl
        acc = a
        acc_unweighted = au
        if params.masked_acc:
          acc_land = aland
          acc_sea = asea
      else:
        if params.save_raw_forecasts:
            seq_real = np.concatenate((seq_real, sr), 0)
            seq_pred = np.concatenate((seq_pred, sp), 0)
        elif params.save_channel:
            save_idx = params.save_idx
            print("seq pred shape", seq_pred.shape)
            seq_real = np.concatenate((seq_real, sr[:,:,save_idx:save_idx+1]), 0)
            seq_pred = np.concatenate((seq_pred, sp[:,:,save_idx:save_idx+1]), 0)

        valid_loss = np.concatenate((valid_loss, vl), 0)
        acc = np.concatenate((acc, a), 0)
        acc_unweighted = np.concatenate((acc_unweighted, au), 0)
        if params.masked_acc:
          acc_land = np.concatenate((acc_land, aland), 0)
          acc_sea = np.concatenate((acc_sea, asea), 0)


    prediction_length = seq_real[0].shape[0]
    n_out_channels = seq_real[0].shape[1]
    img_shape_x = seq_real[0].shape[2]
    img_shape_y = seq_real[0].shape[3]

    #save predictions and loss
    if params.log_to_screen:
      logging.info("Saving files at {}".format(os.path.join(params['experiment_dir'], 'autoregressive_predictions' + autoregressive_inference_filetag + '.h5')))
    with h5py.File(os.path.join(params['experiment_dir'], 'autoregressive_predictions'+ autoregressive_inference_filetag +'.h5'), 'a') as f:
      if params.save_raw_forecasts:
          try:
            f.create_dataset("ground_truth", data = seq_real, shape = (n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
          except: 
            del f["ground_truth"]
            f.create_dataset("ground_truth", data = seq_real, shape = (n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
            f["ground_truth"][...] = seq_real
          
          try:
            f.create_dataset("predicted", data = seq_pred, shape = (n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
          except:
            del f["predicted"]
            f.create_dataset("predicted", data = seq_pred, shape = (n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
            f["predicted"][...]= seq_pred
      try:
        f.create_dataset("rmse", data = valid_loss)#, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
      except:
        del f["rmse"]
        f.create_dataset("rmse", data = valid_loss)#, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
        f["rmse"][...] = valid_loss

      try:
        f.create_dataset("acc", data = acc)#, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
      except:
        del f["acc"]
        f.create_dataset("acc", data = acc)#, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
        f["acc"][...] = acc   

      if params.masked_acc:
        try:
          f.create_dataset("acc_land", data = acc_land)#, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
        except:
          del f["acc_land"]
          f.create_dataset("acc_land", data = acc_land)#, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
          f["acc_land"][...] = acc_land  

        try:
          f.create_dataset("acc_sea", data = acc_sea)#, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
        except:
          del f["acc_sea"]
          f.create_dataset("acc_sea", data = acc_sea)#, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
          f["acc_sea"][...] = acc_sea 

      try:
        f.create_dataset("acc_unweighted", data = acc_unweighted)#, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
      except:
        del f["acc_unweighted"]
        f.create_dataset("acc_unweighted", data = acc_unweighted)#, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
        f["acc_unweighted"][...] = acc_unweighted     
        
      f.close()


    if params.save_channel:
      save_idx = params.save_idx
      with h5py.File(os.path.join(params['experiment_dir'], 'autoregressive_predictions_idx' + str(save_idx) + autoregressive_inference_filetag +'.h5'), 'a') as f:
        print("seq_real shape", seq_real.shape)
        try:
          f.create_dataset("ground_truth", data = seq_real[:,:,0:1])#, shape = (n_ics, prediction_length, 1, img_shape_x, img_shape_y), dtype = np.float32)
        except: 
          del f["ground_truth"]
          f.create_dataset("ground_truth", data = seq_real[:,:,0:1])#, shape = (n_ics, prediction_length, 1, img_shape_x, img_shape_y), dtype = np.float32)
          f["ground_truth"][...] = seq_real[:,:,0:1]
        
        try:
          f.create_dataset("predicted", data = seq_pred[:,:,0:1])#, shape = (n_ics, prediction_length, 1, img_shape_x, img_shape_y), dtype = np.float32)
        except:
          del f["predicted"]
          f.create_dataset("predicted", data = seq_pred[:,:,0:1])#, shape = (n_ics, prediction_length, 1, img_shape_x, img_shape_y), dtype = np.float32)
          f["predicted"][...]= seq_pred[:,:,0:1]


