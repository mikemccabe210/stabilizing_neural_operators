import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py
import math
#import cv2
from utils.img_utils import reshape_fields, reshape_precip, sphere_to_torus_np, glide_reflection_np

def get_data_loader(params, files_pattern, distributed, train, long=False):
  dataset = GetDataset(params, files_pattern, train, long)
  seed = torch.random.seed() if train else 0
  sampler = DistributedSampler(dataset, shuffle=train, seed=seed) if distributed else None
  
  dataloader = DataLoader(dataset,
                          batch_size=int(params.batch_size),
                          num_workers=params.num_data_workers,
                          shuffle=False, #(sampler is None),
                          sampler=sampler, # Since validation is on a subset, use a fixed random subset,
                          drop_last=True,
                          pin_memory=torch.cuda.is_available())

  if train:
    return dataloader, dataset, sampler
  else:
    return dataloader, dataset

class GetDataset(Dataset):
  def __init__(self, params, location, train, long=False):
    self.params = params
    self.location = location
    self.train = train
    self.long = long
    self.dt = params.dt
    self.n_history = params.n_history
    self.in_channels = np.array(params.in_channels)
    self.out_channels = np.array(params.out_channels)
    self.n_in_channels = len(self.in_channels)
    self.n_out_channels = len(self.out_channels)
    self.crop_size_ns = params.crop_size_x
    self.crop_size_ew = params.crop_size_y
    self.roll = params.roll
    self.flip = params.flip
    self._get_files_stats()
    if train:
        self.rollout_length = params.rollout_length
    elif long:
        self.rollout_length = params.long_valid_rollout_length
    else:
        self.rollout_length = params.valid_rollout_length
    self.two_step_training = params.two_step_training
    self.orography = params.orography
    self.precip = True if "precip" in params else False
    if self.precip:
        path = params.precip+'/train' if train else params.precip+'/test'
        self.precip_paths = glob.glob(path + "/*.h5")
        self.precip_paths.sort()

    try:
        self.normalize = params.normalize
    except:
        self.normalize = True #by default turn on normalization if not specified in config

    self.norm_inp= (None, None)
    self.norm_tar = (None, None)
    if params.normalization == 'zscore':
        means = np.load(params.global_means_path)#[:, channels]
        stds = np.load(params.global_stds_path)#[:, channels]
        self.norm_inp = (means[:, params.in_channels], stds[:, params.in_channels])
        self.norm_tar = (means[:, params.out_channels], stds[:, params.out_channels])
    elif params.normalization == 'minmax':
        mins = np.load(params.min_path)#[:, channels]
        maxs = np.load(params.max_path)#[:, channels]
        self.norm_inp = (maxs[:, params.in_channels], mins[:, params.in_channels])
        self.norm_tar = (maxs[:, params.out_channels], mins[:, params.out_channels])

    if self.orography:
      self.orography_path = params.orography_path

  def _get_files_stats(self):
    self.files_paths = glob.glob(self.location + "/*.h5")
    self.files_paths.sort()
    # 1999 is broken. Remove this if it gets fixed
    self.files_paths = [name for name in self.files_paths if '1999' not in name]
    self.n_years = len(self.files_paths)

    with h5py.File(self.files_paths[0], 'r') as _f:
        logging.info("Getting file stats from {}".format(self.files_paths[0]))
        self.n_samples_per_year = _f['fields'].shape[0]
        #original image shape (before padding)
        self.img_shape_ns = _f['fields'].shape[2] 
        if self.params.config != 'afno_swe':
            self.img_shape_ns -= 1 #just get rid of one of the pixels
        self.img_shape_ew = _f['fields'].shape[3]

    self.n_samples_total = self.n_years * self.n_samples_per_year
    self.files = [None for _ in range(self.n_years)]
    self.precip_files = [None for _ in range(self.n_years)]
    logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
    logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location, self.n_samples_total, self.img_shape_ns, self.img_shape_ew, self.n_in_channels))
    logging.info("Delta t: {} hours".format(6*self.dt))
    logging.info("Including {} hours of past history in training at a frequency of {} hours".format(6*self.dt*self.n_history, 6*self.dt))


  def _open_file(self, year_idx):
    _file = h5py.File(self.files_paths[year_idx], 'r')
    self.files[year_idx] = _file['fields']  
    if self.orography:
      _orog_file = h5py.File(self.orography_path, 'r')
      self.orography_field = _orog_file['orog']
    if self.precip:
      self.precip_files[year_idx] = h5py.File(self.precip_paths[year_idx], 'r')['tp']
    
  
  def __len__(self):
    return self.n_samples_total


  def __getitem__(self, global_idx):
    year_idx = int(global_idx/self.n_samples_per_year) #which year we are on
    local_idx = int(global_idx%self.n_samples_per_year) #which sample in that year we are on - determines indices for centering
    k = self.rollout_length


    #open image file
    if self.files[year_idx] is None:
        self._open_file(year_idx)

    if not self.precip:
      #if we are not at least self.dt*n_history timesteps into the prediction
      if local_idx < self.dt*self.n_history:
          local_idx += self.dt*self.n_history

      #if we are on the last image in a year predict identity, else predict next timestep
      # step = 0 if local_idx >= self.n_samples_per_year-self.dt else self.dt
    else:
      inp_local_idx = local_idx
      tar_local_idx = local_idx
      #if we are on the last image in a year predict identity, else predict next timestep
      step = 0 if tar_local_idx >= self.n_samples_per_year-self.dt else self.dt
      # first year has 2 missing samples in precip (they are first two time points)
      if year_idx == 0:
        lim = 1458
        local_idx = local_idx%lim 
        inp_local_idx = local_idx + 2
        tar_local_idx = local_idx
        step = 0 if tar_local_idx >= lim-self.dt else self.dt

    #if two_step_training flag is true then ensure that local_idx is not the last or last but one sample in a year
    if local_idx >= self.n_samples_per_year - k*self.dt:
        #set local_idx to last possible sample in a year that allows taking two steps forward
        local_idx = self.n_samples_per_year - (k+1)*self.dt
    step = self.dt

    if self.train and self.roll:
      ew_roll = random.randint(0, self.img_shape_ew)
      if self.params.dfs_type == 'full':
        ns_roll = random.randint(0, self.img_shape_ns)
      else:
        ns_roll = 0
    else:
      ew_roll = 0
      ns_roll = 0

    if self.train and self.flip:
      ew_flip = int(random.random() < .5)
      ns_flip = int(random.random() < .5)
    else:
      ew_flip = 0
      ns_flip = 0


    if self.orography:
        orog = self.orography_field[0:720] 
    else:
        orog = None
    # Cropping doesn't really make sense, but leaving the code here since it's not used - Mike
    if self.train and (self.crop_size_ns or self.crop_size_ew):
      rnd_x = random.randint(0, self.img_shape_ns-self.crop_size_ns)
      rnd_y = random.randint(0, self.img_shape_ew-self.crop_size_ew)    
    else: 
      rnd_x = 0
      rnd_y = 0
      
    if self.precip:
      return reshape_fields(self.files[year_idx][inp_local_idx, self.in_channels], 'inp', self.crop_size_ns, self.crop_size_ew, rnd_x, rnd_y,self.params, y_roll, self.train), \
                reshape_precip(self.precip_files[year_idx][tar_local_idx+step], 'tar', self.crop_size_ns, self.crop_size_ew, rnd_x, rnd_y, self.params, y_roll, self.train)
    else:
        return reshape_fields(self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels],
                'inp', self.crop_size_ns, self.crop_size_ew, rnd_x, rnd_y,self.params, ns_roll, ew_roll,
                self.train, self.normalize, orog, norm_arrays=self.norm_inp, rollout_length=self.rollout_length), \
                    reshape_fields(self.files[year_idx][local_idx + step:local_idx + step + k, self.out_channels], 'tar', self.crop_size_ns,
                            self.crop_size_ew, rnd_x, rnd_y, self.params, ns_roll, ew_roll, ns_flip,
                            ew_flip, self.train, self.normalize, orog, norm_arrays=self.norm_tar, rollout_length=self.rollout_length), year_idx, local_idx, (ns_roll, ew_roll, ns_flip, ew_flip)






    


  
    


