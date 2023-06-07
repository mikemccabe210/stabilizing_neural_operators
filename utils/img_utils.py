import logging
import glob
from types import new_class
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py
import math
import torchvision.transforms.functional as TF
import matplotlib
import matplotlib.pyplot as plt

class PeriodicPad2d(nn.Module):
    """ 
        pad longitudinal (left-right) circular 
        and pad latitude (top-bottom) with zeros
    """
    def __init__(self, pad_width):
       super(PeriodicPad2d, self).__init__()
       self.pad_width = pad_width

    def forward(self, x):
        # pad left and right circular
        out = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular") 
        # pad top and bottom zeros
        out = F.pad(out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0) 
        return out

def reshape_fields(img, inp_or_tar, crop_size_x, crop_size_y,rnd_x, rnd_y, params, ns_roll, ew_roll,
        ns_flip, ew_flip, train, normalize=True, orog=None, norm_arrays=(None, None), rollout_length=1):
    #Takes in np array of size (n_history+1, c, h, w) and returns torch tensor of size ((n_channels*(n_history+1), crop_size_x, crop_size_y)
    
    if len(np.shape(img)) ==3:
      img = np.expand_dims(img, 0)

    # Note - removed part truncating to 721
    n_history = np.shape(img)[0] - 1
    img_shape_x = np.shape(img)[-2]
    img_shape_y = np.shape(img)[-1]
    n_channels = np.shape(img)[1] #this will either be N_in_channels or N_out_channels
    channels = params.in_channels if inp_or_tar =='inp' else params.out_channels

    if crop_size_x == None:
        crop_size_x = img_shape_x
    if crop_size_y == None:
        crop_size_y = img_shape_y

    if normalize:
        if params.normalization == 'minmax':
          maxs, mins = norm_arrays
          img  -= mins
          img /= (maxs - mins)
        elif params.normalization == 'zscore':
          means, stds = norm_arrays
          img -=means
          img /=stds

    if train and (crop_size_x or crop_size_y):
        img = img[:,:,rnd_x:rnd_x+crop_size_x, rnd_y:rnd_y+crop_size_y]

    # No idea if this part works
    if params.add_grid:
        if inp_or_tar == 'inp':
            if params.gridtype == 'linear':
                assert params.N_grid_channels == 2, "N_grid_channels must be set to 2 for gridtype linear"
                x = np.meshgrid(np.linspace(-1, 1, img_shape_x))
                y = np.meshgrid(np.linspace(-1, 1, img_shape_y))
                grid_x, grid_y = np.meshgrid(y, x)
                grid = np.stack((grid_x, grid_y), axis = 0)
            elif params.gridtype == 'sinusoidal':
                assert params.N_grid_channels == 4, "N_grid_channels must be set to 4 for gridtype sinusoidal"
                x1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_x)))
                x2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_x)))
                y1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_y)))
                y2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_y)))
                grid_x1, grid_y1 = np.meshgrid(y1, x1)
                grid_x2, grid_y2 = np.meshgrid(y2, x2)
                grid = np.expand_dims(np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis = 0), axis = 0)
            img = np.concatenate((img, grid), axis = 1 )

    if params.orography and inp_or_tar == 'inp':
        img = np.concatenate((img, np.expand_dims(orog, axis = (0,1) )), axis = 1)
        n_channels += 1

    if inp_or_tar == 'inp':
        img = np.reshape(img, (n_channels*(n_history+1), crop_size_x, crop_size_y))
    elif inp_or_tar == 'tar':
        img = np.reshape(img, (rollout_length*n_channels, crop_size_x, crop_size_y))
    if (((inp_or_tar == 'inp' or train) and params.dfs_type == 'full') or params.grid_res != 'full') and False:
        img = sphere_to_torus_np(img)
        # flips NS winds on the flipped side
        # img[params.v_channels, img.shape[1] // 2+1:] = -img[params.v_channels, img.shape[2] // 2+1:]
        if params.dfs_type != 'full' or (inp_or_tar == 'tar' and not train):
            img = img[:, :img.shape[1]//2]
            pass # Crop back to normal size
        # Leave at 721 for DFS Full
        else:
            img = img[:, :img.shape[1]//2+1]

    # Do augmentations if necessary - Wrong place now that its truncating!
    if ns_roll:
        img = np.roll(img, ns_roll, axis = -2)
    if ew_roll:
        img = np.roll(img, ew_roll, axis=-1)

    # Not implemented - if you do implement this remember to flip signs for directional variables (ie, u/v wind) according to flip type
    #if ns_flip:
    #    img = np.flip(img, dim=2)
    #    img[:, params.v_channels] = -img[:, params.v_channels]
    #if ew_flip:
    #    img = np.flip(img, dim=3)
    #    img[:, params.u_channels] = -img[:, params.u_channels]

    

    return torch.as_tensor(img)
          
def reshape_precip(img, inp_or_tar, crop_size_x, crop_size_y,rnd_x, rnd_y, params, y_roll, train, normalize=True):

    if len(np.shape(img)) ==2:
      img = np.expand_dims(img, 0)

    img = img[:,:720,:]
    img_shape_x = img.shape[-2]
    img_shape_y = img.shape[-1]
    n_channels = 1
    if crop_size_x == None:
        crop_size_x = img_shape_x
    if crop_size_y == None:
        crop_size_y = img_shape_y

    if normalize:
        eps = params.precip_eps
        img = np.log1p(img/eps)
    if params.add_grid:
        if inp_or_tar == 'inp':
            if params.gridtype == 'linear':
                assert params.N_grid_channels == 2, "N_grid_channels must be set to 2 for gridtype linear"
                x = np.meshgrid(np.linspace(-1, 1, img_shape_x))
                y = np.meshgrid(np.linspace(-1, 1, img_shape_y))
                grid_x, grid_y = np.meshgrid(y, x)
                grid = np.stack((grid_x, grid_y), axis = 0)
            elif params.gridtype == 'sinusoidal':
                assert params.N_grid_channels == 4, "N_grid_channels must be set to 4 for gridtype sinusoidal"
                x1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_x)))
                x2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_x)))
                y1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_y)))
                y2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_y)))
                grid_x1, grid_y1 = np.meshgrid(y1, x1)
                grid_x2, grid_y2 = np.meshgrid(y2, x2)
                grid = np.expand_dims(np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis = 0), axis = 0)
            img = np.concatenate((img, grid), axis = 1 )

    if params.roll:
        img = np.roll(img, y_roll, axis = -1)

    if train and (crop_size_x or crop_size_y):
        img = img[:,rnd_x:rnd_x+crop_size_x, rnd_y:rnd_y+crop_size_y]

    img = np.reshape(img, (n_channels, crop_size_x, crop_size_y))
    return torch.as_tensor(img)


def vis_precip(fields):
    pred, tar = fields
    fig, ax = plt.subplots(1, 2, figsize=(24,12))
    ax[0].imshow(pred, cmap="coolwarm")
    ax[0].set_title("tp pred")
    ax[1].imshow(tar, cmap="coolwarm")
    ax[1].set_title("tp tar")
    fig.tight_layout()
    return fig

def vis_swe(fields):
    pred, tar = fields
    fig, ax = plt.subplots(3, 2, figsize=(24,24))
    im = ax[0][0].imshow(pred[0], cmap="turbo")
    ax[0][0].set_title("u pred")
    plt.colorbar(im,ax=ax[0][0])
    
    im = ax[0][1].imshow(tar[0], cmap="turbo")
    ax[0][1].set_title("u tar")
    plt.colorbar(im,ax=ax[0][1])

    im = ax[1][0].imshow(pred[1], cmap="turbo")
    ax[1][0].set_title("v pred")
    plt.colorbar(im,ax=ax[1][0])

    im = ax[1][1].imshow(tar[1], cmap="turbo")
    ax[1][1].set_title("v tar")
    plt.colorbar(im,ax=ax[1][1])

    im = ax[2][0].imshow(pred[2], cmap="turbo")
    ax[2][0].set_title("h pred")
    plt.colorbar(im,ax=ax[2][0])

    im = ax[2][1].imshow(tar[2], cmap="turbo")
    ax[2][1].set_title("h tar")
    plt.colorbar(im,ax=ax[2][1])

    fig.tight_layout()
    return fig



def glide_reflection_np(x, flip_dim=-2, glide_dim=-1):
    n_ew = x.shape[glide_dim]
    flipped_x = np.flip(x, axis=(flip_dim,))
    return np.roll(flipped_x, shift=n_ew // 2, axis=glide_dim)


def sphere_to_torus_np(x, flip_dim=-2, glide_dim=-1):
    """ Performs a sphere to torus mapping for lat/long by reflecting and 
    shifting the input so that both the x and y directions are periodic. 

    Currently assumes (..., c, h, w) format. 
    """
    return np.concatenate([x, glide_reflection_np(x, flip_dim=flip_dim, glide_dim=glide_dim)[:, 1:-1]], axis=flip_dim)
    

