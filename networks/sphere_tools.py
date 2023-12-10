import math
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import time
from e3nn.o3 import spherical_harmonics_alpha_beta
from scipy.special import eval_legendre
from typing import Optional
from torch.nn.common_types import _size_2_t
from torch.profiler import profile, record_function, ProfilerActivity

import os
import os.path as osp
import glob
import numpy as np
import torch
import torch.nn as nn
from scipy.special import gammaln
from math import pi as PI


#####################################
# RANDOM UTILITIES 
######################################

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.utils.spectral_norm(nn.Linear(in_features, hidden_features))
        self.act = act_layer()
        self.fc2 = nn.utils.spectral_norm(nn.Linear(hidden_features, out_features))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def day_length(day_of_year, latitude):
    """ Adapted from here: https://www.dataliftoff.com/plotting-hours-of-daylight-in-python-with-matplotlib/"""
    latitude = latitude.expand(day_of_year.shape[0], -1)
    rval = torch.zeros_like(latitude)
    pi = np.pi
    latitude = torch.rad2deg((latitude-pi/2)) # The source uses lat in degrees
    P = torch.asin(0.39795 * torch.cos(0.2163108 + 2 * torch.atan(0.9671396 * torch.tan(.00860 * (day_of_year - 186)))))
    P = P.expand(-1, latitude.shape[-1])
    inside_arccos = (np.sin(0.8333 * pi / 180) + torch.sin(latitude * pi / 180) * torch.sin(P)) / (torch.cos(latitude * pi / 180) * torch.cos(P))
    
    torch.clamp_(inside_arccos, -1, 1)
    torch.arccos_(inside_arccos)
    rval = 24 - (24 / pi) * inside_arccos
    return (rval - 12) / 12


def glide_reflection(x, flip_dim=-2, glide_dim=-1):
    n_ew = x.shape[glide_dim]
    flipped_x = torch.flip(x, dims=(flip_dim,))
    return torch.roll(flipped_x, shifts=n_ew // 2, dims=glide_dim)


def sphere_to_torus(x, flip_dim=-2, glide_dim=-1):
    """ Performs a sphere to torus mapping for lat/long by reflecting and 
    shifting the input so that both the x and y directions are periodic. 

    Currently assumes (..., c, h, w) format. 
    """
    return torch.cat([x, glide_reflection(x, flip_dim=flip_dim, glide_dim=glide_dim)], dim=flip_dim)

def torus_to_sphere(x, flip_dim=-2, glide_dim=-1):
    b, c, h, w = x.shape
    return .5*(glide_reflection(x[:, :, h//2:])+x[:, :, :h//2]) 

def full_sphere_to_torus(x, flip_dim=-2, glide_dim=-1):
    """ Performs a sphere to torus mapping for lat/long by reflecting and 
    shifting the input so that both the x and y directions are periodic. 

    Currently assumes (..., c, h, w) format. 
    """
    return torch.cat([x, glide_reflection(x, flip_dim=flip_dim, glide_dim=glide_dim)[:,:, 1:-1]], dim=flip_dim)

def sphere_to_torus_rfft(x, flip_dim=1, glide_dim=2):
    xshape = x.shape
    x = torch.fft.rfft(x, dim=glide_dim, norm='ortho')
    x_flip = torch.flip(x, dims=(flip_dim,))
    out_shape = [1 for _ in x.shape]
    out_shape[glide_dim] = -1
    # Implement translation in Fourier space then stack
    trans = torch.exp(.5*2j*np.pi*torch.fft.rfftfreq(xshape[glide_dim], device=x.device)*xshape[glide_dim]).reshape(*out_shape)
    return torch.cat([x, x_flip*trans], dim=flip_dim)

def complex_relu(x):
    x1 = F.relu(x.real)
    x2 = F.relu(x.imag)
    return torch.view_as_complex(torch.stack([x1, x2], dim=-1))

def complex_glu(x):
    x1 = F.glu(x.real)
    x2 = F.glu(x.imag)
    return torch.view_as_complex(torch.stack([x1, x2], dim=-1))

###############################################################3
# Actual modules that should probably also be in outside files for clarity
##################################################################
class ContSphereEmbedding(nn.Module):
    def __init__(self, hidden_features, grid_shape, patch_size=(8, 8), samples_per_year=1460, sph_order=10, max_time_freq=4, layer_size=500, scale_factor=.02,
                 sphere=True, learned_sphere=False, global_dfs=True, steps_per_day=4, steps_per_year=1460, include_pole=False):
        super().__init__()
        self.grid_shape = grid_shape
        self.out_shape = grid_shape
        self.patch_size = patch_size
        self.hidden_features = hidden_features
        self.sphere = sphere
        self.learned_sphere = learned_sphere
        self.global_dfs = global_dfs
        self.sph_order = sph_order
        self.max_time_freq = max_time_freq
        # If operating in space, use spherical harmonics for spatial locations
        if samples_per_year == 1460:
            self.daylight_feats = True
        else:
            self.daylight_feats = False
        
        self.include_pole = include_pole
        #self.scale_factor = scale_factor
        self.samples_per_year = samples_per_year
        self.steps_per_day = steps_per_day
        self.bw_order= sph_order
        num_feats = self.build_features()

        self.mlp = nn.Sequential(nn.Linear(num_feats, layer_size),
            nn.GELU(),
            nn.Linear(layer_size, hidden_features, bias=False),
            )
        self.scale_factor = nn.Parameter(torch.tensor(scale_factor))

    def resize(self, grid_shape):
        self.grid_shape = grid_shape
        self.build_features()

    @torch.no_grad()
    def build_features(self, initial=False):
        # Spherical features assume you're on a sphere
        if self.sphere:
            # If global_dfs, the data is mirrored so we only want to actually compute half
            if self.global_dfs:
                if self.grid_shape[0]/2 % 1 > 1e-5:
                    print('THIS IS FAILING BECAUSE PATCHIFIED REP ISNT DIVISIBLE BY 2')
                else:
                    self.out_shape = self.grid_shape[0] , self.grid_shape[1]
                    # New lat coordinates are widest parts of each patch
                    if self.include_pole:
                        # Grid shape[0] is assumed to be DFSed, so global size is grid[0]/2 * patch[0]
                        ns_axis = torch.cat([torch.linspace(0, np.pi, self.patch_size[0]*self.grid_shape[0]//2+1),
                            torch.linspace(np.pi, 0, self.patch_size[0]*self.grid_shape[0]//2+1)[1:-1]])
                    else:
                        delta = np.pi / (self.patch_size[0]*self.grid_shape[0]//2 +1)
                        ns_axis = np.linspace(delta/2, np.pi-delta/2, self.patch_size[0]*self.grid_shape[0]//2)
                        ns_axis = torch.as_tensor(np.concatenate([ns_axis, ns_axis[::-1]])).float()

                    ns_axis = ns_axis.reshape(-1, self.patch_size[0])
                    ns_axis = ns_axis[torch.arange(ns_axis.shape[0]), torch.sin(ns_axis).max(1).indices]
                    
                    # Since EW is constant, just take middles 
                    ew_axis = torch.linspace(0, 2*np.pi, self.grid_shape[1]*self.patch_size[1]+1)[None, None, :-1]
                    ew_axis = F.avg_pool1d(ew_axis, self.patch_size[1], self.patch_size[1]).squeeze()

                    grid_coords = torch.cartesian_prod(ns_axis, ew_axis)
            else:
                # Note: This is the format for scipy so we're kind of just assuming it's the same because the docs are not helpful in e3nn
                if self.include_pole:
                    ns_coords = torch.linspace(0, np.pi, (self.patch_size[0]*self.grid_shape[0])+1)[:-1]
                else:
                    delta = 1 / (self.patch_size[0]*self.grid_shape[0]+1) / 2
                    ns_coords = torch.linspace(delta, np.pi-delta, (self.patch_size[0]*self.grid_shape[0]))
                ew_coords = torch.linspace(0, 2*np.pi, self.grid_shape[1]*self.patch_size[1]+1)[:-1]
                base_coords = ns_coords, ew_coords
                grid_coords =  torch.cartesian_prod(*base_coords)
                self.default_offsets = torch.cartesian_prod(torch.arange(self.patch_size[0]//2, (self.patch_size[0]*self.grid_shape[0]), self.patch_size[0]),
                            torch.arange(self.patch_size[1]//2, self.patch_size[1]*self.grid_shape[1], self.patch_size[1]))
            if not self.learned_sphere:
                space_features = spherical_harmonics_alpha_beta(list(range(1, self.sph_order+1)), grid_coords[:, 1], grid_coords[:, 0])
            self.register_buffer('time_coords', grid_coords[:, 1].reshape(1, -1)/(2*np.pi), persistent=False) # Format is (n_points, [lat, lon])
            self.register_buffer('lats', grid_coords[:, 0].unsqueeze(0), persistent=False)
            self.register_buffer('time_base', (torch.arange(1, self.max_time_freq+1)*2*np.pi).reshape(1, -1), persistent=False)
            if self.daylight_feats:
                num_feats = 2*self.time_base.shape[1]+1 # Space + time of day + daylight hours
            else:
                num_feats = 4*self.time_base.shape[1] # This is where we removed year features
        # Otherwise assume we're in Fourier space
        else:
            #grid_coords = torch.cartesian_prod(torch.fft.fftfreq( self.grid_shape[0])*self.grid_shape[0], torch.fft.rfftfreq(self.grid_shape[1])*self.grid_shape[1])
            #grid_sgns = torch.sign(grid_coords)
            #grid_coords = torch.log1p(grid_coords.abs()+1e-5)*grid_sgns
            grid_coords = torch.cartesian_prod(torch.fft.fftfreq(self.grid_shape[0]), torch.fft.rfftfreq(self.grid_shape[1]))
            denom = (self.grid_shape[0]//2 + 1)
            base_bw = ((torch.fft.fftfreq(self.grid_shape[0])*self.grid_shape[0]).unsqueeze(1)**2 
                     + (torch.fft.rfftfreq(self.grid_shape[1])*self.grid_shape[1]).unsqueeze(0)**2).sqrt() / denom
            base_bw = base_bw.reshape(-1, 1)
            bw_features = []
            for n in range(1, self.bw_order):
                bw_features.append(1 / (1 + base_bw**(2*n)))
            grid_coords = torch.cat([grid_coords, grid_coords**2, grid_coords**3, grid_coords[:, 1:]**2 + grid_coords[:, :1]**2]+bw_features, dim=-1)

            # TODO take and store std separately
            #grid_coords = (grid_coords - grid_coords.mean(0, keepdims=True)) / (1e-5 + grid_coords.std(0, keepdims=True))
            self.out_shape = torch.fft.fftfreq(self.grid_shape[0]).shape[0], torch.fft.rfftfreq(self.grid_shape[1]).shape[0]
            space_features = grid_coords #torch.stack(stack_feats, -1)
            num_feats = 0
        if self.learned_sphere:
            nabla = ((torch.fft.fftfreq(self.grid_shape[0])).unsqueeze(1)**2
                     + (torch.fft.rfftfreq(self.grid_shape[1])).unsqueeze(0)**2)
            space_features = torch.randn(nabla.shape[0], nabla.shape[1], self.hidden_features // 4, dtype=torch.cfloat)
            space_features = space_features / (1+10000*nabla[:, :, None]) # Diffusion
            space_features = torch.fft.irfft2(space_features.cdouble(), dim=(0, 1), norm='forward').float()
            space_features = (space_features - space_features.mean((0, 1), keepdim=True)) / space_features.std((0, 1), keepdim=True)
            space_features = space_features.reshape(-1, self.hidden_features // 4).contiguous()
            self.space_features = nn.Parameter(space_features)
        else:
            self.register_buffer('space_features', space_features, persistent=False)
        self.space_feats_dim = space_features.shape[-1]
        num_feats += self.space_feats_dim
        return num_feats

    @torch.no_grad()
    def datetime_features(self, dt_ind):
        # Within year - dt_ind is 4*day + hours
        days_per_year = self.samples_per_year / self.steps_per_day
        day = torch.div(dt_ind, self.steps_per_day, rounding_mode='trunc')  # The Jupyter deprecation warnings finally got to me
        #xprod_date = day*self.date_base
        # Within day
        time = (dt_ind % self.steps_per_day) / self.steps_per_day
        # If these are spatial coordinates account for solar time offsets
        if self.sphere:
            lats = self.lats
            time_coords = self.time_coords.expand(time.shape[0], -1) #time_coords[offsets[:, 0], offsets[:, 1]].reshape(-1, 1)
            time = (time_coords + time.expand(-1, time_coords.shape[-1])) % 1
            time = time.squeeze(1)
            if self.daylight_feats:
                with record_function('daylight'):
                    daylight = day_length(day, lats).unsqueeze(-1)
            else:
                daylight = ((day % days_per_year) / days_per_year) * self.time_base
                daylight = torch.cat([torch.sin(daylight), torch.cos(daylight)], -1)
                daylight = daylight.unsqueeze(1).expand(-1, time_coords.shape[1], -1)*torch.cos(self.lats)[:, :, None]
        xprod_time = time.unsqueeze(-1)*self.time_base.unsqueeze(0) 
        coses_time = torch.cos(xprod_time)
        sins_time = torch.sin(xprod_time)
        return [coses_time, sins_time, daylight] # Also add daylight back here

    def forward(self, dt_ind, augs=None):
        # Remember to fix this for larger batches
        dt_ind = dt_ind.reshape(-1, 1)
        if self.sphere:
            space_features = self.space_features
            feats = [space_features.unsqueeze(0).expand(dt_ind.shape[0], -1, -1)]
            with record_function('DT feats'):
                dt_feats = self.datetime_features(dt_ind)
            feats += dt_feats
            feats = torch.cat(feats, -1)
        else:
            feats = self.space_features
        if self.sphere: # Sphere is batch element dependent
            out = self.scale_factor*self.mlp(feats).reshape(dt_ind.shape[0], *self.out_shape, self.hidden_features)
        else: # Frequency is not
            out = self.scale_factor*self.mlp(feats).reshape(1, *self.out_shape, self.hidden_features)
        #if self.sphere and self.global_dfs:
        #    out = sphere_to_torus(out, glide_dim=2, flip_dim=1)
        return out

class FourierInterpFilter(nn.Module):
    def __init__(self, input_size=(1440, 1440), max_lats=720, max_res=640, scale_factor=1, hidden_size=768, grid_res='linear', res_factor=.9, dfs_type='full', include_pole=False):
        """ Combines Fourier resampling and spectral truncation by fixed filter 
        
        Note the fixed filters are current 8th order Butterworth filters (applied to each axis, 2d was complicated
        with different filters at each lat) with latitude-wise truncation based on the data. Basing it on a cosine
        would probably make more sense for us since we're not on the same grid as IFS """

        super().__init__()
        # NS Setup
        if grid_res == 'linear':
            res_fact = 1
        elif grid_res == 'quadratic':
            res_fact = 2/3
        elif grid_res == 'cubic':
            res_fact = .5
        else:
            res_fact = 1

        res_fact = res_factor

        init_res = np.linspace(0, np.pi, max_lats+1)

        max_ns_bw = int(min(res_fact*input_size[0]//2, max_res)) # AA - Unfortunately needs to cutoff based on current resolvables
        max_ew_bw = int(min(max_res, (max_lats+1)*res_fact)) # Shaping - Cuts off based on max resolution
        #max_ns_bw = int(max_res*res_fact)
        self.reduction_factor = 1
        if dfs_type == 'full':
            init_res = np.concatenate([init_res, init_res[1:-1][::-1]])
            init_ns_reses = np.sin(init_res)
            if input_size[0] != 2*max_lats: # Fix for non-DFS later
                self.reduction_factor = int(max_lats/ (input_size[0]/2))
                ns_reses = init_ns_reses.reshape(-1, self.reduction_factor).max(1)
            else:
                ns_reses = init_ns_reses
        ew_filter = torch.ones(input_size[0], input_size[1]//2 + 1)
        for i, res in enumerate(ns_reses):
            ew_filter[i, :int(1+res*max_ew_bw)] = 0.
        ew_filter = ew_filter[None, None, :, :] # Hope this is B C H W
        ew_filter = 1-ew_filter# + torch.randn_like(ew_filter)*1e-4
        ns_filter = torch.ones(input_size[0], input_size[1]//2+1)
        for i, res in enumerate(torch.fft.fftfreq(input_size[0])*input_size[0]):
            cutoff = int(1+max(max_ns_bw**2 - res**2, 0)**.5)
            ns_filter[i, :cutoff] = 0
        ns_filter = ns_filter[None, None, :, :]
        ns_filter= 1-ns_filter# + torch.randn_like(ns_filter)*1e-4
    
        self.apply_zonal = torch.any(ew_filter).item()
        self.apply_2d = torch.any(ns_filter).item()
        self.register_buffer('ew_filter', ew_filter, persistent=False)
        self.register_buffer('ns_filter', ns_filter, persistent=False)
        self.scale_factor = scale_factor
        self.dfs_type = dfs_type
        self.input_size = input_size

    def forward(self, x, augs=None):
        with record_function('clip_interp'):
            dtype = x.dtype
            x = x.float()
            B, C, H, W = x.shape
            h, w = int(H*self.scale_factor), int(W*self.scale_factor)
            # Transform and filter
            if augs is not None and len(augs) > 0:
                ns_roll, _, _, _ = augs
                ns_roll = ns_roll.item() // self.reduction_factor
                ew_filter = torch.roll(self.ew_filter, ns_roll, dims=2)
            else:
                ew_filter = self.ew_filter
            if self.dfs_type != 'full':
                xfft = sphere_to_torus_rfft(x)
            else:
                xfft = torch.fft.rfft(x, dim= -1, norm='ortho')

            xfft = xfft * ew_filter
            xfft = torch.fft.fft(xfft, dim=-2, norm='ortho') * self.ns_filter

        # Resize
            if self.scale_factor != 1:
                mid_fft = xfft.new_zeros(B, C, h, w//2+1)
                mid_fft[:, :, :min(H//2, h//2), :min(W//2+1, w//2+1)] =  xfft[:, :,  :min(H//2, h//2), :min(W//2+1, w//2+1)]
                mid_fft[:, :, -min(H//2, h//2):, :min(W//2+1, w//2+1)] = xfft[:, :,  -min(H//2, h//2):, :min(W//2+1, w//2+1)]
            else:
                mid_fft = xfft
            out = torch.fft.irfft2(mid_fft, norm='ortho').to(dtype=dtype) * self.scale_factor # Each dir gets sqrt(scale_factor)
            if self.dfs_type != 'full':
                out = out[:, :, :out.shape[2]//2]
        return out

def complex_clamp(x, mx=1e-4, mn=None, eps=1e-6):
    out = x.clone()
    x_norms = x.abs()
    out[x_norms > mx] = x[x_norms > mx] / (x_norms[x_norms > mx]+eps) * mx
    return out

class ComplexGroupedMLP(nn.Module):
    def __init__(self, in_features, out_features, groups):
        super().__init__()
        self.block_size = in_features // groups
        self.scale1 = 1 / self.block_size**.5
        base_weight1 = self.scale1 * torch.randn(groups, self.block_size, self.block_size, dtype=torch.cfloat)

        self.weight = nn.Parameter(torch.stack([base_weight1.real, base_weight1.imag], dim=-1))

    def forward(self, x):
        return torch.einsum('...bi,bio->...bo', x, torch.view_as_complex(self.weight))

class ComplexGroupedConv(nn.Module):
    """ Currently not worth it given complex 32 support is extremely weak"""
    def __init__(self, in_features, out_features, groups):
        super().__init__()
        base = nn.Conv2d(in_features, out_features, 1, groups=groups, bias=False, dtype=torch.cfloat)
        self.weight = nn.Parameter(torch.view_as_real(base.weight))
        self.groups = groups

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        x = F.conv2d(x, torch.view_as_complex(self.weight).to(dtype=x.dtype), groups=self.groups)
        return rearrange(x, 'b c h w -> b h w c')
