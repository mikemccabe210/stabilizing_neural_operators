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
#from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# from utils.img_utils import PeriodicPad2d
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

#### 
# SPHERE THINGS
####


sqrt2 = np.sqrt(2.0);

## Change this to the absolute path of your cache folder
cacheDir = '/pscratch/sd/m/mmccabe/cache'

##########################
######## Cache ###########
##########################
def clearCache(cacheDir=cacheDir):

    cFiles = glob.glob(osp.join(cacheDir, '*.pt'));

    for l in range(len(cFiles)):

        os.remove(cFiles[l]);


    return 1;


#############################
######## Utilities ##########
#############################


## Driscoll-Healy Sampling Grid ##
# Input: Bandlimit B
# Output: "Meshgrid" spherical coordinates (theta, phi) of 2B X 2B Driscoll-Healy spherical grid
# These correspond to a Y-Z spherical coordinate parameterzation:
# [X, Y, Z] = [cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)]

def gridDH(B):

    k = torch.arange(0, 2*B).double();

    theta = 2*PI*k / (2*B)
    phi = PI*(2*k + 1) / (4*B);

    theta, phi = torch.meshgrid(theta, phi, indexing='ij');

    return theta, phi;


###########################################################
################ Discrete Legendre Transform ##############
###########################################################

## Recursive computation of d^l_mn ( pi/2)

def triHalfRecur(l, m, n):

    denom = (-1 + l)*np.sqrt( (l-m)*(l+m)*(l-n)*(l+n) );

    c1 = (1 - 2*l)*m*n/denom;

    c2 = -1.0*l*np.sqrt( ( (l-1)*(l-1) - m*m ) * ( (l-1)*(l-1) - n*n ) ) / denom;

    return c1, c2;

def generateLittleHalf(B):

    fName = osp.join(cacheDir, 'littleHalf_{}.pt'.format(B));

    if (osp.isfile(fName) == False):

        #m, n -> m + (B-1), n + (B-1)

        d = torch.empty( B, 2*B - 1, 2*B - 1).double().fill_(0);

        # Fill first two levels (l = 0, l = 1)
        d[0, B-1, B-1] = 1

        d[1, -1 +(B-1), -1 + (B-1)] = 0.5;
        d[1, -1 +(B-1), B-1] = 1.0 / sqrt2;
        d[1, -1 +(B-1), B] = 0.5;

        d[1, (B-1), -1 + (B-1)] = -1.0 / sqrt2;
        d[1, (B-1), B] = 1.0 / sqrt2;

        d[1, B, -1 + (B-1)] = 0.5;
        d[1, B, (B-1)] = -1.0 / sqrt2;
        d[1, B, B] = 0.5;

        ## Fill rest of values through Kostelec-Rockmore recursion

        for l in range(2, B):
            for m in range(0, l):
                for n in range(0, l):

                    if ( (m == 0) and (n == 0) ):

                        d[l, B-1, B-1] = -1.0*( (l-1)/l )*d[l-2, B-1, B-1];

                    else:
                        c1, c2 = triHalfRecur(l, m, n);

                        d[l, m + (B-1), n + (B-1)]= c1 * d[l-1, m + (B-1), n + (B-1)] + c2 * d[l-2, m+(B-1), n+(B-1)];

            for m in range(0, l+1):

                lnV = 0.5*( gammaln(2*l + 1) - gammaln(l+m +1) - gammaln(l-m + 1) ) - l*np.log(2.0);

                d[l, m+(B-1), l+(B-1)] = np.exp(lnV);
                d[l, l+(B-1), m+(B-1)] = np.power(-1.0, l - m) * np.exp(lnV);


            for m in range(0, l+1):
                for n in range(0, l+1):

                    val = d[l, m+(B-1), n+(B-1)]

                    if ( (m != 0) or (n != 0) ):

                        d[l, -m + (B-1), -n + (B-1)] = np.power(-1.0, m-n)*val;
                        d[l, -m + (B-1), n + (B-1)] = np.power(-1.0, l-n)*val;
                        d[l, m+(B-1), -n + (B-1) ] = np.power(-1.0, l+m)*val;



        torch.save(d, fName)

        print('Computed littleHalf_{}'.format(B), flush=True);

    else:

        d = torch.load(fName);


    return d;

def dltWeightsDH(B):

    fName = osp.join(cacheDir, 'dltWeights_{}.pt'.format(B));

    if (osp.isfile(fName) == False):

        W = torch.empty(2*B).double().fill_(0);

        for k in range(0, 2*B):

            C = (2.0/B)*np.sin( PI*(2*k + 1) / (4.0*B) );

            wk = 0.0;

            for p in range(0, B):

                wk += (1.0 / (2*p + 1) ) * np.sin( (2*k + 1)*(2*p + 1) * PI / (4.0 * B));

            W[k] = C * wk;

        torch.save(W, fName);

        print('Computed dltWeights_{}'.format(B), flush=True);

    else:

        W = torch.load(fName);

    return W;

## Inverse (orthogonal) DCT Matrix of dimension N x N
def idctMatrix(N):

    fName = osp.join(cacheDir, 'idctMatrix_{}.pt'.format(N));

    if (osp.isfile(fName) == False):

        DI = torch.empty(N, N).double().fill_(0);

        for k in range(0, N):
            for n in range(0, N):

                DI[k, n] = np.cos(PI*n*(k + 0.5)/N)

        DI[:, 0] = DI[:, 0] * np.sqrt(1.0 / N);
        DI[:, 1:] = DI[:, 1:] * np.sqrt(2.0 / N);

        torch.save(DI, fName);

        print('Computed idctMatrix_{}'.format(N), flush=True);


    else:

        DI = torch.load(fName);

    return DI;

## Inverse (orthogonal) DST Matrix of dimension N x N
def idstMatrix(N):

    fName = osp.join(cacheDir, 'idstMatrix_{}.pt'.format(N));

    if (osp.isfile(fName) == False):

        DI = torch.empty(N, N).double().fill_(0);

        for k in range(0, N):
            for n in range(0, N):

                if (n == (N-1) ):
                    DI[k, n] = np.power(-1.0, k);
                else:
                    DI[k, n] = np.sin(PI*(n+1)*(k + 0.5)/N);

        DI[:, N-1] = DI[:, N-1] * np.sqrt(1.0 / N);
        DI[:, :(N-1)] = DI[:, :(N-1)] * np.sqrt(2.0 / N);

        torch.save(DI, fName);

        print('Computed idstMatrix_{}'.format(N), flush=True);


    else:

        DI = torch.load(fName);

    return DI;

# Normalization coeffs for m-th frequency (C_m)
def normCm(B):

    fName = osp.join(cacheDir, 'normCm_{}.pt'.format(B));

    if (osp.isfile(fName) == False):

        Cm = torch.empty(B).double().fill_(0);

        for m in range(B):
            Cm[m] = np.power(-1.0, m) * np.sqrt(1.0/(2.0*PI));

        torch.save(Cm, fName);

        print('Computed normCm_{}'.format(B), flush=True);


    else:
        Cm = torch.load(fName);

    return Cm;

# Computes sparse matrix of Wigner-d function cosine + sine series coefficients
def wignerCoeffs(B):

    fName = osp.join(cacheDir, 'wignerCoeffs_{}.pt'.format(B));

    if (osp.isfile(fName) == False):

        d = generateLittleHalf(B).cpu().numpy();

        H = 0;
        W = 0;

        indH = [];
        indW = [];
        val = [];

        N = 2*B;
        blocks = []
        for m in range(0, B):
            block = np.zeros((B, N))
            for l in range(np.absolute(m), B):

                for n in range(0, l+1):

                    iH = l + H;
                    iW = n + W;

                    # Cosine series
                    if ( (m % 2) == 0 ):

                        if (n == 0):
                            c = np.sqrt( (2*l + 1)/2.0 ) * np.sqrt( N );
                        else:
                            c = np.sqrt( (2*l + 1)/2.0 ) * np.sqrt( 2.0*N );

                        if ( (m % 4) == 2 ):

                            c *= -1.0;

                        coeff = c * d[l, n + (B-1), -m + (B-1)] * d[l, n+(B-1), B-1];

                    # Sine series
                    else:

                        if (n == l):
                            coeff = 0.0;

                        else:

                            c = np.sqrt( (2*l + 1) / 2.0 ) * np.sqrt( 2.0 * N);

                            if ( (m % 4) == 1 ):
                                c *= -1.0;

                            coeff = c * d[l, (n+1) + (B-1), -m + (B-1)] * d[l, (n+1) + (B-1), B-1];


                    if ( np.absolute(coeff) > 1.0e-15 ):

                        # indH.append(iH);
                        # indW.append(iW);
                        # val.append(coeff);
                        block[iH, iW] = coeff

            blocks.append(block)
            # H += B;
            # W += N;

        # Cat indices, turn into sparse matrix
#         ind = torch.cat( (torch.tensor(indH).long()[None, :], torch.tensor(indW).long()[None, :]), dim=0);
#         val = torch.tensor( val, dtype=torch.double );

#         D = torch.sparse_coo_tensor(ind, val, [B*(B), 2*B*(B)])
        D = torch.as_tensor(np.stack(blocks))
        torch.save(D, fName);

        print('Computed wignerCoeffs_{}'.format(B), flush=True);



    else:

        D = torch.load(fName);

    return D;

# Weighted DCT and DST implemented as linear layers
# Adapted from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
class weightedDCST(nn.Module):
    '''DCT or DST as a linear layer'''

    def __init__(self, B, xform):
        super(weightedDCST, self).__init__()

        self.xform = xform
        self.B = B

        if (self.xform == 'c'):
            W = torch.diag(dltWeightsDH(B))
            XF = torch.matmul(W, idctMatrix(2*B))

        elif(self.xform == 'ic'):
            XF = idctMatrix(2*B).t()

        elif(self.xform == 's'):
            W = torch.diag(dltWeightsDH(B))
            XF = torch.matmul(W, idstMatrix(2*B));

        elif(self.xform == 'is'):
            XF = idstMatrix(2*B).t()

        # self.weight.data = XF.t().data;

        self.register_buffer('weight', XF.t(), persistent=False)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.type(x.dtype))


# Forward Discrete Legendre Transform
class FDLT(nn.Module):

    def __init__(self, B):
        super(FDLT, self).__init__()

        self.B = B;

        self.dct = weightedDCST(B, 'c');
        self.dst = weightedDCST(B, 's');

        sInd = torch.arange(1, B, 2);
        cInd = torch.arange(0, B, 2);

        self.register_buffer('cInd', cInd, persistent=False);
        self.register_buffer('sInd', sInd, persistent=False);

        self.register_buffer('Cm', normCm(B), persistent=False);

        self.register_buffer('D', wignerCoeffs(B), persistent=False);


    def forward(self, psiHat):

        # psiHat = b x M x phi

        B, b = self.B, psiHat.size()[0]

        # Multiply by normalization coefficients
        psiHat = torch.mul(self.Cm[None, B-1:, None], psiHat);

        # Apply DCT + DST to even + odd indexed m
        psiHat[:, self.cInd, :] = self.dct(psiHat[:, self.cInd, :]);
        psiHat[:, self.sInd, :] = self.dst(psiHat[:, self.sInd, :]);

        # Reshape for sparse matrix multiplication
        # psiHat = torch.transpose(torch.reshape(psiHat, (b, 2*B*(2*B - 1) ) ), 0, 1);
        # print(psiHat.shape)
        # psiHat = torch.transpose(torch.reshape(psiHat, (b, 2*B*(B) ) ), 0, 1);
        # psiTilde = torch.einsum('...bi,boi->...bo', psiHat, self.D.type(psiHat.dtype))
        # print(self.D.shape, psiHat.shape, psiTilde.shape)
        # Psi =  b x M x L
        return torch.einsum('...bi,boi->...bo', psiHat, self.D.type(psiHat.dtype)) # torch.permute(torch.reshape(torch.mm(self.D.type(psiHat.dtype), psiHat), (B, B, b)), (2, 0, 1));


# Inverse Discrete Legendre Transform
class IDLT(nn.Module):

    def __init__(self, B):
        super(IDLT, self).__init__()

        self.B = B;

        self.dct = weightedDCST(B, 'ic');
        self.dst = weightedDCST(B, 'is');


        sInd = torch.arange(1, B, 2);
        cInd = torch.arange(0, B, 2);

        self.register_buffer('cInd', cInd, persistent=False);
        self.register_buffer('sInd', sInd, persistent=False);

        self.register_buffer('iCm', torch.reciprocal(normCm(B)), persistent=False);

        self.register_buffer('DT', torch.transpose(wignerCoeffs(B), 1, 2), persistent=False);

    def forward(self, Psi):

        # Psi: b x M x L

        B, b = self.B, Psi.size()[0]

        psiHat = torch.einsum('...bi,boi->...bo', Psi, self.DT.type(Psi.dtype))

        # psiHat = torch.reshape(torch.transpose(torch.mm(self.DT.type(Psi.dtype), torch.transpose(torch.reshape(Psi, (b, (B)*B)), 0, 1)), 0, 1), (b, B, 2*B))

         #Apply DCT + DST to even + odd indexed m
        psiHat[:, self.cInd, :] = self.dct(psiHat[:, self.cInd, :]);
        psiHat[:, self.sInd, :] = self.dst(psiHat[:, self.sInd, :]);

        # f: b x theta x phi
        return torch.mul(self.iCm[None, B-1:, None], psiHat);


#############################################################
################ Spherical Harmonic Transforms ##############
#############################################################


class FTSHT(nn.Module):

    '''
    The Forward "Tensorized" Discrete Spherical Harmonic Transform

    Input:

    B: (int) Transform bandlimit

    '''
    def __init__(self, B):
        super(FTSHT, self).__init__()

        self.B = B;

        self.FDL = FDLT(B);

    def forward(self, psi):

        '''
        Input:

        psi: ( b x 2B x 2B torch.double or torch.cdouble tensor )
             Real or complex spherical signal sampled on the 2B X 2B DH grid with b batch dimensions

        Output:

        Psi: (b x (2B - 1) x B torch.cdouble tensor)
             Complex tensor of SH coefficients over b batch dimensions

        '''

        # psi: b x theta x phi (real or complex)
        B, b = self.B, psi.size()[0]

        ## FFT in polar component
        # psiHat: b x  M x Phi
        psiHat = torch.fft.rfft(psi, dim=1, norm='forward')[:, :B]
        # Forward DLT
        Psi = self.FDL(psiHat);

        return Psi


class ITSHT(nn.Module):

    '''
    The Inverse "Tensorized" Discrete Spherical Harmonic Transform

    Input:

    B: (int) Transform bandlimit

    '''

    def __init__(self, B, out_factor=4):
        super(ITSHT, self).__init__()

        self.B = B;
        self.out_size = out_factor*B
        self.IDL = IDLT(B);

    def forward(self, Psi):

        '''
        Input:

        Psi: (b x (2B - 1) x B torch.cdouble tensor)
             Complex tensor of SH coefficients over b batch dimensions

        Output:

        psi: ( b x 2B x 2B torch.cdouble tensor )
             Complex spherical signal sampled on the 2B X 2B DH grid with b batch dimensions

        '''

        # Psi: b x  M x L (complex)
        B, b = self.B, Psi.size()[0];
        # Inverse DLT
        psiHat = self.IDL(Psi)

        return torch.fft.irfft(psiHat, dim=1, n=self.out_size, norm='forward');

class ForwardSHTWrapper(nn.Module):
    """ Assumes input is batch x h x w x c """
    def __init__(self, patch_size=(90, 180), hidden_size=768):
        super(ForwardSHTWrapper, self).__init__()
        self.B = min(patch_size)//2
        self.out_factor = max(patch_size)//min(patch_size)

        self.hidden_dimension = hidden_size
        self.in_rearrange = Rearrange('b h w c -> (b c) w h')
        self.out_rearrange = Rearrange('(b c) blk -> b blk c', c=self.hidden_dimension)
        self.register_buffer('triu_inds', torch.triu_indices(self.B, self.B), persistent=False)

        self.SHT = FTSHT(self.B)

    def forward(self, x):
        x = self.in_rearrange(x)
        x = self.SHT(x)
        x = x[:, self.triu_inds[0], self.triu_inds[1]]
        x = self.out_rearrange(x)

        return x


class BackwardSHTWrapper(nn.Module):
    """ Assumes input is batch x h x w x c """
    def __init__(self, patch_size=(90, 180), hidden_size=768):
        super(BackwardSHTWrapper, self).__init__()
        self.B = min(patch_size)//2
        self.out_factor = max(patch_size)//min(patch_size)

        self.hidden_dimension = hidden_size
        self.out_rearrange = Rearrange('(b c) w h -> b h w c', c=self.hidden_dimension)
        self.in_rearrange = Rearrange('b blk c -> (b c) blk', c=self.hidden_dimension)
        self.register_buffer('triu_inds', torch.triu_indices(self.B, self.B), persistent=False)

        self.iSHT = ITSHT(self.B)

    def forward(self, x):
        x = self.in_rearrange(x)
        b = x.shape[0]
        out = x.new_zeros(b, self.B, self.B)
        out[:, self.triu_inds[0], self.triu_inds[1]] = x
        out = self.iSHT(out)
        out = self.out_rearrange(out)
        return out


class SPHFilter(nn.Module):
    def __init__(self, patch_size=(1440, 720), hidden_size=20):
        super().__init__()
        self.fwd = ForwardSHTWrapper(patch_size, hidden_size)
        self.bkwd = BackwardSHTWrapper(patch_size, hidden_size)
        self.patch_size = patch_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        filt_x = x[:, :self.patch_size[1]]
        filt_x = self.bkwd(self.fwd(filt_x))
        x[:, :self.patch_size[1]] = filt_x
        return rearrange(x, 'b h w c -> b c h w')




#####################################
# RANDOM UTILITIES IN HERE BECAUSE NOTEBOOKS DONT LIKE LOCAL IMPORTS
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


def fourier_resample(x, factor=1/8):
    """ Does resampling in Fourier space """
    dtype = x.dtype
    x = x.float()
    B, C, H, W = x.shape
    h, w = int(H*factor), int(W*factor)
    xfft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
    mid_fft = xfft.new_zeros(B, C, h, w//2+1)
    mid_fft[:, :, :min(H//2, h//2), :min(W//2+1, w//2+1)] =  xfft[:, :,  :min(H//2, h//2), :min(W//2+1, w//2+1)]
    mid_fft[:, :, -min(H//2, h//2):, :min(W//2+1, w//2+1)] = xfft[:, :,  -min(H//2, h//2):, :min(W//2+1, w//2+1)]
    return torch.fft.irfft2(mid_fft, dim=(-2, -1), norm='ortho').to(dtype=dtype)*factor

# This is a low sigma gaussian kernel I took off a random website
fi = [[[[0.0000,0.0000,0.0001,0.0000,0.0000],
[0.0000,0.0111,0.0833,0.0111,0.0000],
[0.0001,0.0833,0.6220,0.0833,0.0001],
[0.0000,0.0111,0.0833,0.0111,0.0000],
[0.0000,0.0000,0.0001,0.0000,0.0000],]]]

def _default_2d_filter():
    default_filter = torch.tensor(fi)
    return default_filter

def blur_2d(input: torch.Tensor, stride = 1, filter = None, dfs_full=True) -> torch.Tensor:
    """Applies a spatial low-pass filter. Code taken from Mosaic library.
    Args:
        input (torch.Tensor): A 4d tensor of shape NCHW
        stride (int | tuple, optional): Stride(s) along H and W axes. If a single value is passed, this
            value is used for both dimensions.
        filter (torch.Tensor, optional): A 2d or 4d tensor to be cross-correlated with the input tensor
            at each spatial position, within each channel. If 4d, the structure
            is required to be ``(C, 1, kH, kW)`` where ``C`` is the number of
            channels in the input tensor and ``kH`` and ``kW`` are the spatial
            sizes of the filter.
    By default, the filter used is:
    .. code-block:: python
            [1 2 1]
            [2 4 2] * 1/16
            [1 2 1]
    Returns:
        The blurred input
    """
    _, c, h, w = input.shape
    n_in_channels = c

    if filter is None:
        filter = _default_2d_filter()
    if (filter.shape[0] == 1) and (n_in_channels > 1):
        # filt is already a rank 4 tensor
        filter = filter.repeat((n_in_channels, 1, 1, 1))
    _, _, filter_h, filter_w = filter.shape
    pad = filter_h // 2
    if dfs_full:
        input = F.pad(input, (pad, pad, pad, pad), 'circular')
    else:
        input = F.pad(input, (pad, pad, pad, pad))
    return F.conv2d(input, filter, stride=stride, padding=0, groups=n_in_channels, bias=None)

## The scatter ops are copied from torch_scatter - since they have
## dependencies due to custom CUDA kernels we're not using,
## I just copied the code directly.

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device).real
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    # Changed this to do float division regardless
    out.true_divide_(count)
    return out

def fold_vector(x, factor=(8, 8), dims=(2, 3)):
    """Stacks FFT output according to modes that would be aliased under factor downsampling
    though I actually don't know if the rules hold for 2D data.
    Started parameterizing it, but it assumes b c h w format """
    dtype = x.dtype
    hf, wf = factor
    hlen, wlen = x.shape[dims[0]]//hf, x.shape[dims[1]]//wf
    # Stack "fold" indices from rfft axis
    w_stacks = []
    for i in range(0, wf, 2):
        w_stacks.append(torch.arange(i*wlen,(i+1)*wlen+1, device=x.device))
        w_stacks.append(torch.arange((i+2)*wlen, (i+1)*wlen-1, -1, device=x.device))
    w_stacks = torch.cat(w_stacks, -1)
    # Select axes in fold order
    x = torch.index_select(x, dims[1], w_stacks)
    # Rearrange all axes
    return rearrange(x, 'b c (h1 h2) (w1 w2) -> b (c h1 w1) h2 w2', h1=hf, w1 = wf)

def unfold_vector(x, factor=(8, 8), dims=(2, 3)):
    """Unstacks channels into modes where the original modes would have been
    indistinguishable at original sampling rate. 
    though I actually don't know if the rules hold for 2D data.
    
    Started parameterizing it, but it assumes b c h w format """
    hf, wf = factor
    hlen, wlen = x.shape[dims[0]], x.shape[dims[1]]-1
    # Get indices for scatter
    w_stacks = []
    for i in range(0, wf, 2):
        w_stacks.append(torch.arange(i*wlen,(i+1)*wlen+1, device=x.device))
        w_stacks.append(torch.arange((i+2)*wlen, (i+1)*wlen-1, -1, device=x.device))
    w_stacks = torch.cat(w_stacks, -1)
    # Unpatch
    x = rearrange(x, 'b (c h1 w1) h2 w2 -> b c (h1 h2) (w1 w2)', h1=hf, w1=wf)
    # RFFT axis
    return scatter_mean(x, index=w_stacks, dim=-1)


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


class ZonalEmbedding(nn.Module):
    def __init__(self, hidden_features, grid_shape, patch_size=(8, 8), layer_size=500, scale_factor=.02,
                 global_dfs=True):
        super().__init__()
        self.grid_shape = grid_shape
        self.out_shape = grid_shape
        self.patch_size = patch_size
        self.hidden_features = hidden_features
        self.global_dfs = global_dfs
        # If operating in space, use spherical harmonics for spatial locations
        self.bw_order = 10
        num_feats = self.build_features()
        self.mlp = nn.Sequential(nn.Linear(num_feats, layer_size),
            nn.GLU(),
            nn.Linear(layer_size//2, hidden_features, bias=False)
            )
        self.scale_factor = scale_factor

    def resize(self, grid_shape):
        self.grid_shape = grid_shape
        self.build_features()

    @torch.no_grad()
    def build_features(self, initial=False):
        # Otherwise assume we're in Fourier spacei
        full_ns = self.grid_shape[0]*self.patch_size[0] 
        if self.global_dfs:
            base_lats = torch.linspace(-np.pi/2, np.pi/2, full_ns//2+1 + 2)[1:-1] # Need to avoid polar zero somehow even if this is technically wrong
            base_lats = torch.cat([base_lats, torch.flip(base_lats[1:-1], (0,))])
        else:
            base_lats = torch.linspace(-np.pi/2, np.pi/2, self.full_ns)
        
        lat_coefs =  torch.cos(base_lats).reshape(-1, self.patch_size[0]).max(1).values
        grid_coords = torch.fft.rfftfreq(self.grid_shape[1])
        # BW Filter features
        denom = (self.grid_shape[1]//2 + 1)*lat_coefs
        bw_base = (grid_coords*self.grid_shape[1]).unsqueeze(0) / denom.unsqueeze(1)
        bw_feats = []
        for n in range(1, self.bw_order):
            bw_feats.append(1/(1 + bw_base**(2*n)))

        space_features = lat_coefs.unsqueeze(1) * (1-grid_coords.unsqueeze(0)) # Think this should be monotic with wavelength
        space_features = torch.stack([space_features, space_features**2, space_features**3] + bw_feats,-1).unsqueeze(0)
        grid_coords = (space_features - space_features.mean((1,2), keepdims=True)) / (1e-5 + space_features.std((1, 2), keepdims=True))
        # TODO take and store std separately
        self.register_buffer('space_features', space_features, persistent=False)
        return space_features.shape[-1]

    def forward(self, augs=None):
        # Remember to fix this for larger batches
        out = self.scale_factor*self.mlp(self.space_features)
        return out

class BWInterpFilter(nn.Module):
    def __init__(self, input_size=(1440, 1440), max_lats=720, max_res=640, scale_factor=1, grid_res='linear', res_factor=.9, dfs_type='full', hidden_size=768, filter_order=7,
            include_pole=False):
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
        self.include_pole = include_pole
        if include_pole:
            init_res = np.linspace(0, np.pi, max_lats+1)
        else:
            delta = np.pi / (max_lats+1)
            init_res = np.linspace(delta/2, np.pi-delta/2, max_lats)
            
        
        max_ns_bw = int(min(res_fact*input_size[0]//2, max_res)) # AA - Unfortunately needs to cutoff based on current resolvables
        max_ew_bw = int(min(max_res, (max_lats+1)*res_fact)) # Shaping - Cuts off based on max resolution
        #max_ns_bw = int(max_res*res_fact)

        self.reduction_factor = 1
        if dfs_type == 'full':
            if include_pole:
                init_res = np.concatenate([init_res, init_res[1:-1][::-1]])
            else:
                init_res = np.concatenate([init_res, init_res[::-1]])
            init_ns_reses = np.sin(init_res)
            if input_size[0] != 2*max_lats: # Fix for non-DFS later
                self.reduction_factor = int(max_lats/ (input_size[0]/2))
                ns_reses = init_ns_reses.reshape(-1, self.reduction_factor).max(1)
            else:
                ns_reses = init_ns_reses
        # ew_filter = torch.ones(input_size[0], input_size[1]//2 + 1)
        ew_lanczos = torch.zeros(input_size[0], input_size[1]//2+1)
        for i, res in enumerate(ns_reses):
            cutoff = min(int(1+res*max_ew_bw), input_size[0]//2+1)
            cutoff = min(int(1+res*max_ew_bw*res_factor), res_factor*input_size[0]//2+1)
            # ew_filter[i,] = torch.fft.rfftfreq(input_size[1])*input_size[1]
            ew_lanczos[i] = 1/ (1+(torch.fft.rfftfreq(input_size[1])*input_size[1] / cutoff)**(2*filter_order))
        # ew_filter = ew_filter[None, None, :, :].bool() # Hope this is B C H W

        # ns_filter = torch.zeros(input_size[0], input_size[1]//2+1)
        
        ns_filter = ((torch.fft.fftfreq(input_size[0])*input_size[0]).unsqueeze(1)**2 
                     + (torch.fft.rfftfreq(input_size[1])*input_size[1]).unsqueeze(0)**2).sqrt()
        ns_filter = 1 / (1 + (ns_filter/max_ns_bw)**(2*filter_order))
             
        ns_filter = ns_filter[None, None, :, :]#.bool()

        self.apply_zonal = True#torch.any(ew_filter).item()
        self.apply_2d = torch.any(ns_filter).item()
        # self.register_buffer('ew_filter', ew_filter, persistent=False)
        self.register_buffer('ns_filter', ns_filter, persistent=False)
        self.register_buffer('oned_lanczos', ew_lanczos[None, None, :, :], persistent=False)
        # self.register_buffer('ns_2dlanczos', ns_2dlanczos[None, None, :, None], persistent=False)
        # self.register_buffer('ew_2dlanczos', ew_2dlanczos[None, None, None, :], persistent=False)
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

            if self.dfs_type != 'full':
                xfft = sphere_to_torus_rfft(x)
            #TODO still need to apply filter here
            else:
                xfft = torch.fft.rfft(x, dim= -1, norm='ortho')
            #xfft.masked_fill_(ew_filter, 0)
                if self.apply_zonal:
                #xfft_clip = complex_clamp(torch.masked_select(xfft, ew_filter))
                    xfft = xfft * self.oned_lanczos
            #xfft = xfft * ew_filter
            xfft = torch.fft.fft(xfft, dim=-2, norm='ortho')# * self.ns_filter
            if self.apply_2d:
            #xfft_clip = complex_clamp(torch.masked_select(xfft, self.ns_filter))
                xfft = xfft * self.ns_filter
        # Resize
            if self.scale_factor != 1:
                mid_fft = xfft.new_zeros(B, C, h, w//2+1)
                mid_fft[:, :, :min(H//2, h//2), :min(W//2+1, w//2+1)] =  xfft[:, :,  :min(H//2, h//2), :min(W//2+1, w//2+1)]
                mid_fft[:, :, -min(H//2, h//2):, :min(W//2+1, w//2+1)] = xfft[:, :,  -min(H//2, h//2):, :min(W//2+1, w//2+1)]
            else:
                mid_fft = xfft
        #mid_fft = torch.fft.ifft(mid_fft, norm='ortho', dim=-2)
        #if self.scale_factor == 1:
        #    mid_fft = mid_fft * self.ew_filter
            out = torch.fft.irfft2(mid_fft, norm='ortho').to(dtype=dtype) * self.scale_factor # Each dir gets sqrt(scale_factor)
            if self.dfs_type != 'full':
                out = out[:, :, :out.shape[2]//2]
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
            #ew_filter = torch.sigmoid(rearrange(self.gen_zonal(), 'b h w c -> b c h w'))#+self.ew_filter)
            #ns_filter =  torch.sigmoid(rearrange(self.gen_2d(x), 'b h w c -> b c h w'))#+self.ns_filter)
            if self.dfs_type != 'full':
                xfft = sphere_to_torus_rfft(x)
            #TODO still need to apply filter here
            else:
                xfft = torch.fft.rfft(x, dim= -1, norm='ortho')
            #xfft.masked_fill_(ew_filter, 0)
                #if self.apply_zonal:
                #xfft_clip = complex_clamp(torch.masked_select(xfft, ew_filter))
                 #   xfft = xfft.masked_fill_(ew_filter, 0)
            xfft = xfft * ew_filter
            xfft = torch.fft.fft(xfft, dim=-2, norm='ortho') * self.ns_filter
            #if self.apply_2d:
            #xfft_clip = complex_clamp(torch.masked_select(xfft, self.ns_filter))
            #    xfft = xfft.masked_fill_(self.ns_filter, 0)
        # Resize
            if self.scale_factor != 1:
                mid_fft = xfft.new_zeros(B, C, h, w//2+1)
                mid_fft[:, :, :min(H//2, h//2), :min(W//2+1, w//2+1)] =  xfft[:, :,  :min(H//2, h//2), :min(W//2+1, w//2+1)]
                mid_fft[:, :, -min(H//2, h//2):, :min(W//2+1, w//2+1)] = xfft[:, :,  -min(H//2, h//2):, :min(W//2+1, w//2+1)]
            else:
                mid_fft = xfft
        #mid_fft = torch.fft.ifft(mid_fft, norm='ortho', dim=-2)
        #if self.scale_factor == 1:
        #    mid_fft = mid_fft * self.ew_filter
            out = torch.fft.irfft2(mid_fft, norm='ortho').to(dtype=dtype) * self.scale_factor # Each dir gets sqrt(scale_factor)
            if self.dfs_type != 'full':
                out = out[:, :, :out.shape[2]//2]
        return out

def complex_clamp(x, mx=1e-4, mn=None, eps=1e-6):
    out = x.clone()
    x_norms = x.abs()
    out[x_norms > mx] = x[x_norms > mx] / (x_norms[x_norms > mx]+eps) * mx
    return out
    
class MiniResBlock(nn.Module):
    def __init__(self, features, activation=nn.ReLU(), groups=1, kernel_size=3):
        super().__init__()
        self.activation = activation
        padding = kernel_size // 2
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(features,
                features,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode='circular',
                groups=groups))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(features,
                features,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode='circular',
                groups=groups))
        #self.norm = nn.BatchNorm2d(features)

    def forward(self, x):
        #out = self.norm(x)
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x

class FusionBlock(nn.Module):
    def __init__(self, features, activation=nn.GELU(), groups=1, kernel_size=3):
        super().__init__()
        #self.side_conv = MiniResBlock(features, activation, groups, kernel_size)
        self.comb_block = MiniResBlock(features, activation, groups, kernel_size)

    def forward(self, *xs):
        side, x = xs
        res = side
        output = self.comb_block(res+x)
        return output

class HackInstanceNorm2d(nn.Module):
    def __init__(self, num_features, affine=False):
       super().__init__()
       self.norm = nn.InstanceNorm2d(num_features)#nn.InstanceNorm2d(num_features, affine=True)

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.norm(x)
        x = rearrange(x, 'b c h w -> b h w c')
        return x

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
        #
        #self.block_size = in_features // groups
        #self.scale1 = 1 / self.block_size**.5
        #base_weight1 = self.scale1 * torch.randn(groups, self.block_size, self.block_size, dtype=torch.cfloat)

        #self.weight = nn.Parameter(torch.stack([base_weight1.real, base_weight1.imag], dim=-1))

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        x = F.conv2d(x, torch.view_as_complex(self.weight).to(dtype=x.dtype), groups=self.groups)
        return rearrange(x, 'b c h w -> b h w c')

######################################################################################3
# Project modules
######################################################################################

class AFNO2D(nn.Module):
    def __init__(self, hidden_size, grid_size, grid_res, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1,
            activation_type='glu', dfs_type='full', fourier_conv_type='meta'):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"
        self.grid_size = grid_size
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.grid_res = grid_res
        # Different things we've parameterized
        self.fourier_conv_type = fourier_conv_type
        self.dfs_type = dfs_type
        self.activation = complex_glu if activation_type=='glu' else complex_relu
        activation_factor = .5 if activation_type=='glu' else 1
        dfs_factor = 2 if dfs_type == 'lite' else 1
        # Apply one full rank Fourier conv
        if self.fourier_conv_type == 'full':
            self.filter1 = nn.Parameter(torch.view_as_real(.02*torch.randn(1, dfs_factor*grid_size[0], grid_size[1]//2+1, self.num_blocks, self.block_size, dtype=torch.cfloat)))
            self.filter2 = 1. # No way we have memory for two full ones
        # Apply a rank 1 Fourier conv to each channel twice
        elif self.fourier_conv_type == 'factored':
            self.filt1_ns = nn.Parameter(torch.view_as_real(.02*torch.randn(1,dfs_factor* grid_size[0], 1, self.num_blocks, self.block_size, dtype=torch.cfloat)))
            self.filt1_ew = nn.Parameter(torch.view_as_real(.02*torch.randn(1, 1, grid_size[1]//2+1, self.num_blocks, self.block_size, dtype=torch.cfloat)))
            self.filt2_ns = nn.Parameter(torch.view_as_real(.02*torch.randn(1, dfs_factor*grid_size[0], 1, self.num_blocks, self.block_size, dtype=torch.cfloat)))
            self.filt2_ew = nn.Parameter(torch.view_as_real(.02*torch.randn(1, 1, grid_size[1]//2+1, self.num_blocks, self.block_size, dtype=torch.cfloat)))
        # Apply a real coefficient conv to the rfft and n-s component of the fft2 separately
        elif self.fourier_conv_type == 'lat_filter':
            self.filter1 = nn.Parameter(.02*torch.randn(1, dfs_factor*grid_size[0], grid_size[1]//2+1, 1))
            self.filter2 = nn.Parameter(.02*torch.randn(1, dfs_factor*grid_size[0], 1, 1))
        # Use meta-network to define two full Fourier convs
        elif self.fourier_conv_type == 'meta':
            self.conv_gen1 = ContSphereEmbedding(hidden_size, (dfs_factor*grid_size[0], grid_size[1]), scale_factor=1.,  sphere=False)
            self.conv_gen2 = ContSphereEmbedding(hidden_size, (dfs_factor*grid_size[0], grid_size[1]), sphere=False)
        else:
            print('Convolution type unknown: defaulting to None')

        # Quick Kaiming-style init (haven't done complex math to make sure its correct)
        self.conv1 = nn.utils.spectral_norm(ComplexGroupedMLP(self.hidden_size, self.hidden_size, num_blocks))
        self.conv2 = nn.utils.spectral_norm(ComplexGroupedMLP(self.hidden_size, self.hidden_size, num_blocks))

    def forward(self, x, dt_ind):
        dtype = x.dtype
        #x = x.float()
        
        B, H, W, C = x.shape
        # If we're doing the cheap sst in reduced space only, then it should be mixed with the rfft
        # to cut half the rffts. If we're also applying lat and full frequency filters, do that now too.
        if self.dfs_type == 'lite':
            x = sphere_to_torus_rfft(x)
            H = H*2 # Apply DFS
            if self.fourier_conv_type == 'lat_filter':
                x = x*self.filter1
            x = torch.fft.fft(x, norm='ortho', dim = 1)
            if self.fourier_conv_type == 'lat_filter':
                x = x*self.filter2
        else:
            with torch.cuda.amp.autocast(enabled=False):
                delta = np.pi/129
                colats = np.sin(np.linspace(delta/2, np.pi-delta/2, 128)).reshape(-1, 4).max(1)
                colats = torch.tensor(np.concatenate([colats, colats[::-1]]), dtype=x.dtype, device=x.device)[None, :, None, None]
                x = x * colats
                x = torch.fft.rfft2(x.float(), dim=(1, 2), norm="ortho")

        if self.fourier_conv_type == 'meta':
            # Phase-only conv
            with record_function('Conv gen'):
                filt1 = torch.exp(2j*np.pi*self.conv_gen1(dt_ind).reshape(1, H, W//2+1, self.num_blocks, self.block_size))
                filt2 = torch.tanh(self.conv_gen2(dt_ind).reshape(1, H, W//2+1, self.num_blocks, self.block_size))
        elif self.fourier_conv_type == 'full':
            filt1 = torch.view_as_complex(self.filter1)
            filt2 = self.filter2
        # Just set these to 1 to avoid more branches    
        else:
            filt1 = 1.
            filt2 = 1.
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        # First linear section
        #x = x*filt1
        with record_function('convs and einsums'):
            with record_function('ein'):
                x = self.conv1(x) 
            with record_function('complex conv'):
                x = x * filt1 
            if self.fourier_conv_type == 'factored':
                x = x*torch.view_as_complex(self.filt1_ns) * torch.view_as_complex(self.filt1_ew)
            with record_function('complex activation'):
                x = self.activation(x)

        # Second linear part
        #x = x * filt2
            with record_function('ein'):
                x = self.conv2(x)
            with record_function('complex conv'):
                x = x*filt2
        if self.fourier_conv_type == 'factored':
            x = torch.view_as_complex(self.filt2_ns) * x * torch.view_as_complex(self.filt2_ew)

        x = x.reshape(B, H, W // 2 + 1, C)
        # Again, if DFS-lite then cut the ops in half at the expense of 2 kernels
        if self.dfs_type == 'lite':
            x = torch.fft.ifft(x, dim=1, norm='ortho')[:, :H//2]
            x = torch.fft.irfft(x, dim=2, norm='ortho')
        elif self.fourier_conv_type == 'lat_filter':
            x = torch.fft.ifft(x*self.filter2, dim=1, norm='ortho')
            x = torch.fft.irfft(x*self.filter1, dim=2, norm='ortho')
        else:
            with torch.cuda.amp.autocast(enabled=False):
                x = torch.fft.irfft2(x.cfloat(), dim=(1,2), norm="ortho")
                x = x / colats
        x = x.type(dtype)
        return x


class Block(nn.Module):
    def __init__(
            self,
            dim,
            grid_size,
            grid_res='linear',
            dfs_type='full',
            fourier_conv_type='meta',
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            fno_activation_type='glu',
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0
        ):
        super().__init__()
        norm_layer = HackInstanceNorm2d
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(dim, grid_size, grid_res, num_blocks, sparsity_threshold, hard_thresholding_fraction, 
                dfs_type=dfs_type, activation_type=fno_activation_type, fourier_conv_type=fourier_conv_type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip
        self.grid_res = grid_res

    def forward(self, x, dt_ind):
        residual = x
        with record_function('AFNO'):
            x = self.norm1(x)
            x = self.filter(x, dt_ind)

            if self.double_skip:
                x = x + residual
                residual = x
        with record_function('MLP'):
            x = self.norm2(x)
            x = self.mlp(x)
            x = self.drop_path(x)
            x = x + residual
        return x

class PrecipNet(nn.Module):
    def __init__(self, params, backbone):
        super().__init__()
        self.params = params
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.backbone = backbone
        self.ppad = PeriodicPad2d(1)
        self.conv = nn.Conv2d(self.out_chans, self.out_chans, kernel_size=3, stride=1, padding=0, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.backbone(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        return x

class AFNONet(nn.Module):
    def __init__(
            self,
            params,
            img_size=(720, 1440),
            patch_size=(16, 16),
            in_chans=2,
            out_chans=2,
            embed_dim=384,
            depth=8,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        ):
        super().__init__()
        self.params = params
        img_size = params.img_size
        self.max_lats = img_size[0]
        if self.max_lats != 720: # Hack to add extra dset
            self.max_res = self.max_lats
        else:
            self.max_res = 640 # Want to remove this at some point
        if 'resuming' in params and params.resuming:
            self.img_size = tuple([int(s*params.scale_factor*params.rescale_factor) for s in img_size])
        else:
            self.img_size = tuple([int(s*params.scale_factor) for s in img_size])
        if params.dfs_type == 'full':
            self.img_size = self.img_size[0]*2, self.img_size[1]
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = params.num_blocks
        # For the v fields
        means = np.load(params.global_means_path)[:, params.v_channels].squeeze((0, 2, 3))
        stds = np.load(params.global_stds_path)[:, params.v_channels].squeeze((0, 2, 3))
        v_offsets = means/stds
        self.register_buffer('v_offsets', torch.as_tensor(v_offsets)[None, :, None, None], persistent=False)
        # New parameterizations
        self.pos_embedding_type = params.pos_embedding
        self.up_type = params.up_type
        self.grid_res = params.grid_res # Note - only applies on output if not using a Fourier upsampler
        self.res_factor = params.res_factor
        self.resample_type = params.resample_type
        if self.grid_res != 'full':
            #self.clip = SPHFilter()
            self.clip = BWInterpFilter(input_size=self.img_size, max_lats=self.max_lats, max_res=self.max_res, 
                    res_factor=self.res_factor, grid_res=self.grid_res, dfs_type=params.dfs_type, hidden_size=self.in_chans)
        # IDK
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
   
        #self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.dfs_type = params.dfs_type
        self.h = self.img_size[0] // self.patch_size[0]
        self.w = self.img_size[1] // self.patch_size[1]
        
        # Up/Downsampling stuffi
        if self.resample_type == 'patch':
            self.patch_embed = OneStepPatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=embed_dim)
            self.output_head = OneStepUnpatch(self.img_size, self.patch_size, self.out_chans, embed_dim)
        else:
            self.patch_embed = MultiStepPatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, 
                    embed_dim=embed_dim, dfs_type=self.dfs_type, embeddings_everywhere=self.pos_embedding_type=='meta_per_res')
            self.output_head = FourierConvUpsample(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.out_chans,
                    embed_dim=embed_dim, dfs_type=self.dfs_type, res_factor=self.res_factor, grid_res = self.grid_res, max_lats=self.max_lats, max_res=self.max_res)
        
        # Pos Embedding
        if self.pos_embedding_type == 'meta':
            self.cont_pos_embed = ContSphereEmbedding(self.embed_dim, (self.h, self.w), self.patch_size,
                    steps_per_day=params.steps_per_day,samples_per_year=params.steps_per_year,  global_dfs=self.dfs_type=='full')
        elif self.pos_embedding_type == 'learned':
            self.pos_embed = nn.Parameter(torch.zeros(1, self.h, self.w, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
        else:
            self.pos_embed = 0.
        
        if params.config == 'afno_swe':
            delta = np.pi / (img_size[0]+1)
            colats = np.linspace(delta/2, np.pi-delta/2, img_size[0])
            colats = torch.as_tensor(np.concatenate([colats, colats[::-1]])).float()[None, None, :, None]
        else:
            pass

        #self.register_buffer(colats, 'colats', persistent=False)
        # Processing layer
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, grid_size=(self.h, self.w),grid_res=self.grid_res,  mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction,
            fourier_conv_type=params.fourier_conv_type, fno_activation_type=params.activation_type, dfs_type=params.dfs_type) 
        for i in range(depth)])


        # Eh, could probably improve init, but not this
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def resize(self, img_size):
        old_img_size = self.img_size
        self.img_size = img_size
        scale_factor = img_size[0] / old_img_size[0]
        if scale_factor != 1:
            self.patch_embed.resize(img_size, scale_factor)
            pos_embed = self.pos_embed.transpose(1, 2).reshape(1, self.embed_dim, self.h, self.w)
            pos_embed = F.interpolate(pos_embed, scale_factor=scale_factor, mode='bicubic').reshape(1, self.embed_dim, int(self.h*self.w*scale_factor**2)).transpose(1, 2)
            self.h = self.img_size[0] // self.patch_size[0]
            self.w = self.img_size[1] // self.patch_size[1]
            self.pos_embed = nn.Parameter(pos_embed)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def forward_features(self, x, dt_ind, augs=()):
        B = x.shape[0]
        # Assumes whatever comes from patchifiers is channels first
        x = rearrange(x, 'b c h w -> b h w c')
        with record_function('pos embed'):
            if self.pos_embedding_type == 'meta': 
                x = self.cont_pos_embed(dt_ind, augs) + x
            else:
                x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x, dt_ind)
        return x

    def forward(self, x, dt_ind, augs=(), extra_out=False):
        params = self.params
        # Move this out later because it should apply to valid too, but CPU is too slow
        with record_function('preprocess'):
            if self.dfs_type == 'full':
                if params.config == 'afno_swe':
                    #x[:, self.params.u_channels] = torch.sin(self.colats) * x[:, self.params.u_channels]
                    x = sphere_to_torus(x)
                    x[:, self.params.v_channels, x.shape[2] // 2:] += self.v_offsets
                    x[:, self.params.v_channels, x.shape[2] // 2:] = -x[:, self.params.v_channels, x.shape[2] // 2 :]
                    x[:, self.params.v_channels, x.shape[2] // 2:] -= self.v_offsets
                else:
                    x = full_sphere_to_torus(x)
                    x[:, self.params.v_channels, x.shape[2] // 2+1:] += self.v_offsets
                    x[:, self.params.v_channels, x.shape[2] // 2+1:] = -x[:, self.params.v_channels, x.shape[2] // 2 + 1:]
                    x[:, self.params.v_channels, x.shape[2] // 2+1:] -= self.v_offsets
            #if self.grid_res != 'full' and self.training:
            #    with torch.no_grad():
            #        x = self.clip(x, augs)
            res = x.clone()
        # Downsample - B C H W -> B C H W 
        with record_function('downsample'):
            x, linear_paths = self.patch_embed(x, dt_ind, augs)
        # Next actually do our thing - B C H W -> B H W C
        with record_function('processer blocks'):
            x = self.forward_features(x, dt_ind, augs)
        # Upsample again - B H W C input (lin paths B C H W), B C H W output
        with record_function('upscale'):
            x = self.output_head(x, linear_paths, dt_ind, augs)
        

        with record_function('postprocess'):
            x = x+res
        
            if self.grid_res != 'full':
                x = self.clip(x, augs)
            #x = self.clip(x, augs)
            #x = self.clip(x, augs)
            #x = self.clip(x, augs)
        # Easier to handle augmentation if we don't need to reverse everything
        # so just output the DFS model for DFS Full during training and undo
        # it during validationi, this is going to need to be fixed for MS rollouts
            if self.dfs_type == 'full':
                if self.params.config == 'afno_swe':
                    with record_function('glide reflection'):
                        x_sph_t_tor = glide_reflection(x[:, :, self.img_size[0] // 2:])
                    with record_function('manage offsets'):
                        x_sph_t_tor[:, self.params.v_channels+params.u_channels] += self.v_offsets
                        x_sph_t_tor[:, self.params.v_channels+params.u_channels] =  -x_sph_t_tor[:, self.params.v_channels+params.u_channels]
                        x_sph_t_tor[:, self.params.v_channels+params.u_channels] -= self.v_offsets#            x_sph_t_tor = glide_reflection(x[:, :, self.img_size[0] // 2:])
                    with record_function('out stuff'):
                        if not extra_out and False:
                            x = (x[:, :, :self.img_size[0] // 2])
                            x[:, :, 1:] = .5*(x[:, :, 1:] + x_sph_t_tor[:, :, :-1])
                        else:
                            x = (x[:, :, :self.img_size[0] // 2])
                            x = .5*(x + x_sph_t_tor)
                            #x[:, self.params.u_channels] = 1/torch.sin(self.colats) * x[:, self.params.u_channels]

                else:
                    with record_function('glide reflection'):
                        x_sph_t_tor = glide_reflection(x[:, :, self.img_size[0] // 2:])
                    with record_function('manage offsets'):
                        x_sph_t_tor[:, self.params.v_channels] += self.v_offsets
                        x_sph_t_tor[:, self.params.v_channels] =  -x_sph_t_tor[:, self.params.v_channels]
                        x_sph_t_tor[:, self.params.v_channels] -= self.v_offsets#            x_sph_t_tor = glide_reflection(x[:, :, self.img_size[0] // 2:])
                    with record_function('out stuff'):
                        if not extra_out:
                            x = (x[:, :, :self.img_size[0] // 2])
                            x[:, :, 1:] = .5*(x[:, :, 1:] + x_sph_t_tor[:, :, :-1])
                        else:
                            x = (x[:, :, :self.img_size[0] // 2+1])
                            x[:, :, 1:-1] = .5*(x[:, :, 1:-1] + x_sph_t_tor[:, :, :-1])
        #    x = self.clip(x)
        return x

#####################################################################################3
# Downsampling layers - Note changed these so they output in img shape and not "token" form
######################################################################################


class OneStepPatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, dt_ind, augs):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x, []

    def resize(self, img_size, scale_factor):
        self.img_size = img_size
        self.num_patches = self.num_patches*scale_factor**2

class MultiStepPatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, blur=False, dfs_type='full', embeddings_everywhere=True):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        # No idea if this blur does anything, but the original version wasn't parameterized so it was on for everything
        self.use_blur = blur
        if self.use_blur:
            self.register_buffer('blur', _default_2d_filter(), persistent=False)
        self.num_patches = num_patches
        self.log_ps = int(np.log2(patch_size[0])) # Currently powers of two
        channels = [int(embed_dim / 2**(self.log_ps-i)) for i in range(int(self.log_ps+1))]
        # Use independent linear layers at each resolution - ends up being slighter cheaper than repeatedly downsampling
        # and all layers are going to be low rank anyway since they're just linear
        self.head = nn.utils.spectral_norm(nn.Conv2d(in_chans, channels[0], 1))
        self.projs = nn.ModuleList([nn.utils.spectral_norm(nn.Conv2d(in_chans, channels[i], kernel_size=2**i, stride=2**i)) for i in range(1, self.log_ps+1)])
        if embeddings_everywhere:
            self.embedding_set = nn.ModuleList([ContSphereEmbedding(channels[i], (self.img_size[0] // 2**i, self.img_size[1] // 2**i), (2**i, 2**i),
                global_dfs=dfs_type=='full') 
                for i in range(0, self.log_ps+1)])
        self.apply_embeddings = embeddings_everywhere

    def forward(self, x, dt_ind, augs):
        B, H, W, C = x.shape
        linear_paths = []
        linear_paths.append(self.head(x)) 
        if self.apply_embeddings:
            linear_paths[0] = linear_paths[0] + rearrange(self.embedding_set[0](dt_ind, augs), 'b h w c -> b c h w')
        for i, proj in enumerate(self.projs, 1):
            if self.use_blur:
                x = blur_2d(x, filter=self.blur) # Blur to reduce aliasing - not sure if does anything
            out = proj(x) 
            if self.apply_embeddings:
                out = out + rearrange(self.embedding_set[i](dt_ind, augs), 'b h w c -> b c h w')
            linear_paths.append(out)
        x = linear_paths.pop()
        return x, linear_paths

##############################################################################################33
# Upsampling layers
##################################################################################################
class OneStepUnpatch(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), out_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size=patch_size
        self.head = nn.Linear(embed_dim, out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

    def forward(self, x, linear_paths, dt_ind, augs):
        x = self.head(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        return x

class FourierConvUpsample(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768,
            grid_res='linear',
            res_factor=.9,
            dfs_type='full',
            max_lats=720,
            max_res=640):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.log_ps = int(np.log2(patch_size[0]))
        self.res_factor = res_factor
        channels = ([int(embed_dim / 2**(self.log_ps-i)) for i in range(int(self.log_ps+1))])
        self.out = nn.utils.spectral_norm(nn.Conv2d(channels[0], in_chans, 1))
        self.projs = nn.ModuleList([nn.utils.spectral_norm(nn.Conv2d(channels[i+1], channels[i], 1)) for i in range(self.log_ps)][::-1])
        self.merge_layers = nn.ModuleList([FusionBlock(channels[i]) for i in range(self.log_ps)][::-1])
        self.upsamplers = nn.ModuleList([BWInterpFilter((img_size[0]//2**(self.log_ps-i), img_size[1]//2**(self.log_ps-i)), scale_factor=2,
            grid_res=grid_res, res_factor=res_factor, dfs_type=dfs_type, max_lats=max_lats, max_res=max_res, hidden_size=channels[-(i+1)])
            for i in range(self.log_ps)])
        self.ln1s = nn.ModuleList([nn.LayerNorm(channels[-(i+2)]) for i in range(self.log_ps)])
        self.ln2s = nn.ModuleList([nn.LayerNorm(channels[-(i+2)]) for i in range(self.log_ps)])

    def forward(self, x, linear_path, dt_ind, augs):
        B, H, W, C = linear_path[0].shape
        x = rearrange(x, 'b h w c -> b c h w')
        for i, proj in enumerate(self.projs):
            skip = linear_path.pop()
            x = proj(x)
            x = self.upsamplers[i](x, augs)
            #x = rearrange(x, 'b c h w -> b h w c')
            #x1, x2 = torch.tensor_split(x, 2, dim=-1)
            #ln1 = self.ln1s[i]
            #ln2 = self.ln2s[i]
            #x = ln1(x1)*ln2(x2)
            #x = x1*x2
            #x = rearrange(x, 'b h w c -> b c h w')
            #x = F.glu(x, dim=1)
            x = self.merge_layers[i](skip, x)
        x = self.out(x)
        return x

if __name__ == "__main__":
    model = AFNONet(img_size=(720, 1440), patch_size=(4,4), in_chans=3, out_chans=10)
    sample = torch.randn(1, 3, 720, 1440)
    result = model(sample)
    print(result.shape)
    print(torch.norm(result))

