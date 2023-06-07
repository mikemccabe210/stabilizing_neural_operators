''' From PINO: https://github.com/devzhk/PINO/blob/master/models/basics.py '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from einops import rearrange
try:
    from afnonet import ContSphereEmbedding, glide_reflection, sphere_to_torus
    from spectral_norm_utils import spectral_norm
except:
    from .afnonet import ContSphereEmbedding, glide_reflection, sphere_to_torus
    from .spectral_norm_utils import spectral_norm

def _get_act(activation):
    if activation == 'tanh':
        func = F.tanh
    elif activation == 'gelu':
        func = F.gelu
    elif activation == 'relu':
        func = F.relu_
    elif activation == 'elu':
        func = F.elu_
    elif activation == 'leaky_relu':
        func = F.leaky_relu_
    else:
        raise ValueError(f'{activation} is not supported')
    return func

def spectral_hook(m, grad_in, grad_out):
    g_conv, g_inp = grad_in
    #out_mags = grad_out[0]
    #phase = g_conv.angle()
    #mags = out_mags.abs().mean(0, keepdim=True) #/ (m.inps.abs().mean(0, keepdim=True)+1e-7)
    #orig_norm = torch.linalg.norm(g_conv.reshape(-1)) 
    #new_norm = (torch.linalg.norm(mags.reshape(-1)))
    #norm_rat = orig_norm / new_norm # Don't want to mess up other layers so keep same norm
    #mags = norm_rat * mags
    return (g_conv*1000, g_inp)

class dummy_spectral_mult(nn.Module):
    def forward(self, weights, inputs):
        self.weights = weights
        self.inps = inputs
        return weights * inputs

def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", a, b)


def compl_mul2d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixy,ioxy->boxy", a, b)


def compl_mul3d(a, b):
    return torch.einsum("bixyz,ioxyz->boxyz", a, b)

@torch.jit.script
def compl_mul2d_v2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bixys,ioxyt->stboxy", a, b)
    return torch.stack([tmp[0,0,:,:,:,:] - tmp[1,1,:,:,:,:], tmp[1,0,:,:,:,:] + tmp[0,1,:,:,:,:]], dim=-1)


def ifft2_and_filter(x, size_0, size_1, zonal_filter=None, twod_filter=None):
    if twod_filter is not None:
        x = x * twod_filter
    if zonal_filter is not None:
        x = torch.fft.ifft(x, dim=-2, norm='ortho', n=size_0)
        x = x * zonal_filter
        return torch.fft.irfft(x.cdouble(), dim=-1, norm='ortho', n=size_1)
    else:
        return torch.fft.irfft2(x.cdouble(), dim=(-2,-1), norm='ortho', s=(size_0, size_1))

class SpectralRescale(nn.Module):
    def __init__(self, h, channels, momentum=.1, affine=True, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.affine=affine
        if affine:
            self.mags = nn.Parameter(torch.view_as_real(torch.ones(1, channels, 1, dtype=torch.cfloat)))
            self.bias = nn.Parameter(torch.view_as_real(torch.zeros(1, channels, 1, dtype=torch.cfloat)))
        self.register_buffer('mean', torch.view_as_real(torch.zeros(1, 1, h, dtype=torch.cfloat)))
        self.register_buffer('std', torch.ones(1, 1, h))
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            mean, std = x.mean((0, 1), keepdims=True), x.std((0, 1), keepdims=True)
            with torch.no_grad():
                self.mean.copy_((1-self.momentum)*self.mean+self.momentum*torch.view_as_real(mean))
                self.std.copy_((1-self.momentum)*self.std+self.momentum*std)
        else:
            mean, std = torch.view_as_complex(self.mean), self.std

        x = (x - mean) / (self.eps + std)
        if self.affine:

            x = x * torch.view_as_complex(self.mags) + torch.view_as_complex(self.bias)
        return x

class mod_relu(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.))
        #self.b.requiresGrad = True

    def forward(self, x):
        return torch.polar(F.gelu(torch.abs(x) + self.b), x.angle())
        #return F.relu(torch.abs(x) + self.b) * torch.exp(1.j * torch.angle(x)) 

def dfs_padding(x, padding):
    b, c, h, w = x.shape
    out = F.pad(x, (0, 0, padding, padding))
    gr = glide_reflection(x)
    out[:, :, :padding] = gr[:, :, -padding:]
    out[:, :, -padding:] = gr[:, :, :padding]
    return out

def complex_glu(x, mag_bias, phase_bias):
    mag, phase = x.abs(), x.angle()
    return torch.polar(torch.sigmoid(mag+mag_bias), phase+phase_bias)


def spectral_dfs(x):
    b, c, h, w = x.shape
    phase_shift = torch.ones(w, device=x.device, dtype=x.dtype)
    phase_shift[1::2] *= -1
    append = torch.flip(x, dims=(2,))
    append *= phase_shift[None, None, None, :]
    return torch.cat([x, append], dim=2)

def spectral_idfs(x):
    b, c, h, w = x.shape
    phase_shift = torch.ones(w, device=x.device, dtype=x.dtype)
    phase_shift[1::2] *= -1
    xdfs = torch.flip(x[:, :, h//2:], dims=(2,))
    xdfs *= phase_shift[None, None, None, :]
    return .5*(x[:, :, :h//2] + xdfs)

def spectral_dfs_padding(x, padding, ew_padding=0):
    if padding==0:
        return x
    b, c, h, w = x.shape
    out = F.pad(x, (0, 0, padding, padding), 'reflect')
    phase_shift = torch.ones(w, device=x.device, dtype=x.dtype)
    phase_shift[1::2] *= -1
    out[:, :, :padding] *= phase_shift[None, None, None, :]
    out[:, :, -padding:] *= phase_shift[None, None, None, :]
    if ew_padding != 0:
        out = F.pad(out, (ew_padding, 0, 0, 0), 'reflect')
        out = F.pad(out, (0, ew_padding, 0, 0))
        out[:, :, :, :ew_padding] = out[:, :, :, :ew_padding].conj()
        #out = torch.cat([left, F.pad(out, (0, ew_padding, 0, 0))], -1)
    return out


class SlowComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super().__init__()
        padding_raw = [k//2 for k in kernel_size]
        self.padding = (padding_raw[1], padding_raw[1], padding_raw[0], padding_raw[0])
        self.groups = groups
        self.kernel_size = kernel_size
        temp_conv = nn.Conv2d(in_channels, out_channels, kernel_size, groups=groups, dtype=torch.cfloat)
        self.weight = nn.Parameter(torch.view_as_real(temp_conv.weight))
        #print(temp_conv.weight.shape)
            
            
    def forward(self, x):
        x = spectral_dfs_padding(x, self.padding[-1], self.padding[0])
        return F.conv2d(x, torch.view_as_complex(self.weight), groups=self.groups)#F.fold(x, (h, w), (1, 1))

class SlowComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super().__init__()
        self.padding = kernel_size//2
        self.groups = groups
        self.kernel_size = kernel_size
        temp_conv = nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups, dtype=torch.cfloat)
        self.weight = nn.Parameter(torch.view_as_real(temp_conv.weight))
        #print(temp_conv.weight.shape)


    def forward(self, x):
        x = spectral_dfs_padding(x, self.padding)
        return F.conv1d(x, torch.view_as_complex(self.weight), groups=self.groups)#F.fold(x, (h, w), (1, 1))


class SpectralLN(nn.Module):
    def __init__(self, grid_shape):
        super().__init__()
        h, w = grid_shape
        self.mags = nn.Parameter(torch.ones(1, 1, 2*h, w//2+1).float())
        self.bias = nn.Parameter(torch.zeros(1, 1,2* h, w//2+1, 2))

    def forward(self, x, eps=1e-12):
        dtype = x.dtype
        x = torch.fft.rfft2(sphere_to_torus(x), norm='ortho')
        mu = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        x = (x - mu) / (std+eps)
        x = x*self.mags + torch.view_as_complex(self.bias)
        x = torch.fft.irfft2(x.cdouble(), norm='ortho').to(dtype)
        x = .5*(x[:, :, :x.shape[2]//2] + glide_reflection(x[:, :, x.shape[2]//2:]))
        return x
################################################################
# 1d fourier layer
################################################################



class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, s=[x.size(-1)], dim=[2])
        return x

################################################################
# 2d fourier layer
################################################################

class Diffusion2d(nn.Module):
    """
    Implements implicit cartesian diffusion/hyperdiffusion 
    with learnable step sizes.

    Note - the signs in the diffusion matrix are technically opposites
    but the convention is to use opposing signs on the differentiation 
    operators.
    """
    def __init__(self, in_channels, diffusion_offset=0):
        super().__init__()
        self.nu = nn.Parameter(-5*torch.ones(in_channels)) # LR sort of built in
        self.nu2 = nn.Parameter(-5*torch.ones(in_channels)) # LR sort of built in
        self.diffusion_offset=diffusion_offset

    def forward(self, x):
        dtype = x.dtype
        b, c, h, w = x.shape
        x = sphere_to_torus(x)
        xft = torch.fft.rfft2(x, norm='ortho')
        zonal = 2*np.pi*torch.fft.rfftfreq(w, device=x.device)
        merid = 2*np.pi*torch.fft.fftfreq(h*2, device=x.device)
        nabla2 = zonal[None, :]**2 + merid[:, None]**2
        D2 = (self.diffusion_offset+F.softplus(self.nu2)[None, :, None, None]) * nabla2[None, None, :, :]
        nabla4 = zonal[None, :]**4 + merid[:, None]**4 
        D4 = (self.diffusion_offset+F.softplus(self.nu)[None, :, None, None]) * nabla4[None, None, :, :]
        D = 1 + D2 + D4 # Implicit diff matrix
        xft = xft / D # D is diagonal so we can implement this as elementwise division
        x = torch.fft.irfft2(xft.cdouble(), norm='ortho').to(dtype)
        x = .5*(x[:, :, :h]+glide_reflection(x[:, :, h:]))
        return x

class SphericalDiffusionNS_FD(nn.Module):
    def __init__(self, in_channels, diffusion_offset):
        super().__init__()
        self.in_channels = in_channels
        kernel = torch.tensor([[-.5, 0, .5],
                    [1., -2, 1]])
        self.register_buffer('weight', kernel.reshape(2, 1, 3, 1))
        self.h = nn.Parameter(torch.tensor(-5.))
        self.diffusion_offset = diffusion_offset

    def apply_stencil(self, x, weight, cots):
        x = dfs_padding(x, 1)
        raw_diffs = F.conv2d(x, weight, groups=self.in_channels)
        d1, d2 = raw_diffs.tensor_split(2, dim=1)
        d1 = d1 * cots
        return d1+d2

    def forward(self, x):
        print('step-diff', self.h)
        b, c, h, w =  x.shape
        delta = 1 / (h+1) / 2
        lats = torch.linspace(delta, np.pi-delta, h, device=x.device)
        cots = 1/torch.tan(lats)
        cots = cots[None, None, :, None]
        weight = torch.repeat_interleave(self.weight, self.in_channels, dim=0)
        nabla2 = self.apply_stencil(x, weight, cots)
        return (x + F.softplus(self.h)*(nabla2))  


class Diffusion1d(nn.Module):
    """
    Implements implicit cartesian diffusion/hyperdiffusion 
    with learnable step sizes in meridional direction only.

    Note - the signs in the diffusion matrix are technically opposites
    but the convention is to use opposing signs on the differentiation 
    operators.
    """
    def __init__(self, in_channels, n_lats=128, pre_transformed=True, diffusion_offset=0):
        super().__init__()
        self.nu = nn.Parameter(-5*torch.ones(in_channels, n_lats)) # LR sort of built in
        self.nu2 = nn.Parameter(-5*torch.ones(in_channels, n_lats)) # LR sort of built in
        self.pre_transformed = pre_transformed
        self.offset = diffusion_offset
        self.n_lats = 128
    def forward(self, x):
        dtype = x.dtype
        b, c, h, w = x.shape
        if self.pre_transformed:
            xft = x
            w = (w-1)*2
        else:
            xft = torch.fft.rfft(x, norm='ortho')
        delta = 1/(h+1) / 2
        
        lats = torch.linspace(delta, np.pi-delta, h, device=xft.device)
        zonal = 2*np.pi*torch.fft.rfftfreq(w, device=x.device)[None, :]*(1/torch.sin(lats)[:, None])

        nabla2 = zonal**2
        D2 = (self.offset + F.softplus(self.nu2)[None, :, :, None]) * nabla2[None, None, :, :]
        nabla4 = zonal**4 
        D4 = (self.offset + F.softplus(self.nu)[None, :, :, None]) * nabla4[None, None, :, :]
        D = 1 + D2 + D4 # Implicit diff matrix
        xft = xft / D # D is diagonal so we can implement this as elementwise division
        
        if self.pre_transformed:
            return xft
        else:
            x = torch.fft.irfft(xft.cdouble(), norm='ortho').to(dtype)
            return x



class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x, gridy=None):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2, 3])

        if gridy is None:
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, device=x.device,
                                 dtype=torch.cfloat)
            out_ft[:, :, :self.modes1, :self.modes2] = \
                compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            out_ft[:, :, -self.modes1:, :self.modes2] = \
                compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

            # Return to physical space
            x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[2, 3])
        else:
            factor1 = compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            factor2 = compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
            x = self.ifft2d(gridy, factor1, factor2, self.modes1, self.modes2) / (size1 * size2)
        return x

    def ifft2d(self, gridy, coeff1, coeff2, k1, k2):

        # y (batch, N, 2) locations in [0,1]*[0,1]
        # coeff (batch, channels, kmax, kmax)

        batchsize = gridy.shape[0]
        N = gridy.shape[1]
        device = gridy.device
        m1 = 2 * k1
        m2 = 2 * k2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=k1, step=1), \
                            torch.arange(start=-(k1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=k2, step=1), \
                            torch.arange(start=-(k2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(gridy[:,:,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(gridy[:,:,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (N, m1, m2)
        basis = torch.exp( 1j * 2* np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        coeff3 = coeff1[:,:,1:,1:].flip(-1, -2).conj()
        coeff4 = torch.cat([coeff1[:,:,0:1,1:].flip(-1).conj(), coeff2[:,:,:,1:].flip(-1, -2).conj()], dim=-2)
        coeff12 = torch.cat([coeff1, coeff2], dim=-2)
        coeff43 = torch.cat([coeff4, coeff3], dim=-2)
        coeff = torch.cat([coeff12, coeff43], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", coeff, basis)
        Y = Y.real
        return Y


class SpectralConv2dV2(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, aa_factor=1.):
        super(SpectralConv2dV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        #self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1+1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        
    def forward(self, x: torch.Tensor, zonal_filter=None, twod_filter=None):
        size_0 = x.size(-2)
        size_1 = x.size(-1)
        batchsize = x.shape[0]
        dtype=x.dtype 

        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x.float(), dim=(-2,-1), norm='ortho')
        x_ft = torch.view_as_real(x_ft)

        out_ft = torch.zeros(batchsize, self.out_channels,  size_0, size_1//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d_v2(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d_v2(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft = torch.view_as_complex(out_ft)

        #Return to physical space
        x = ifft2_and_filter(out_ft, size_0, size_1, zonal_filter, twod_filter).to(dtype)
        #x = torch.fft.irfft2(out_ft.double(), dim=(-2,-1), norm='ortho', s=(size_0, size_1)).to(dtype)
        return x


class DSConvSpectralFNO(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size, modes1, modes2, num_layers=4, diffusion_offset=0):
        super(DSConvSpectralFNO, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        # The num_layers * .25 is from fixup init since we're using residual connections without an init
        self.spectral_mult = dummy_spectral_mult()
        self.slow_conv = spectral_norm(nn.Conv2d(in_channels, in_channels, (11, 1), bias=False, groups=in_channels),
                depth_separable=True)

        n, m = grid_size
        delta = 1 / (n+1) / 2
        lats = torch.linspace(delta, np.pi-delta, n)
        k = torch.fft.rfftfreq(m)*m
        rel_circ = 1 / torch.sin(lats)
        waves = k[None, :] * rel_circ[:, None]

        inds = torch.searchsorted(k, waves.flatten()).reshape(n, m//2+1)
        useable_inds = (inds < m//2+1)
        used_inds = torch.masked_select(inds, useable_inds)
        used_waves =  torch.masked_select(waves, useable_inds)
        total_inds = torch.stack([torch.maximum(used_inds-1, torch.tensor(0)), used_inds]).T
        coefs = (used_waves - used_inds).abs()
        coefs = torch.stack([coefs, 1-coefs]).T
        self.register_buffer('useable_inds', useable_inds)
        self.register_buffer('total_inds', total_inds)
        self.register_buffer('coefs', coefs)
        self.lat_embs = nn.Parameter(.02*torch.view_as_real(torch.randn(used_inds.shape[0], dtype=torch.cfloat)))


        conv = torch.view_as_real(torch.randn(1, in_channels, used_inds.shape[0], dtype=torch.cfloat))
        self.mags = nn.Parameter(conv[..., 0])
        self.phases = nn.Parameter(conv[..., 1])

    def forward(self, x: torch.Tensor, zonal_filter=None, twod_filter=None):
        size_0 = x.size(-2)
        size_1 = x.size(-1)
        batchsize = x.shape[0]
        dtype=x.dtype
        x = x.float()
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, dim=(-1), norm='ortho')
        x_masked = torch.masked_select(x_ft, self.useable_inds[None, None, :, :]).reshape(batchsize, x.shape[1], -1)
        conv = torch.polar(torch.sigmoid(self.mags), self.phases)
        x_masked = x_masked * conv

        x_ft = torch.zeros_like(x_ft)
        x_ft[:, :, self.useable_inds] = x_masked
        x = torch.fft.irfft(x_ft.cdouble(), norm='ortho').to(dtype)
        x_pad = dfs_padding(x, 5)
        z = self.slow_conv(x_pad)
        x = x + z
        return x

class DSConvSpectral(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size, modes1, modes2, num_layers=4, diffusion_offset=0):
        super(DSConvSpectral, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        # The num_layers * .25 is from fixup init since we're using residual connections without an init
        self.spectral_mult = dummy_spectral_mult()
        self.slow_conv = spectral_norm(nn.Conv2d(in_channels, in_channels, (11, 1), bias=False, groups=in_channels),
                depth_separable=True)

        self.mlp = nn.Sequential(SlowComplexConv1d(in_channels, in_channels, 1),
                mod_relu(in_channels),
                SlowComplexConv1d(in_channels, in_channels, 1)
                )
        n, m = grid_size
        delta = 1 / (n+1) / 2
        lats = torch.linspace(delta, np.pi-delta, n)
        k = torch.fft.rfftfreq(m)*m
        rel_circ = 1 / torch.sin(lats)
        waves = k[None, :] * rel_circ[:, None]

        inds = torch.searchsorted(k, waves.flatten()).reshape(n, m//2+1)
        useable_inds = (inds < m//2+1)
        used_inds = torch.masked_select(inds, useable_inds)
        used_waves =  torch.masked_select(waves, useable_inds)
        total_inds = torch.stack([torch.maximum(used_inds-1, torch.tensor(0)), used_inds]).T
        coefs = (used_waves - used_inds).abs()
        coefs = torch.stack([coefs, 1-coefs]).T
        self.register_buffer('useable_inds', useable_inds)
        self.register_buffer('total_inds', total_inds)
        self.register_buffer('coefs', coefs)
        self.lat_embs = nn.Parameter(.02*torch.view_as_real(torch.randn(used_inds.shape[0], dtype=torch.cfloat)))


        conv = torch.view_as_real(torch.randn(1, in_channels, used_inds.shape[0], dtype=torch.cfloat))
        self.mags = nn.Parameter(conv[..., 0])
        self.phases = nn.Parameter(conv[..., 1])
        self.freq_mags = nn.Parameter(torch.view_as_real(torch.ones(1, 1, used_inds.shape[0], dtype=torch.cfloat)))

        self.spectral_rescale = SpectralRescale(used_inds.shape[0], in_channels)
        self.register_buffer('means', torch.zeros(used_inds.shape[0]))
        self.register_buffer('stds', torch.ones(used_inds.shape[0]))
        
        self.diffusion_offset = diffusion_offset
        self.ns_scaling_mag = nn.Parameter(torch.randn(in_channels, 10))
        self.ns_scaling_phase = nn.Parameter(torch.randn(in_channels, 10))
        #self.mlp = nn.Conv2d(in_channels, out_channels, 1)
        #self.diffusion = Diffusion1d(out_channels, diffusion_offset=diffusion_offset)
        #self.post_diffusion = SphericalDiffusionNS_FD(out_channels, diffusion_offset=diffusion_offset)
        # Change to just init weights without extra layer later
        #temp_conv = nn.Linear(in_channels*2, in_channels, dtype=torch.cfloat)
        #self.mlp_weight = nn.Parameter(torch.view_as_real(temp_conv.weight * num_layers**.25))


    def build_conv(self, size_0, size_1):
        ew_mags = torch.zeros(self.in_channels, size_0, size_1//2+1, device=self.ew_mags.device, dtype=self.ew_mags.dtype)
        ew_phases = torch.zeros(self.in_channels, size_0, size_1//2+1, device=self.ew_mags.device, dtype=self.ew_mags.dtype)
        flat_mags = self.ew_mags.index_select(1, self.total_inds.flatten()).reshape(-1, *self.total_inds.shape)
        flat_mags = torch.einsum('btc,tc->bt', flat_mags,self.coefs)
        flat_phases = self.ew_phases.index_select(1, self.total_inds.flatten()).reshape(-1, *self.total_inds.shape)
        flat_phases = torch.einsum('btc,tc->bt', flat_phases,self.coefs)

        ew_mags[:, self.useable_inds] = flat_mags
        ew_phases[:, self.useable_inds] = flat_phases
        ew_mags = ew_mags.unsqueeze(0)
        ew_phases = ew_phases.unsqueeze(0)
        return torch.tanh(ew_mags)*torch.exp(1j*ew_phases)

    def build_interp_conv(self, size_0, size_1):
        dtype = self.ew_mags.dtype
        mmags = torch.sigmoid(self.ew_mags)
        mags = torch.fft.rfft(mmags,dim=1, norm='forward')
        mags = torch.fft.irfft(mags.cdouble(), dim=1, norm='forward', n=size_0).to(dtype)

        #phases = self.ew_pmags*torch.exp(1j*self.ew_phases)
        phases = torch.fft.rfft(self.ew_phases, dim=1, norm='forward')
        phases = torch.fft.irfft(phases.cdouble(), dim=1, norm='forward', n=size_0).to(dtype)
        
        return (mags * torch.exp(1j*phases)).unsqueeze(0) 

    def build_full_conv(self, size_0, size_1):
        conv = torch.tanh(self.mags)*torch.exp(self.phases*1j)
        x_ft = torch.zeros(*conv.shape[:2], size_0, size_1, dtype=conv.dtype)
        x_ft[:, :, self.useable_inds] = conv
        return x_ft

    def forward(self, x: torch.Tensor, zonal_filter=None, twod_filter=None):
        size_0 = x.size(-2)
        size_1 = x.size(-1)
        batchsize = x.shape[0]
        #x = dfs_padding(x, 5)
        #x = F.pad(x, (1, 1, 0, 0), 'circular')
        dtype=x.dtype
        x = x.float()
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        #x_ft = self.slow_conv(x_ft)
        x_ft = torch.fft.rfft(x, dim=(-1), norm='ortho')
        
        x_masked = torch.masked_select(x_ft, self.useable_inds[None, None, :, :]).reshape(batchsize, x.shape[1], -1)
        x_linear = self.mlp(self.spectral_rescale(x_masked))
        conv = complex_glu(x_linear, self.mags, self.phases)
        #conv = torch.sigmoid(self.mags)*torch.exp(self.phases*1j) #self.build_interp_conv(size_0, size_1)
        x_masked = x_masked * conv#torch.tanh(ew_mags)*torch.exp(ew_phases*1j)
        #x_ft = torch.fft.rfft2(x, dim=(-2,-1), norm='ortho')
        #x_masked = rearrange(x_masked, 'b c t -> b t c')
        #x_masked = F.linear(x_masked, torch.view_as_complex(self.mlp_weight))
        # Weird fft bug introduces high frequency noise at single precision/gpu if order is swapped
        #x_masked = rearrange(x_masked, 'b t c -> b c t')
        


        x_ft = torch.zeros_like(x_ft)
        x_ft[:, :, self.useable_inds] = x_masked
        #x_ft = self.diffusion(x_ft)
        x = torch.fft.irfft(x_ft.cdouble(), norm='ortho').to(dtype)
        #x = self.post_diffusion(x)
        x_pad = dfs_padding(x, 5)
        z = self.slow_conv(x_pad)
       # scale = torch.fft.irfft(torch.polar(torch.sigmoid(self.ns_scaling_mag), self.ns_scaling_phase), norm='forward', n=size_0)
       # z = z * scale[None, :, :, None]
        x = x + z
        
        
        #x = ifft2_and_filter(x_ft, size_0, size_1, zonal_filter, twod_filter).to(dtype)
        #Return to physical space
        #x = torch.fft.irfft2(x_ft, dim=(-2,-1), norm='ortho', s=(size_0, size_1)).to(dtype)
        return x

class DSConvSpectral2d(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size, modes1, modes2, num_layers=4, diffusion_offset=0):
        super(DSConvSpectral2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        # The num_layers * .25 is from fixup init since we're using residual connections without an init
        self.spectral_mult = dummy_spectral_mult()

        self.mlp = nn.Sequential(SlowComplexConv1d(in_channels, in_channels, 1),
                mod_relu(in_channels),
                SlowComplexConv1d(in_channels, in_channels, 1)
                )
        n, m = grid_size
        delta = 1 / (n+1) / 2
        lats = torch.linspace(delta, np.pi-delta, n)
        k = torch.fft.rfftfreq(m)*m
        rel_circ = torch.fft.fftfreq(n*2)*n*2
        waves = (k[None, :]**2 + rel_circ[:, None]**2)**.5

        #inds = torch.searchsorted(k, waves.flatten()).reshape(n, m//2+1)
        useable_inds = waves <= m//2
        used_inds = torch.masked_select(waves, useable_inds) # This is a dummy - just using size
        self.register_buffer('useable_inds', useable_inds)
        self.lat_embs = nn.Parameter(.02*torch.view_as_real(torch.randn(used_inds.shape[0], dtype=torch.cfloat)))


        conv = torch.view_as_real(torch.randn(1, in_channels, used_inds.shape[0], dtype=torch.cfloat))
        self.mags = nn.Parameter(conv[..., 0])
        self.phases = nn.Parameter(conv[..., 1])
        self.freq_mags = nn.Parameter(torch.view_as_real(torch.ones(1, 1, used_inds.shape[0], dtype=torch.cfloat)))

        self.spectral_rescale = SpectralRescale(used_inds.shape[0], in_channels)
        self.register_buffer('means', torch.zeros(used_inds.shape[0]))
        self.register_buffer('stds', torch.ones(used_inds.shape[0]))
        
        self.diffusion_offset = diffusion_offset
        self.ns_scaling_mag = nn.Parameter(torch.randn(in_channels, 10))
        self.ns_scaling_phase = nn.Parameter(torch.randn(in_channels, 10))
        #self.mlp = nn.Conv2d(in_channels, out_channels, 1)
        #self.diffusion = Diffusion1d(out_channels, diffusion_offset=diffusion_offset)
        #self.post_diffusion = SphericalDiffusionNS_FD(out_channels, diffusion_offset=diffusion_offset)
        # Change to just init weights without extra layer later
        #temp_conv = nn.Linear(in_channels*2, in_channels, dtype=torch.cfloat)
        #self.mlp_weight = nn.Parameter(torch.view_as_real(temp_conv.weight * num_layers**.25))


    def build_conv(self, size_0, size_1):
        ew_mags = torch.zeros(self.in_channels, size_0, size_1//2+1, device=self.ew_mags.device, dtype=self.ew_mags.dtype)
        ew_phases = torch.zeros(self.in_channels, size_0, size_1//2+1, device=self.ew_mags.device, dtype=self.ew_mags.dtype)
        flat_mags = self.ew_mags.index_select(1, self.total_inds.flatten()).reshape(-1, *self.total_inds.shape)
        flat_mags = torch.einsum('btc,tc->bt', flat_mags,self.coefs)
        flat_phases = self.ew_phases.index_select(1, self.total_inds.flatten()).reshape(-1, *self.total_inds.shape)
        flat_phases = torch.einsum('btc,tc->bt', flat_phases,self.coefs)

        ew_mags[:, self.useable_inds] = flat_mags
        ew_phases[:, self.useable_inds] = flat_phases
        ew_mags = ew_mags.unsqueeze(0)
        ew_phases = ew_phases.unsqueeze(0)
        return torch.tanh(ew_mags)*torch.exp(1j*ew_phases)

    def build_interp_conv(self, size_0, size_1):
        dtype = self.ew_mags.dtype
        mmags = torch.sigmoid(self.ew_mags)
        mags = torch.fft.rfft(mmags,dim=1, norm='forward')
        mags = torch.fft.irfft(mags.cdouble(), dim=1, norm='forward', n=size_0).to(dtype)

        #phases = self.ew_pmags*torch.exp(1j*self.ew_phases)
        phases = torch.fft.rfft(self.ew_phases, dim=1, norm='forward')
        phases = torch.fft.irfft(phases.cdouble(), dim=1, norm='forward', n=size_0).to(dtype)
        
        return (mags * torch.exp(1j*phases)).unsqueeze(0) 

    def build_full_conv(self, size_0, size_1):
        conv = torch.sigmoid(self.mags)*torch.exp(self.phases*1j)
        x_ft = torch.zeros(*conv.shape[:2], size_0, size_1, dtype=conv.dtype)
        x_ft[:, :, self.useable_inds] = conv
        return x_ft

    def forward(self, x: torch.Tensor, zonal_filter=None, twod_filter=None):
        size_0 = x.size(-2)
        size_1 = x.size(-1)
        batchsize = x.shape[0]
        #x = dfs_padding(x, 5)
        #x = F.pad(x, (1, 1, 0, 0), 'circular')
        dtype=x.dtype
        x = x.float()
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        #x_ft = self.slow_conv(x_ft)
        x_ft = torch.fft.rfft(x, dim=(-1), norm='ortho')
        x_ft = spectral_dfs(x_ft)
        x_ft = torch.fft.fft(x_ft, dim=-2, norm='ortho')
        x_masked = torch.masked_select(x_ft, self.useable_inds[None, None, :, :]).reshape(batchsize, x.shape[1], -1)
        x_linear = self.mlp(self.spectral_rescale(x_masked))
        conv = complex_glu(x_linear, self.mags, self.phases)
        #conv = torch.sigmoid(self.mags)*torch.exp(self.phases*1j) #self.build_interp_conv(size_0, size_1)
        x_masked = x_masked * conv#torch.tanh(ew_mags)*torch.exp(ew_phases*1j)
        #x_ft = torch.fft.rfft2(x, dim=(-2,-1), norm='ortho')
        #x_masked = rearrange(x_masked, 'b c t -> b t c')
        #x_masked = F.linear(x_masked, torch.view_as_complex(self.mlp_weight))
        # Weird fft bug introduces high frequency noise at single precision/gpu if order is swapped
        #x_masked = rearrange(x_masked, 'b t c -> b c t')
        


        x_ft = torch.zeros_like(x_ft)
        x_ft[:, :, self.useable_inds] = x_masked
        #x_ft = self.diffusion(x_ft)
        x_ft = torch.fft.ifft(x_ft, dim=-2, norm='ortho')
        x_ft = spectral_idfs(x_ft)
        x = torch.fft.irfft(x_ft.cdouble(), norm='ortho').to(dtype) 
        return x

class DSConvReducedSpectral(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size, modes1, modes2,
            interp_size=(10, 10), aa_factor=1., conv_interp_type='fourier'):
        super(DSConvReducedSpectral, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.grid_size = grid_size
        self.conv_interp_type = conv_interp_type

        self.small_conv = nn.Parameter(torch.view_as_real(torch.randn(1, in_channels, *interp_size, dtype=torch.cfloat)))
        # Fix this later
        temp_conv = nn.Linear(in_channels, out_channels, dtype=torch.cfloat)
        self.mlp_weight = nn.Parameter(torch.view_as_real(temp_conv.weight))

    def forward(self, x: torch.Tensor, zonal_filter=None, twod_filter=None):
        size_0 = x.size(-2)
        size_1 = x.size(-1)
        batchsize = x.shape[0]
        dtype=x.dtype
        x = x.float()
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, dim=(-2,-1), norm='ortho')
        
        #conv = self.weight_gen(x).to(x.dtype)
        #conv = torch.view_as_complex(conv.reshape(*conv.shape[:-1], conv.shape[-1]//2, 2))
        if self.conv_interp_type == 'fourier':
            conv = torch.fft.ifft2(torch.view_as_complex(self.small_conv), s=self.grid_size, dim=(-2, -1), norm='ortho')
            conv = conv[..., :self.grid_size[1]//2+1]
        else:
            conv = rearrange(self.small_conv, 'b c h w i -> b (c i) h w')
            conv = F.interpolate(conv, size=(self.grid_size[0], self.grid_size[1]//2+1), mode=self.conv_interp_type)
            conv = torch.view_as_complex(rearrange(conv, 'b (c i) h w -> b c h w i', i=2).contiguous())
        x_ft = x_ft * conv
        x_ft = rearrange(x_ft, 'b c h w -> b h w c')
        x_ft = F.linear(x_ft, torch.view_as_complex(self.mlp_weight))
        x_ft = rearrange(x_ft, 'b h w c -> b c h w')
        #Return to physical space
        x = ifft2_and_filter(x_ft, size_0, size_1, zonal_filter, twod_filter).to(dtype)
        return x


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3,4])
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(2), x.size(3), x.size(4)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(2), x.size(3), x.size(4)), dim=[2,3,4])
        return x


class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, activation='tanh'):
        super(FourierBlock, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.speconv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)
        self.linear = nn.Conv1d(in_channels, out_channels, 1)
        if activation == 'tanh':
            self.activation = torch.tanh_
        elif activation == 'gelu':
            self.activation = nn.GELU
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        '''
        input x: (batchsize, channel width, x_grid, y_grid, t_grid)
        '''
        x1 = self.speconv(x)
        x2 = self.linear(x.view(x.shape[0], self.in_channel, -1))
        out = x1 + x2.view(x.shape[0], self.out_channel, x.shape[2], x.shape[3], x.shape[4])
        if self.activation is not None:
            out = self.activation(out)
        return out


class DFSEmbedding(nn.Module):
    def __init__(self, hidden_features, grid_shape, patch_size=(8, 8), samples_per_year=1460, sph_order=10, max_time_freq=4, layer_size=500, scale_factor=.02,
                 sphere=True, learned_sphere=False, global_dfs=True, steps_per_day=4, steps_per_year=1460, include_pole=False, bw_limit=None):
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
        self.scale_factor = scale_factor
        self.samples_per_year = samples_per_year
        self.steps_per_day = steps_per_day
        self.bw_order= sph_order


        if bw_limit is None:
            self.bw_limit = grid_shape[1]//2
        self.bw_limit = grid_shape[1]//2
        num_feats = self.build_features()


        self.mlp = nn.Sequential(nn.Linear(num_feats, layer_size),
            nn.GLU(),
            nn.Linear(layer_size//2, hidden_features, bias=False)
            )


    def resize(self, grid_shape):
        self.grid_shape = grid_shape
        self.build_features()

    @torch.no_grad()
    def build_features(self, initial=False):
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
        if not self.learned_sphere:
            space_features = spherical_harmonics_alpha_beta(list(range(1, self.sph_order+1)), grid_coords[:, 1], grid_coords[:, 0])
        self.register_buffer('time_coords', grid_coords[:, 1].reshape(1, -1)/(2*np.pi), persistent=False) # Format is (n_points, [lat, lon])
        self.register_buffer('lats', grid_coords[:, 0].unsqueeze(0), persistent=False)
        self.register_buffer('time_base', (torch.arange(1, self.max_time_freq+1)*2*np.pi).reshape(1, -1), persistent=False)
        if self.daylight_feats:
            num_feats = 2*self.time_base.shape[1]+1 # Space + time of day + daylight hours
        else:
                num_feats = 4*self.time_base.shape[1]

        if self.learned_sphere:
            mesh = ((torch.fft.fftfreq(self.grid_shape[0]*2)*self.bw_limit)[:, None]**2 + (torch.fft.rfftfreq(self.grid_shape[1])*self.bw_limit)[None, :]**2).sqrt() <= self.bw_limit
            self.register_buffer('used_mask', mesh)
            space_features = torch.randn(self.hidden_features // 4, self.used_mask.sum(), dtype=torch.cfloat) * .02
            self.space_features = nn.Parameter(torch.view_as_real(space_features))
        else:
            self.register_buffer('space_features', space_features, persistent=False)
        self.space_feats_dim = space_features.shape[-1]
        num_feats += self.hidden_features // 4
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
                daylight = daylight.unsqueeze(1).expand(-1, time_coords.shape[1], -1)
        xprod_time = time.unsqueeze(-1)*self.time_base.unsqueeze(0)
        coses_time = torch.cos(xprod_time)
        sins_time = torch.sin(xprod_time)
        return [coses_time, sins_time, daylight]

    def build_space_features(self):
        space_features = self.space_features
        mags = self.space_features[..., 0]
        phases = self.space_features[..., 0]
        spect = torch.polar(torch.sigmoid(mags), phases)
        full_spect = space_features.new_zeros(mags.shape[0], self.bw_limit*2, self.bw_limit+1, dtype=torch.view_as_complex(space_features).dtype)
        full_spect[:, self.used_mask[:, :]] = spect 
        with torch.cuda.amp.autocast(enabled=False):
            spatial = torch.fft.irfft2(full_spect.cdouble(), norm='backward').float()
        if self.include_pole:
            pass
        else:
            out = spatial[:, :, :spatial.shape[2]//2]

        return rearrange(out, 'c h w -> (h w) c')
            

    def forward(self, dt_ind, augs=None):
        # Remember to fix this for larger batches
        dt_ind = dt_ind.reshape(-1, 1)
        if self.sphere:
            space_features = self.build_space_features()
            print('space_features', space_features.shape)
            feats = [space_features.unsqueeze(0).expand(dt_ind.shape[0], -1, -1)]
            with record_function('DT feats'):
                dt_feats = self.datetime_features(dt_ind)
            feats += dt_feats
            feats = torch.cat(feats, -1)
            print(feats.shape)
        else:
            feats = self.space_features
        if self.sphere: # Sphere is batch element dependent
            out = self.scale_factor*self.mlp(feats).reshape(dt_ind.shape[0], *self.out_shape, self.hidden_features)
        else: # Frequency is not
            out = self.scale_factor*self.mlp(feats).reshape(1, *self.out_shape, self.hidden_features)
        #if self.sphere and self.global_dfs:
        #    out = sphere_to_torus(out, glide_dim=2, flip_dim=1)
        return out

