''' From PINO: https://github.com/devzhk/PINO/blob/master/models/basics.py '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from einops import rearrange
try:
    from sphere_tools import ContSphereEmbedding, glide_reflection, sphere_to_torus
    from spectral_norm_utils import spectral_norm
except:
    from .sphere_tools import ContSphereEmbedding, glide_reflection, sphere_to_torus
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
# 2d fourier layer
################################################################

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
        
        self.ns_scaling_mag = nn.Parameter(torch.randn(in_channels, 10))
        self.ns_scaling_phase = nn.Parameter(torch.randn(in_channels, 10))


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
        x = torch.fft.irfft(x_ft.cdouble(), norm='ortho').to(dtype)
        x_pad = dfs_padding(x, 5)
        z = self.slow_conv(x_pad)

        x = x + z

        return x

class DSConvSpectral2d(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size, modes1, modes2, num_layers=4, diffusion_offset=0):
        super(DSConvSpectral2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        # The num_layers * .25 is from fixup init since we're using residual connections without an init
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

        dtype=x.dtype
        x = x.float()
        x_ft = torch.fft.rfft(x, dim=(-1), norm='ortho')
        x_ft = spectral_dfs(x_ft)
        x_ft = torch.fft.fft(x_ft, dim=-2, norm='ortho')
        x_masked = torch.masked_select(x_ft, self.useable_inds[None, None, :, :]).reshape(batchsize, x.shape[1], -1)
        x_linear = self.mlp(self.spectral_rescale(x_masked))
        conv = complex_glu(x_linear, self.mags, self.phases)
        x_masked = x_masked * conv
        x_ft = torch.zeros_like(x_ft)
        x_ft[:, :, self.useable_inds] = x_masked
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



