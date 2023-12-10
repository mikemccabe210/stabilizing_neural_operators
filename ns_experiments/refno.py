''' from original FNO repo '''
import torch
import numpy as np
import torch.nn as nn


from spectral_norm_utils import spectral_norm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from einops import rearrange



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
    return torch.stack([tmp[0, 0, :, :, :, :] - tmp[1, 1, :, :, :, :], tmp[1, 0, :, :, :, :] + tmp[0, 1, :, :, :, :]],
                       dim=-1)


def ifft2_and_filter(x, size_0, size_1, zonal_filter=None, twod_filter=None):
    if twod_filter is not None:
        x = x * twod_filter
    if zonal_filter is not None:
        x = torch.fft.ifft(x, dim=-2, norm='ortho', n=size_0)
        x = x * zonal_filter
        return torch.fft.irfft(x.cdouble(), dim=-1, norm='ortho', n=size_1)
    else:
        return torch.fft.irfft2(x.cdouble(), dim=(-2, -1), norm='ortho', s=(size_0, size_1))


class SpectralRescale(nn.Module):
    def __init__(self, h, channels, momentum=.1, affine=True, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.mags = nn.Parameter(torch.view_as_real(torch.ones(1, channels, h, dtype=torch.cfloat)))
            self.bias = nn.Parameter(torch.view_as_real(torch.zeros(1, channels, h, dtype=torch.cfloat)))
        self.register_buffer('mean', torch.view_as_real(torch.zeros(1, 1, h, dtype=torch.cfloat)))
        self.register_buffer('std', torch.ones(1, 1, h))
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            mean, std = x.mean((0, 1), keepdims=True), x.std((0, 1), keepdims=True)
            with torch.no_grad():
                self.mean.copy_((1 - self.momentum) * self.mean + self.momentum * torch.view_as_real(mean))
                self.std.copy_((1 - self.momentum) * self.std + self.momentum * std)
        else:
            mean, std = torch.view_as_complex(self.mean), self.std
        #
        x = (x - mean) / (self.eps + std)
        if self.affine:
            x = x * torch.view_as_complex(self.mags) + torch.view_as_complex(self.bias)
        return x


class mod_relu(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return torch.polar(F.gelu(torch.abs(x) + self.b), x.angle())


class complex_silu(nn.Module):
    def forward(self, x):
        return torch.complex(F.silu(x.real), F.silu(x.imag))

def complex_glu(x, mag_bias, phase_bias):
    mag, phase = x.abs(), x.angle()
    return torch.polar(torch.sigmoid(mag + mag_bias), phase + phase_bias)


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
    xdfs = torch.flip(x[:, :, h // 2:], dims=(2,))
    xdfs *= phase_shift[None, None, None, :]
    return .5 * (x[:, :, :h // 2] + xdfs)


def spectral_dfs_padding(x, padding, ew_padding=0):
    if padding == 0:
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
        # out = torch.cat([left, F.pad(out, (0, ew_padding, 0, 0))], -1)
    return out


class SlowComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super().__init__()
        padding_raw = [k // 2 for k in kernel_size]
        self.padding = (padding_raw[1], padding_raw[1], padding_raw[0], padding_raw[0])
        self.groups = groups
        self.kernel_size = kernel_size
        temp_conv = nn.Conv2d(in_channels, out_channels, kernel_size, groups=groups, dtype=torch.cfloat)
        self.weight = nn.Parameter(torch.view_as_real(temp_conv.weight))
        # print(temp_conv.weight.shape)

    def forward(self, x):
        x = spectral_dfs_padding(x, self.padding[-1], self.padding[0])
        return F.conv2d(x, torch.view_as_complex(self.weight), groups=self.groups)  # F.fold(x, (h, w), (1, 1))


class SlowComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super().__init__()
        self.padding = kernel_size // 2
        self.groups = groups
        self.kernel_size = kernel_size
        temp_conv = nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups, dtype=torch.cfloat)
        self.weight = nn.Parameter(torch.view_as_real(temp_conv.weight))
        # print(temp_conv.weight.shape)

    def forward(self, x):
        x = spectral_dfs_padding(x, self.padding)
        return F.conv1d(x, torch.view_as_complex(self.weight), groups=self.groups)  # F.fold(x, (h, w), (1, 1))


################################################################
# 1d fourier layer
################################################################



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
        k_x1 = torch.cat((torch.arange(start=0, end=k1, step=1), \
                          torch.arange(start=-(k1), end=0, step=1)), 0).reshape(m1, 1).repeat(1, m2).to(device)
        k_x2 = torch.cat((torch.arange(start=0, end=k2, step=1), \
                          torch.arange(start=-(k2 - 1), end=0, step=1)), 0).reshape(1, m2).repeat(m1, 1).to(device)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(gridy[:, :, 0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(gridy[:, :, 1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (N, m1, m2)
        basis = torch.exp(1j * 2 * np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        coeff3 = coeff1[:, :, 1:, 1:].flip(-1, -2).conj()
        coeff4 = torch.cat([coeff1[:, :, 0:1, 1:].flip(-1).conj(), coeff2[:, :, :, 1:].flip(-1, -2).conj()], dim=-2)
        coeff12 = torch.cat([coeff1, coeff2], dim=-2)
        coeff43 = torch.cat([coeff4, coeff3], dim=-2)
        coeff = torch.cat([coeff12, coeff43], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", coeff, basis)
        Y = Y.real
        return Y

class DSConvSpectralFNO(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size, modes1, modes2, num_layers=4, ratio=1.0):
        super(DSConvSpectralFNO, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        n, m = grid_size
        k = torch.fft.rfftfreq(m) * m
        rel_circ = torch.fft.fftfreq(n) * n
        waves = (k[None, :] ** 2 + rel_circ[:, None] ** 2) ** .5
        # inds = torch.searchsorted(k, waves.flatten()).reshape(n, m//2+1)
        useable_inds = waves <= ((m // 2) * ratio)
        self.register_buffer('useable_inds', useable_inds)

        # The num_layers * .25 is from fixup init since we're using residual connections without an init
        self.oxo_conv = spectral_norm(nn.Conv2d(in_channels, in_channels, 1))

        n, m = grid_size
        conv = torch.view_as_real(torch.randn(1, in_channels, n, m//2+1, dtype=torch.cfloat))
        self.mags = nn.Parameter(conv[..., 0])
        self.phases = nn.Parameter(conv[..., 1])

    def forward(self, x: torch.Tensor, zonal_filter=None, twod_filter=None):
        batchsize = x.shape[0]
        dtype = x.dtype
        x = x.float()
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, norm='ortho')
        x_masked = torch.masked_select(x_ft, self.useable_inds[None, None, :, :]).reshape(batchsize, x.shape[1], -1)
        conv = torch.masked_select(torch.polar(torch.sigmoid(self.mags), self.phases),  self.useable_inds[None, None, :, :]).reshape(x.shape[1], -1)
        x_masked = x_masked * conv
        # print(conv.shape)
        x_ft = torch.zeros_like(x_ft)
        x_ft[:, :, self.useable_inds] = x_masked
        x = torch.fft.irfft2(x_ft.cdouble(), norm='ortho').to(dtype)
        x = self.oxo_conv(x)
        return x


class DSConvSpectral2d(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size, modes1, modes2, num_layers=4, ratio=1.):
        super(DSConvSpectral2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.mlp = nn.Sequential(SlowComplexConv1d(in_channels, in_channels, 1),
                                 mod_relu(in_channels),
                                 SlowComplexConv1d(in_channels, in_channels, 1)
                                 )
        self.channel_mix = spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        n, m = grid_size
        delta = 1 / (n + 1) / 2
        k = torch.fft.rfftfreq(m) * m
        rel_circ = torch.fft.fftfreq(n) * n
        waves = (k[None, :] ** 2 + rel_circ[:, None] ** 2) ** .5

        # inds = torch.searchsorted(k, waves.flatten()).reshape(n, m//2+1)
        useable_inds = waves <= ((m // 2) * ratio)
        used_inds = torch.masked_select(waves, useable_inds)  # This is a dummy - just using size
        self.register_buffer('useable_inds', useable_inds)
        conv = torch.view_as_real(torch.randn(1, in_channels, used_inds.shape[0], dtype=torch.cfloat))
        self.mags = nn.Parameter(conv[..., 0])
        self.phases = nn.Parameter(conv[..., 1])
        self.spectral_rescale = SpectralRescale(used_inds.shape[0], in_channels)

    def forward(self, x: torch.Tensor, zonal_filter=None, twod_filter=None):
        size_0 = x.size(-2)
        size_1 = x.size(-1)
        batchsize = x.shape[0]
        dtype = x.dtype
        x = x.float()
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, norm='ortho')
        x_masked = torch.masked_select(x_ft, self.useable_inds[None, None, :, :]).reshape(batchsize, x.shape[1], -1)
        x_linear = self.mlp(self.spectral_rescale(x_masked))
        conv = complex_glu(x_linear, self.mags, self.phases)
        x_masked = x_masked * conv

        x_ft = torch.zeros_like(x_ft)
        x_ft[:, :, self.useable_inds] = x_masked
        x = torch.fft.irfft2(x_ft, norm='ortho')
        return self.channel_mix(x)



class ReFNN2d(nn.Module):
    def __init__(self, dim=128,
                 in_dim=2,
                 layers=4,
                 grid_size=(64, 64),
                 nonlinear=False,
                 ratio=1.):
        super(ReFNN2d, self).__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        conv = DSConvSpectral2d if nonlinear else DSConvSpectralFNO
        self.encoder = spectral_norm(nn.Conv2d(in_dim, dim, 1))
        self.spectral_blocks = nn.ModuleList([conv(dim, dim, grid_size, 1, 1, ratio=ratio) for _ in range(layers)])
        self.mlp_blocks = nn.ModuleList([nn.Sequential(spectral_norm(nn.Conv2d(dim, 4*dim, 1)),
                                                       nn.SiLU(),
                                                       spectral_norm(nn.Conv2d(4*dim, dim, 1))) for _ in range(layers)])
        self.gates = nn.ParameterList([nn.Parameter(torch.tensor(0.)) for _ in range(layers)])
        self.final_gate = nn.Parameter(torch.tensor(0.))
        self.decoder = spectral_norm(nn.Conv2d(dim, in_dim, 1))
        self.embed_ratio = dim/in_dim
        self.pos_embedding = nn.Parameter(torch.randn((1, dim)+grid_size) * .02)


    def forward(self, x, *args, **kwargs):
        '''
        (b,c,h,w) -> (b,1,h,w)
        '''
        x = self.encoder(x) * self.embed_ratio ** .5  # project
        x = x + self.pos_embedding
        for i, (speconv, mlp, gate) in enumerate(zip(self.spectral_blocks, self.mlp_blocks, self.gates)):
            x1 = mlp(x)
            x1 = speconv(x1)
            x = x1 + x

        x = self.decoder(x) / self.embed_ratio ** .5
        return x
