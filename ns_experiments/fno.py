import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
from functools import partial


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes, resdiual=True, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.linear = nn.Conv2d(in_dim, out_dim, 1)
        self.residual = resdiual
        self.act = nn.ReLU(inplace=True)

        fourier_weight = [nn.Parameter(torch.FloatTensor(
            in_dim, out_dim, n_modes, n_modes, 2)) for _ in range(2)]
        self.fourier_weight = nn.ParameterList(fourier_weight)
        for param in self.fourier_weight:
            nn.init.xavier_normal_(param, gain=1/(in_dim*out_dim))

    @staticmethod
    def complex_matmul_2d(a, b):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        B, I, M, N = x.shape
        if self.residual:
            res = self.linear(x)
            # res.shape == [batch_size, grid_size, grid_size, out_dim]

        # x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        x_ft = torch.fft.rfft2(x, s=(M, N), norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=4)
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft = torch.zeros(B, I, M, N // 2 + 1, 2, device=x.device)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :self.n_modes, :self.n_modes] = self.complex_matmul_2d(
            x_ft[:, :, :self.n_modes, :self.n_modes], self.fourier_weight[0])

        out_ft[:, :, -self.n_modes:, :self.n_modes] = self.complex_matmul_2d(
            x_ft[:, :, -self.n_modes:, :self.n_modes], self.fourier_weight[1])

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = torch.fft.irfft2(out_ft, s=(N, M), norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        if self.residual:
            x = self.act(x + res)
        else:
            x = self.act(self.linear(x))
        return x


class FNOZongyi2DBlock(nn.Module):
    def __init__(self, modes1, modes2, width, input_dim=2, grid_size=(64, 64), dropout=0.1, n_layers=4, residual=False, conv_residual=True):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.input_dim = input_dim
        self.in_proj = nn.Conv2d(input_dim, self.width, 1)
        self.residual = residual
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.pos_embedding = nn.Parameter(torch.randn((1, width)+grid_size) * .02)

        self.spectral_layers = nn.ModuleList([])
        for i in range(n_layers):
            self.spectral_layers.append(SpectralConv2d(in_dim=width,
                                                       out_dim=width,
                                                       n_modes=modes1,
                                                       resdiual=conv_residual,
                                                       dropout=dropout))

        self.feedforward = nn.Sequential(
            nn.Conv2d(self.width, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1))

    def forward(self, x, **kwargs):
        # x.shape == [n_batches, *dim_sizes, input_size]
        x = self.in_proj(x)
        x = self.pos_embedding + x
        for layer in self.spectral_layers:
            x = layer(x) + x if self.residual else layer(x)

        x = self.feedforward(x)
        # x.shape == [n_batches, *dim_sizes, 1]
        return x