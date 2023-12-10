''' from original FNO repo '''
import torch
import numpy as np
import torch.nn as nn
try:
    from .fno_utils import SpectralConv2dV2, _get_act, DSConvSpectral, DSConvSpectralFNO, DSConvReducedSpectral, SpectralLN, DSConvSpectral2d, DFSEmbedding
    from .sphere_tools import sphere_to_torus, glide_reflection, ContSphereEmbedding
    from .spectral_norm_utils import spectral_norm
except:
    from fno_utils import SpectralConv2dV2, _get_act, DSConvSpectral, DSConvSpectralFNO, DSConvReducedSpectral, SpectralLN, DSConvSpectral2d, DFSEmbedding
    from projects.stabilizing_neural_operators.networks.sphere_tools import sphere_to_torus, glide_reflection, ContSphereEmbedding
    from spectral_norm_utils import spectral_norm

class FNN2d(nn.Module):
    def __init__(self, modes1, modes2,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=1,
                 activation='tanh',
                 grid_size=(128, 256),
                 dfs_type='full',
                 taper_poles=True,
                 conv_type='fno',
                 aa_rate = 1.,
                 use_embedding=False,
                 residual=False,
                 conv_interp_type='fourier',
                 mean_constraint=False,
                 diffusion_offset=0.):
        super(FNN2d, self).__init__()

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
        dfs_type = 'lite'
        self.dfs_type = dfs_type
        self.residual = residual
        self.diffusion_offset = diffusion_offset
        dfs_mult = 2 if self.dfs_type=='full' else 1
        if dfs_type == 'full':
            modes1 = [m*2 for m in modes1]
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * 4
        else:
            if conv_type != 'fno':
                layers = [l*3 for l in layers]
            self.layers = layers
        self.fc0 = spectral_norm(nn.Linear(in_dim, self.layers[0], bias=False))
        self.embed_ratio =  self.layers[0]/in_dim
        if conv_type == 'fno':
            self.sp_convs = nn.ModuleList([SpectralConv2dV2(
                in_size, out_size, mode1_num, mode2_num)
                for in_size, out_size, mode1_num, mode2_num
                in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])
        elif conv_type == 'depth_separable':
            sp_convs = []
            for j, (in_size, out_size, mode1_num, mode2_num) in enumerate(zip(self.layers, self.layers[1:], self.modes1, self.modes2)):
                if j != 22:
                    sp_convs.append(DSConvSpectral(
                        in_size, out_size, (dfs_mult*grid_size[0], grid_size[1]), mode1_num, mode2_num,
                        diffusion_offset=self.diffusion_offset))
                else:
                    sp_convs.append(DSConvSpectral2d(
                        in_size, out_size, (dfs_mult*grid_size[0], grid_size[1]), mode1_num, mode2_num,
                        diffusion_offset=self.diffusion_offset))
            self.sp_convs = nn.ModuleList(sp_convs)
        elif conv_type == 'ds_interp':
            self.sp_convs = nn.ModuleList([DSConvReducedSpectral(
                in_size, out_size, (dfs_mult*grid_size[0], grid_size[1]), mode1_num, mode2_num,
                conv_interp_type=conv_interp_type)
                for in_size, out_size, mode1_num, mode2_num
                in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([spectral_norm(nn.Conv2d(in_size, out_size, 1, bias=False))
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])
        self.w2s = nn.ModuleList([spectral_norm(nn.Conv2d(in_size, out_size, 1, bias=False))
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])
        self.gates = nn.ParameterList([nn.Parameter(torch.tensor(0.)) for _ in self.layers])

        self.fc1 = spectral_norm(nn.Linear(layers[-1], out_dim, bias=False))
        #self.fc2 = nn.Linear(fc_dim, out_dim)
        self.activation = _get_act(activation)
        self.mean_constraint = mean_constraint
        self.taper_poles = taper_poles
        self.use_embedding = use_embedding
        if self.taper_poles:
            nlats = grid_size[0]
            delta = 1 / (nlats+1) / 2
            lats = np.sin(np.linspace(delta, np.pi-delta, nlats))
            if self.dfs_type == 'full':
                lats = np.concatenate([lats, lats[::-1]])
                lats = np.fft.rfft(lats)
                lats[5:] = 0
                lats = np.fft.irfft(lats)
            taper = torch.as_tensor(lats).float()[None, None, :, None]
            self.register_buffer('taper', taper)
        if self.use_embedding: # Parameterize this later
            self.cont_pos_embed = ContSphereEmbedding(self.layers[0], (grid_size[0]*dfs_mult, grid_size[1]), (1, 1),
                    steps_per_day=24,
                    samples_per_year=1008,
                    global_dfs=self.dfs_type=='full',
                    sph_order=15,
                    learned_sphere=False)
        if aa_rate <= 1:
            nyquist = grid_size[1]//2 # assumes zonal direction is at least as large
            filter_order = 9
            ew_lanczos = torch.zeros(grid_size[0]*dfs_mult, nyquist+1)
            delta = 1 / grid_size[0] / 2
            ns_reses = np.sin(np.linspace(delta, np.pi-delta, grid_size[0]))
            if self.dfs_type == 'full':
                ns_reses = np.concatenate([ns_reses, ns_reses[::-1]])
            for i, res in enumerate(ns_reses):
                # Node one-d filters don't use aa factor since the twod filter will cut those off. one-d is for shaping only
                cutoff = min(int(1+res*nyquist), nyquist)
                #ew_lanczos[i] = 1/ (1+(torch.fft.rfftfreq(grid_size[1])*grid_size[1] / cutoff)**(2*filter_order))
                ew_lanczos[i, :cutoff] = 1 
            #from scipy.ndimage import gaussian_filter1d
            #ew_lanczos = torch.as_tensor(gaussian_filter1d(ew_lanczos.numpy(), 10, 0, mode='wrap')).float()


            # Two-d filter is scaled by anti-aliasing factor
            ns_filter = ((torch.fft.fftfreq(grid_size[0]*dfs_mult)*grid_size[0]*dfs_mult).unsqueeze(1)**2
                     + (torch.fft.rfftfreq(grid_size[1])*grid_size[1]).unsqueeze(0)**2).sqrt()
            ns_filter = (ns_filter <= nyquist*aa_rate).float()
            #ns_filter = 1 / (1 + (ns_filter/(aa_rate*nyquist))**(2*filter_order))

            ns_filter = (ns_filter[None, None, :, :])

            self.register_buffer('twod_filter', ns_filter, persistent=False)
            self.register_buffer('zonal_filter' , ew_lanczos[None, None, :, :], persistent=False)
        else:
            self.twod_filter = None
            self.zonal_filter = None

    
    def forward(self, x, dt_ind, *args, **kwargs):
        '''
        (b,c,h,w) -> (b,1,h,w)
        '''
        if self.training:
            print([g for g in self.gates])
            twod_filter = None
            zonal_filter = None
        else:
            twod_filter = self.twod_filter
            zonal_filter = self.zonal_filter
        length = len(self.ws)
        batchsize = x.shape[0]
        if self.dfs_type == 'full':
            x = sphere_to_torus(x)
            x[:, 1, x.shape[2]//2:] = -x[:, 1, x.shape[2]//2:]
            
        if self.residual:
            res = x.clone()

        size_x, size_y = x.shape[2], x.shape[3]
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x) * self.embed_ratio**.5# project

        x = x.permute(0, 3, 1, 2)
        #proj = x
        #x = self.lns[-1](x)
        for i, (speconv, w, w2) in enumerate(zip(self.sp_convs, self.ws, self.w2s)):
            gate = torch.sigmoid(self.gates[i])
            if i ==0:
                if self.use_embedding:
                    x1 = (self.cont_pos_embed(dt_ind, ()).permute(0, 3, 1, 2) + gate*x)
            else:
                x1 = x
            x1 = w(x1)
            x1 = self.activation(x1)
            x1 = w2(x1)
            if self.taper_poles:
                x1 = speconv(x1*self.taper, zonal_filter=zonal_filter, twod_filter=twod_filter) / self.taper
            else:
                x1 = speconv(x1, zonal_filter=zonal_filter, twod_filter=twod_filter)
            if i != length - 1:
                x = (1-gate)*x1+gate*x
            else:
                x = (1-gate)*x1+gate*x
        #x = self.last_layer(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x) / self.embed_ratio**.5
        #x = self.activation(x)
        #x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)

        gate = torch.sigmoid(self.gates[-1])
        if self.residual:
            x = (1-gate)*x + gate*res
        # Post filtering 
        if twod_filter is not None:
            dtype = x.dtype
            x = torch.fft.rfft2(x.float(), norm='ortho') * twod_filter
            if zonal_filter is not None:
                x = torch.fft.ifft(x, norm='ortho', dim=-2) * zonal_filter
                x = torch.fft.irfft(x.cdouble(), norm='ortho', dim=-1).to(dtype)
            else:
                x = torch.fft.irfft2(x.cdouble(), norm='ortho', dim=(-2, -1)).to(dtype)


        if self.dfs_type == 'full':
            x_stt = glide_reflection(x[:, :, x.shape[2]//2:])
            x_stt[:, 1] = -x_stt[:, 1]
            x = .5*(x[:,:, :x.shape[2]//2]+x_stt)
        
        if self.mean_constraint:
            x = x - torch.mean(x, dim=(-2,-1), keepdim=True)

        return x

def fno(params):
    if params.mode_cut > 0:
        params.modes1 = [params.mode_cut]*len(params.modes1)
        params.modes2 = [params.mode_cut]*len(params.modes2)

    if params.embed_cut > 0:
        params.layers = [params.embed_cut]*len(params.layers)

    if params.fc_cut > 0 and params.embed_cut > 0:
        params.fc_dim = params.embed_cut * params.fc_cut

    return FNN2d(params.modes1, params.modes2, layers=params.layers, fc_dim=params.fc_dim,
                in_dim=len(params.in_channels), out_dim=len(params.out_channels), activation='gelu',
                dfs_type=params.dfs_type, use_embedding=params.fno_emb, conv_type=params.conv_type,
                aa_rate=params.res_factor, mean_constraint=False, taper_poles=params.taper_poles,
                grid_size=params.img_size, residual=params.fno_residual, conv_interp_type=params.conv_interp_type,
                diffusion_offset=params.diffusion_offset)
