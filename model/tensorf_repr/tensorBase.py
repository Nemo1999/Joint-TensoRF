# This file is modified from TensoRF https://github.com/apchenstu/TensoRF

import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
from torch.profiler import record_function
import numpy as np
import time
from icecream import ic
import sys
# setting path
sys.path.append('../../')
import util

class TVLoss(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):

        batch_size = x.size()[0]
        ic(batch_size)
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        total = 0
        if count_h > 0:
            h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
            h_tv /= count_h
            total += h_tv
        if count_w > 0:
            w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
            w_tv /= count_w
            total += w_tv
        return self.TVLoss_weight*2*(total)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def positional_encoding(positions, freqs, progress=1.0):
        # progress 0: fully mask postitional encoding
        # progress 1: fully expose positional encoding
        levels = torch.arange(freqs, device=positions.device)
        freq_bands = (2**levels)  # (F,)
        mask = (progress * freqs - levels).clamp_(min=0.0, max=1) # linear anealing
        pts = positions[..., None] * freq_bands
        pts_sin = torch.sin(pts)*mask
        pts_cos = torch.cos(pts)*mask
        pts = torch.cat([pts_sin , pts_cos], dim=-1)
        pts = pts.reshape(
            positions.shape[:-1] + (freqs * 2 * positions.shape[-1], ))  # (..., DF)
        return pts

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]

    return alpha, weights, T[:,-1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):

    rgb = features
    return rgb

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):

        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, view_pe_progress=1.0, fea_pe_progress=1.0):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape, fea_pe_progress)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe, view_pe_progress)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender_Fea4(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):

        super(MLPRender_Fea4, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, featureC)
        layer4 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3, torch.nn.ReLU(inplace=True), layer4)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLP_Albedo(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):

        super(MLP_Albedo, self).__init__()

        self.in_mlpC =  2*feape*inChanel  + inChanel
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender_Fea_WeakView(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):

        super(MLPRender_Fea_WeakView, self).__init__()

        self.in_mlpC = (2*feape + 1) * inChanel
        self.mid_mlpC = (2*viewpe ) * 3
        self.viewpe = viewpe
        self.feape = feape
        self.layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        self.layer2 = torch.nn.Linear(featureC, featureC)
        self.layer3 = torch.nn.Linear(featureC + self.mid_mlpC , 3 )

        # create mlp
        #self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        #torch.nn.init.constant_(self.mlp[-1].bias, 0)
        torch.nn.init.constant_(self.layer3.bias, 0)

    def forward(self, pts, viewdirs, features, view_pe_progress=1.0, fea_pe_progress=1.0):
        indata = [features]
        middata = []
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape, fea_pe_progress)]
        if self.viewpe > 0:
            middata += [positional_encoding(viewdirs, self.viewpe, view_pe_progress)]
        mlp_in = torch.cat(indata, dim=-1)
        x = self.layer1(mlp_in)
        x = torch.nn.functional.relu(x, inplace=True)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x, inplace=True)
        middata += [x]
        mlp_mid = torch.cat(middata, dim=-1)
        rgb = self.layer3(mlp_mid)
        rgb = torch.sigmoid(rgb)
        return rgb


class GaRF_Render(torch.nn.Module):
    # source: https://github.com/sfchng/Gaussian-Activated-Radiance-Fields/blob/main/model/nerf_gaussian.py
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(GaRF_Render, self).__init__()
        self.in_mlpC = inChanel
        self.mid_mlpC = 3
        self.viewpe = viewpe
        self.feape = feape
        self.fea_gaussian_linear = torch.nn.Linear(self.in_mlpC, featureC)
        self.view_gaussian_linear = torch.nn.Linear(self.mid_mlpC, featureC)
        self.layer1 = torch.nn.Linear(featureC, featureC)
        self.layer2 = torch.nn.Linear(featureC + featureC , featureC)
        self.layer3 = torch.nn.Linear(featureC  , 3 )
        # uniform init
        self.uniform_init_weights(self.fea_gaussian_linear.weight)
        self.uniform_init_weights(self.view_gaussian_linear.weight)
        self.uniform_init_weights(self.layer1.weight)
        self.uniform_init_weights(self.layer2.weight)
        self.uniform_init_weights(self.layer3.weight)

    def forward(self, pts, viewdirs, features, view_pe_progress=1.0, fea_pe_progress=1.0):
        indata = self.fea_gaussian_init(features)
        middata = [ self.view_gaussian_init(viewdirs) ]
        x = self.layer1(indata)
        x = self.gaussian(x)
        middata.append(x)
        x = torch.cat(middata, dim=-1)
        x = self.layer2(x)
        x = self.gaussian(x)
        x = self.layer3(x)
        rgb = torch.sigmoid(x)
        return rgb

    def fea_gaussian_init(self, x):
        x_ = self.fea_gaussian_linear(x)
        mu = torch.mean(x_, axis = -1).unsqueeze(-1)
        out = (-0.5*(x_-mu)**2/0.1**2).exp() # we use opt.arch.gaussian.sigma=0.1
        return out
    def view_gaussian_init(self,x):
        x_ = self.view_gaussian_linear(x)
        mu = torch.mean(x_, axis = -1).unsqueeze(-1)
        out = (-0.5*(x_-mu)**2/0.1**2).exp() # we use opt.arch.gaussian.sigma=0.1
        return out
    def gaussian(self,x):
        """
        Args:
            opt
            x (torch.Tensor [B,num_rays,])
        """
        out = (-0.5*(x)**2/ 0.1 **2).exp() # we use opt.arch.gaussian.sigma=0.1
        return out
    def uniform_init_weights(self, weight):
        torch.nn.init.uniform_(weight, -0.1 , 0.1) # we use opt.init.weight.range=0.1

class MLPRender_Fea_WeakView_Density(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):

        super(MLPRender_Fea_WeakView_Density, self).__init__()

        self.in_mlpC = (2*feape + 1) * inChanel
        self.mid_mlpC = (2*viewpe ) * 3
        self.viewpe = viewpe
        self.feape = feape
        self.layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        self.layer2 = torch.nn.Linear(featureC, featureC + 1)
        self.layer3 = torch.nn.Linear(featureC + self.mid_mlpC , 3 )

        # create mlp
        #self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        #torch.nn.init.constant_(self.mlp[-1].bias, 0)
        torch.nn.init.constant_(self.layer3.bias, 0)

    def forward(self, pts, viewdirs, features, view_pe_progress=1.0, fea_pe_progress=1.0):
        indata = [features]
        middata = []
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape, fea_pe_progress)]
        if self.viewpe > 0 :
            middata += [positional_encoding(viewdirs, self.viewpe, view_pe_progress)]
        mlp_in = torch.cat(indata, dim=-1)
        x = self.layer1(mlp_in)
        x = torch.nn.functional.relu(x, inplace=True)
        x = self.layer2(x)
        sigma_features, x = x[...,-1], x[...,:-1]
        x = torch.nn.functional.relu(x, inplace=True)
        if viewdirs != None:
            middata += [x]
            mlp_mid = torch.cat(middata, dim=-1)
            rgb = self.layer3(mlp_mid)
            rgb = torch.sigmoid(rgb)
            return  rgb, sigma_features
        else:
            return  None, sigma_features






class MLPRender_PE(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3+2*viewpe*3)+ (3+2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs, pts]
        #ic(viewdirs.shape)
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
            #ic(self.posepe)
            #ic(pts.shape)

        if self.viewpe > 0:
            #ic(self.viewpe)
            #ic(viewdirs.shape)
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3+2*viewpe*3) + inChanel
        self.viewpe = viewpe

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb



class TensorBase(torch.nn.Module):
    def __init__(self, aabb,
                 gridSize,
                 device,
                 density_n_comp = 8,
                 appearance_n_comp = 24,
                 app_dim = 27,
                 shadingMode = 'MLP_PE',
                 alphaMask = None,
                 near_far=[2.0,6.0],
                 density_shift = -10,
                 alphaMask_thres=0.001,
                 distance_scale=25,
                 rayMarch_weight_thres=0.0001,
                 pos_pe = 6,
                 view_pe = 6,
                 fea_pe = 6,
                 featureC=128,
                 step_ratio=2.0,
                 fea2denseAct = 'softplus',
                 dtype=torch.float32,
                 volume_init_scale=0.1,
                 volume_init_bias=0.1
    ):
        super(TensorBase, self).__init__()

        # not part of reset
        self.device=device
        self.dtype = dtype
        self.alphaMask = alphaMask

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]

        self.reset(
            aabb,
            gridSize,
            density_n_comp,
            appearance_n_comp,
            app_dim,
            density_shift,
            alphaMask_thres,
            distance_scale,
            rayMarch_weight_thres,
            fea2denseAct,
            near_far,
            step_ratio,
            shadingMode,
            pos_pe,
            view_pe,
            fea_pe,
            featureC,
            volume_init_scale,
            volume_init_bias)

    def reset(self, aabb, gridSize, density_n_comp, appearance_n_comp, app_dim, density_shift, alphaMask_thres, distance_scale,
              rayMarch_weight_thres, fea2denseAct, near_far, step_ratio, shadingMode, pos_pe, view_pe, fea_pe, featureC, volume_init_scale, volume_init_bias):
        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = torch.tensor(aabb,dtype=self.dtype)
        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct
        self.near_far = near_far
        self.step_ratio = step_ratio
        self.update_stepSize(gridSize)
        self.volume_init_scale = volume_init_scale
        self.volume_init_bias = volume_init_bias
        self.init_svd_volume(gridSize[0], self.device, init_scale=self.volume_init_scale, init_bias=self.volume_init_bias)
        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, self.device)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == 'MLP_PE':
            self.renderModule = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC).to(device)
        elif shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == 'MLP_Fea_WeakView':
            self.renderModule = MLPRender_Fea_WeakView(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == 'MLP':
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == 'MLP_Albedo':
            self.renderModule = MLP_Albedo(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == 'MLP_Fea4':
            self.renderModule = MLPRender_Fea4(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == "MLP_Fea_WeakView_Density":
            self.renderModule = MLPRender_Fea_WeakView_Density(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == 'SH':
            self.renderModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = RGBRender
        elif shadingMode == "GaRF":
            self.renderModule = GaRF_Render(self.app_dim, None , None, featureC).to(device)
        else:
            raise Exception("Unrecognized shading module")
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize).to(self.device)
        self.units=self.aabbSize / (self.gridSize-1)
        self.stepSize=torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass

    def compute_densityfeature(self, xyz_sampled):
        pass

    def compute_appfeature(self, xyz_sampled):
        pass

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    def get_reset_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC,
            'volume_init_scale': self.volume_init_scale,
            'volume_init_bias' : self.volume_init_bias
        }


    # states that need to be saved besides mondel parameters state_dict
    def save_param_state(self):
        reset_kwargs = self.get_reset_kwargs()
        ckpt = {'tensorf_reset_kwargs': reset_kwargs}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        return ckpt

    # states that need to be saved besides mondel parameters state_dict
    def load_param_state(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.to(device=self.device,dtype=self.dtype))
        self.reset(** ckpt["tensorf_reset_kwargs"])

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1, simulate_euclid_sample=False, simulate_euclid_depth=False, ndc_near_plane=1.0):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples, device=self.device, dtype=self.dtype).unsqueeze(0)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)
        ndc_depth = 2 * ndc_near_plane / (1 - ndc_near_plane - interpx)
        simulated_interpx = util.grad_rescale(interpx, ndc_depth)
        if simulate_euclid_sample:
            rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * simulated_interpx[..., None]
        else:
            rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)

        if simulate_euclid_depth:
            return rays_pts, ndc_depth, ~mask_outbbox
        else:
            return rays_pts, interpx, ~mask_outbbox
    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples>0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        rays_dd, rays_od = rays_d.detach(), rays_o.detach()
        vec = torch.where(rays_dd==0, torch.full_like(rays_dd, 1e-6), rays_dd)
        #assert torch.max(torch.abs(vec[2] - 1.0)).item() <  0.001, "z value of ray direction are not 1.0, need normalize to match accurate near/far depth"
        # assertion fails, the z component of rays_d is  1.0917 , not normalized

        if hasattr(self.opt.nerf,"bbox_cycle_xy" ) and self.opt.nerf.bbox_cycle_xy:
            rate_a = (self.aabb[1][2] - rays_od[...,2]) / vec[...,2]
            rate_b = (self.aabb[0][2] - rays_od[...,2]) / vec[...,2]
            t_min = torch.minimum(rate_a, rate_b).clamp(min=near, max=far)

        else:
            rate_a = (self.aabb[1] - rays_od) / vec
            rate_b = (self.aabb[0] - rays_od) / vec
            t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples, dtype=self.dtype, device=self.device)[None]
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * rng
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]

        if hasattr(self.opt.nerf,"bbox_cycle_xy") and self.opt.nerf.bbox_cycle_xy:
            mask_outbbox = ((self.aabb[0][2]>rays_pts[..., 2]) | (rays_pts[...,2]>self.aabb[1][2]))
            W = self.aabb[1][0] - self.aabb[0][0]
            H = self.aabb[1][1] - self.aabb[0][1]
            # wrap around X and Y axis
            rays_pts[...,0][rays_pts[...,0]>self.aabb[1][0]] -= W
            rays_pts[...,0][rays_pts[...,0]<self.aabb[0][0]] += W
            rays_pts[...,1][rays_pts[...,1]>self.aabb[1][1]] -= H
            rays_pts[...,1][rays_pts[...,1]<self.aabb[0][1]] += H
        else:
            mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox


    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 5
        #raise Exception("inspect alpha masks")
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered]


    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features + self.density_shift)


    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)


        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        with torch.no_grad():
            if alpha_mask.any():
                xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
                sigma_feature = self.compute_densityfeature(xyz_sampled)
                if self.component_wise_feature2density:
                    validsigma = sigma_feature
                else:
                    validsigma = self.feature2density(sigma_feature)
                sigma[alpha_mask] = validsigma


        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])

        return alpha


    def forward(self, opt, center, ray_dir, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1, c2f_mode=None, c2f_parameter_density=None, c2f_parameter_color=None, c2f_kernel_size=None, is_test_optim=False, view_pe_progress=1.0, fea_pe_progress=1.0):
        self.opt = opt
        self.abs_components = opt.arch.abs_components
        self.component_wise_feature2density = opt.arch.component_wise_feature2density
        self.plane_feature2density = opt.arch.plane_feature2density
        self.convolve_plane_only = opt.arch.convolve_plane_only
        self.convolve_positive_only = opt.arch.convolve_positive_only
        self.ignore_negative_split = opt.arch.ignore_negative_split
        # sample points
        viewdirs = ray_dir
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(center, viewdirs, is_train=is_train,N_samples=N_samples, simulate_euclid_sample = opt.camera.ndc_simulate_euclid_sample, simulate_euclid_depth = opt.camera.ndc_simulate_euclid_depth, ndc_near_plane=opt.arch.ndc_near_plane if hasattr(opt.arch, "ndc_near_plane") else 1.0)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(center, viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid


        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])
            if self.component_wise_feature2density:
                validsigma = sigma_feature
            else:
                validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma


        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres
        with record_function("compute appearance feature + Rendering"):
            if app_mask.any():
                app_features = self.compute_appfeature(xyz_sampled[app_mask])
                if self.opt.arch.shading.detach_viewdirs:
                    viewdirs = viewdirs.detach()
                if self.opt.arch.shading.detach_xyz:
                    xyz_sampled = xyz_sampled.detach()

                valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features, view_pe_progress, fea_pe_progress)
                rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])


        rgb_map = rgb_map.clamp(0,1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * ray_dir[..., -1]
            if opt.camera.ndc_simulate_euclid_depth:
                depth_map = depth_map.clamp(min=0, max=10)
            else:
                depth_map = depth_map  -  self.near_far[0] + 0.05
        opacity = acc_map
        return rgb_map, depth_map, opacity # rgb, sigma, alpha, weight, bg_weight
