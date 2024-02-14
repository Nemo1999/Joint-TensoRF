from . import planar
from . import planar_svd
from . import base
import warp
from easydict import EasyDict as edict
from util import interp_schedule
import torch
import torch.nn.functional as torch_F
import math
import scipy
import numpy as np
from typing import Tuple
from icecream import ic
import camera 
from warp import lie
import wandb
# training cycle is same as planar


class Model(planar_svd.Model):
    def __init__(self, opt):
        super().__init__(opt)


# ============================ computation graph for forward/backprop ============================


class Graph(planar.Graph):

    def __init__(self, opt):
        super().__init__(opt)
    def compute_loss(self, opt, var, mode=None):
        loss = edict()
        if opt.loss_weight.render is not None:
            image_pert = var.image_pert.view(
                opt.batch_size, 3, opt.H_crop*opt.W_crop).permute(0, 2, 1)
            loss.render = self.MSE_loss(var.rgb_warped, image_pert)
        if opt.loss_weight.total_variance is not None:
            loss.total_variance = self.Parseval_Loss(self.neural_image)
        return loss
    
    def Parseval_Loss(self, svdImage):
        # ParseVal Loss in PREF (similar to TV loss in spatial domain)
        return svdImage.Parseval_Loss()


class NeuralImageFunction(planar_svd.NeuralImageFunction):

    def __init__(self, opt):
        super().__init__(opt)
        # Prebuild Reusable Fourier Basis
        self.basis_h = torch.stack(list(torch.exp(2j*3.141592*f/self.H * torch.arange(0,self.H)) for f in self.freqs_h)).unsqueeze(0).unsqueeze(3).expand(3,-1,-1,self.W) #3, self.max_ranks[0], H, W
        self.basis_w = torch.stack(list(torch.exp(2j*3.141592*f/self.W * torch.arange(0,self.W)) for f in self.freqs_w)).unsqueeze(0).unsqueeze(2).expand(3,-1,self.H,-1) #3, self.max_ranks[1], H, W
        self.basis_h = self.basis_h.to(self.device)
        self.basis_w = self.basis_w.to(self.device)
    
    def initialize_param(self, opt):
        super().initialize_param(opt)
        # arch options
        self.H, self.W = self.resolution[0], self.resolution[1]
        
        # PREF setting for 3D
        """
        self.max_ranks = list(map( lambda x: math.floor(math.log2(x))+1 , self.resolution))
        self.freqs_h = torch.tensor([0] + [2**i for i in range(self.max_ranks[0]-1)])
        self.freqs_w = torch.tensor([0] + [2**i for i in range(self.max_ranks[1]-1)])
        """
        # PREF setting for 2D
        self.max_ranks = self.resolution[0]//5, self.resolution[1]//5
        self.freqs_h = torch.arange(0,self.max_ranks[0])
        self.freqs_w = torch.arange(0,self.max_ranks[1])

    def define_network(self,opt):
        shape1 = (3,self.max_ranks[0], self.resolution[1])
        rank1 = torch.complex(torch.zeros(*shape1), torch.zeros(*shape1))
        
        shape2 = (3, self.max_ranks[1], self.resolution[0])
        rank2 = torch.complex(torch.zeros(*shape2), torch.zeros(*shape2))
        
        self.register_parameter(name='rank1', param=torch.nn.Parameter(rank1))
        self.register_parameter(name='rank2', param=torch.nn.Parameter(rank2))

    def forward(self, opt, coord_2D):  # [B,...,3]
        cur_rank = self.get_scheduled_rank()
        
        cur_rank1 = min(cur_rank, self.max_ranks[0]) 
        cur_rank2 = min(cur_rank, self.max_ranks[1])
        
        rank1_ifft = torch.fft.ifft(self.rank1[:, :cur_rank1,:], dim=2, norm="forward") # 3, cur_rank1, self.resolution[1]
        rank2_ifft = torch.fft.ifft(self.rank2[:, :cur_rank2,:], dim=2, norm="forward") # 3, cur_rank2, self.resolution[0] 

        if self.kernel_type != "none":
            kernel = self.get_kernel()
            complex_kernel = torch.complex(kernel, torch.zeros_like(kernel))

            rank1_ifft = torch_F.conv1d(rank1_ifft[:, :, :], complex_kernel.expand(cur_rank1, 1, -1),
                                 bias=None, stride=1, padding="same", dilation=1, groups=cur_rank1)
            rank2_ifft = torch_F.conv1d(rank2_ifft[:, :, :], complex_kernel.expand(cur_rank1, 1, -1),
                                 bias=None, stride=1, padding="same", dilation=1, groups=cur_rank2)

        img1 = torch.sum(rank1_ifft.unsqueeze(2) * self.basis_h[:,:cur_rank1, ...], dim=1, keepdim=False)
        img2 = torch.sum(rank2_ifft.unsqueeze(3) * self.basis_w[:,:cur_rank2, ...], dim=1, keepdim=False)
       
        rbg = torch.real(img1) + torch.real(img2)
        assert rbg.shape == (
            3, self.resolution[0], self.resolution[1]), f"rbg image has shape {rbg.shape}"
        B = coord_2D.shape[0]
        #ic(coord_2D)
        # coord_2D += 0.5
        # coord_2D[:,:,0] *= self.resolution[0]
        # coord_2D[:,:,1] *= self.resolution[1]

        sampled_rbg = torch_F.grid_sample(rbg.expand(B, -1, -1, -1), coord_2D.unsqueeze(1), align_corners=False).squeeze(2)
        #ic(sampled_rbg)
        #ic(coord_2D[0][0])
        #ic(sampled_rbg.shape)
        return sampled_rbg.permute(0, 2, 1)

    def Parseval_Loss(self):
        r1_v = torch.arange(0,self.W)[None,None,...].to(self.device) * self.rank1
        r1_u = self.freqs_h[None, ... , None].to(self.device) * self.rank1

        r2_v = torch.arange(0,self.H)[None,None,...].to(self.device) * self.rank2
        r2_u = self.freqs_w[None, ... , None].to(self.device) * self.rank2

        return sum(torch.linalg.norm(r) for r in [r1_v, r1_u, r2_v, r2_u])