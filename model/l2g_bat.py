import pdb
import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
import visdom
import matplotlib.pyplot as pl
from icecream import ic
import util,util_vis
from util import log,debug
from . import bat
import camera
import wandb
import roma
from . import tensorf_repr
from util import interp_schedule


class Model(bat.Model):
    def __init__(self,opt):
        super().__init__(opt)

    def build_networks(self, opt):
        super().build_networks(opt)
        self.graph.warp_embedding = torch.nn.Embedding(len(self.train_data), opt.arch.embedding_dim).to(opt.device)
        self.graph.warp_mlp = localWarp(opt).to(opt.device)

        pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device)
        # add synthetic pose perturbation to all training data
        if opt.data.dataset=="blender":
            pose = pose_GT
            if opt.camera.noise:
                pose = camera.pose.compose([pose, self.graph.pose_noise])
        else: pose = self.graph.pose_eye[None].repeat(len(self.train_data),1,1)
        # use Embedding so it could be checkpointed
        self.graph.optimised_training_poses = torch.nn.Embedding(len(self.train_data),12,_weight=pose.view(-1,12)).to(opt.device)

        if opt.error_map_size:
            self.graph.error_map = torch.ones([len(self.train_data), opt.error_map_size*opt.error_map_size], dtype=torch.float).to(opt.device)

    def setup_optimizer(self, opt):
        super().setup_optimizer(opt)
        # add warp MLP and warp embedding into pose_optim param_group
        if hasattr(opt.optim, "pose_grad_accum_iter"):
            accum_scale = 1.0
            #accum_scale = 1 / opt.optim.pose_grad_accum_iter
        else:
            accum_scale = 1.0

        if opt.optim.sched_pose.type == "ExponentialLR":
            lr_init = opt.optim.lr_pose * accum_scale
        else:
            # LambdaLR
            lr_init = 1.0 * accum_scale #opt.optim.sched_pose.lr_pose_schedule[0]
        self.optim_pose.add_param_group(dict(params=self.graph.warp_embedding.parameters(),lr=lr_init))
        self.optim_pose.add_param_group(dict(params=self.graph.warp_mlp.parameters(),lr=lr_init))

    @torch.no_grad()
    def get_all_training_poses(self,opt):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device)
        pose = self.graph.optimised_training_poses.weight.data.detach().clone().view(-1,3,4)
        return pose,pose_GT



class Graph(bat.Graph):
    def __init__(self,opt):
        super().__init__(opt)
        self.tvloss = tensorf_repr.TVLoss()
    def get_pose(self,opt,var,mode=None):
        if mode=="train":
            # add the pre-generated pose perturbations
            if opt.data.dataset=="blender":
                if opt.camera.noise:
                    var.pose_noise = self.pose_noise[var.idx]
                    pose = camera.pose.compose([var.pose, var.pose_noise])
                else: pose = var.pose
            else: pose = self.pose_eye[None]
            ic(pose.shape)
            ic(pose.isnan().any())
            # add learnable pose correction
            batch_size = len(var.idx)
            if opt.error_map_size:
                num_points = var.ray_idx.shape[1]
                camera_cords_grid_3D = camera.gather_camera_cords_grid_3D(opt,batch_size,intr_inv=var.intr_inv,ray_idx=var.ray_idx).detach()
            else:
                num_points = len(var.ray_idx)
                camera_cords_grid_3D = camera.get_camera_cords_grid_3D(opt,batch_size,intr_inv=var.intr_inv,ray_idx=var.ray_idx).detach()
            ic(camera_cords_grid_3D.shape)
            ic(camera_cords_grid_3D.mean())
            ic(camera_cords_grid_3D.abs().mean())
            ic(camera_cords_grid_3D.isnan().any())

            camera_cords_grid_2D = camera_cords_grid_3D[...,:2]
            embedding = self.warp_embedding.weight[var.idx,None,:].expand(-1,num_points,-1)
            ic(embedding.shape)
            ic(embedding.mean())
            ic(embedding.abs().mean())
            ic(embedding.isnan().any())
            local_se3_refine = self.warp_mlp(opt,torch.cat((camera_cords_grid_2D,embedding),dim=-1))
            ic(local_se3_refine.shape)
            ic(local_se3_refine.isnan().any())
            local_pose_refine = camera.lie.se3_to_SE3(local_se3_refine)
            ic(local_pose_refine.shape)
            ic(local_pose_refine.isnan().any())
            local_pose = camera.pose.compose([local_pose_refine, pose[:,None,...]])
            ic(local_pose.shape)
            ic(local_se3_refine.isnan().any())
            return local_pose

        elif mode in ["val","eval","test-optim"]:
            # align test pose to refined coordinate system (up to sim3)
            sim3 = self.sim3
            center = torch.zeros(1,1,3,device=opt.device)
            center = camera.cam2world(center,var.pose)[:,0] # [N,3]
            center_aligned = (center-sim3.t0)/sim3.s0@sim3.R*sim3.s1+sim3.t1
            R_aligned = var.pose[...,:3]@self.sim3.R
            t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
            pose = camera.pose(R=R_aligned,t=t_aligned)
            if opt.optim.test_photo and mode!="val":
                pose = camera.pose.compose([var.pose_refine_test, pose])

        else: pose = var.pose
        return pose

    def forward(self,opt,var,mode=None):
        # rescale the size of the scene condition on the pose
        if opt.data.dataset=="blender":
            depth_min,depth_max = opt.nerf.depth.range
            position = camera.Pose().invert(self.optimised_training_poses.weight.data.detach().clone().view(-1,3,4))[...,-1]
            diameter = ((position[self.idx_grid[...,0]]-position[self.idx_grid[...,1]]).norm(dim=-1)).max()
            depth_min_new = (depth_min/(depth_max+depth_min))*diameter
            depth_max_new = (depth_max/(depth_max+depth_min))*diameter
            opt.nerf.depth.range = [depth_min_new, depth_max_new]

        # render images
        batch_size = len(var.idx)
        if mode in ["train"]:
            # sample rays for optimization
            if opt.error_map_size:
                sample_weight = self.error_map + 2*self.error_map.mean(-1,keepdim=True) # 1/3 importance + 2/3 random
                var.ray_idx_coarse = torch.multinomial(sample_weight, opt.nerf.n_rays//batch_size, replacement=False) # [B, N], but in [0, opt.error_map_size*opt.error_map_size)
                inds_x, inds_y = var.ray_idx_coarse // opt.error_map_size, var.ray_idx_coarse % opt.error_map_size # `//` will throw a warning in torch 1.10... anyway.
                sx, sy = opt.H / opt.error_map_size, opt.W / opt.error_map_size
                inds_x = (inds_x * sx + torch.rand(batch_size, opt.nerf.n_rays//batch_size, device=opt.device) * sx).long().clamp(max=opt.H - 1)
                inds_y = (inds_y * sy + torch.rand(batch_size, opt.nerf.n_rays//batch_size, device=opt.device) * sy).long().clamp(max=opt.W - 1)
                var.ray_idx = inds_x * opt.W + inds_y
            else:
                var.ray_idx = torch.randperm(opt.H*opt.W,device=opt.device)[:opt.nerf.n_rays//batch_size]# 3/3 random

            local_pose = self.get_pose(opt,var,mode=mode)
            ret = self.local_render(opt,local_pose,intr=var.intr,ray_idx=var.ray_idx,mode=mode, intr_inv=var.intr_inv) # [B,N,3],[B,N,1]
        elif  mode in ["test-optim"]:
            # sample random rays for optimization
            var.ray_idx = torch.randperm(opt.H*opt.W,device=opt.device)[:opt.nerf.n_rays//batch_size]
            pose = self.get_pose(opt,var,mode=mode)
            ret = self.render(opt,pose,intr_inv=var.intr_inv, intr=var.intr,ray_idx=var.ray_idx,mode=mode) # [B,N,3],[B,N,1]
        else:
            # render full image (process in slices)
            pose = self.get_pose(opt,var,mode=mode)
            ret = self.render_by_slices(opt,pose,intr_inv=var.intr_inv, intr=var.intr,mode=mode)
        var.update(ret)
        return var

    def compute_loss(self,opt,var,mode=None):
        loss = edict()
        batch_size = len(var.idx)
        image = var.image.view(batch_size,3,opt.H*opt.W).permute(0,2,1)

        if mode in ["train","test-optim"]:
            if opt.error_map_size:
                image = torch.gather(image, 1, var.ray_idx[...,None].expand(-1,-1,3))
            else:
                image = image[:,var.ray_idx]

        # compute image losses
        if opt.loss_weight.render is not None:
            loss.render = self.MSE_loss(var.rgb,image)

        if mode in ["train"] and opt.error_map_size:
                ema_error = 0.1 * torch.gather(self.error_map, 1, var.ray_idx_coarse) + 0.9 * render_error.detach()
                self.error_map.scatter_(1, var.ray_idx_coarse, ema_error)

        # calculate TV_Loss / Ortho_Loss / L1_Loss
        loss.L1 = self.nerf.tensorf.density_L1()
        ic(loss.L1)
        loss.TV_density = self.nerf.tensorf.TV_loss_density(self.tvloss)
        loss.TV_color = self.nerf.tensorf.TV_loss_app(self.tvloss)

        # global alignment
        if mode in ["train"]:
            source = torch.cat((var.camera_grid_3D,var.camera_center),dim=1)
            target = torch.cat((var.grid_3D,var.center),dim=1)
            R_global, t_global = roma.rigid_points_registration(target, source)
            svd_poses = torch.cat((R_global,t_global[...,None]),-1q)
            self.optimised_training_poses.weight.data = svd_poses.detach().clone().view(-1,12)
            if opt.loss_weight.global_alignment is not None:
                loss.global_alignment = self.MSE_loss(target,camera.cam2world(source,svd_poses))

        return loss

    def local_render(self,opt,local_pose,intr_inv=None,ray_idx=None,mode=None,intr=None):
        ic(torch.any(torch.isnan(local_pose)))
        ic(torch.any(torch.isnan(ray_idx)))
        batch_size = len(local_pose)
        if opt.error_map_size:
            camera_grid_3D = camera.gather_camera_cords_grid_3D(opt,batch_size,intr_inv=intr_inv,ray_idx=ray_idx).detach()
        else:
            camera_grid_3D = camera.get_camera_cords_grid_3D(opt,batch_size,intr_inv=intr_inv,ray_idx=ray_idx).detach()

        camera_center = torch.zeros_like(camera_grid_3D) # [B,HW,3]
        grid_3D = camera.cam2world(camera_grid_3D[...,None,:],local_pose)[...,0,:] # [B,HW,3]
        center = camera.cam2world(camera_center[...,None,:],local_pose)[...,0,:] # [B,HW,3]
        ray = grid_3D-center # [B,HW,3]
        ret = edict(camera_grid_3D=camera_grid_3D, camera_center=camera_center, grid_3D=grid_3D, center=center) # [B,HW,K] use for global alignment
        ic(torch.any(torch.isnan(ray)))
        ic(torch.any(torch.isnan(center)))
        ic(center.shape)
        ic(ray.shape)
        if opt.camera.ndc:
            # convert center/ray representations to NDC
            center,ray = camera.convert_NDC(opt,center,ray,intr=intr)
        render_ret =  self.render_rays(opt, center, ray, mode=mode, n_views=batch_size, n_pixels_per_view=center.shape[1])
        ret.update(render_ret)
        return ret
class NeRF(bat.NeRF):
        def __init__(self,opt):
            super().__init__(opt)


class localWarp(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        # point-wise se3 prediction
        input_2D_dim = 2
        self.mlp_warp = torch.nn.ModuleList()
        L = util.get_layer_dims(opt.arch.layers_warp)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = input_2D_dim+opt.arch.embedding_dim
            if li in opt.arch.skip_warp: k_in += input_2D_dim+opt.arch.embedding_dim
            linear = torch.nn.Linear(k_in,k_out)
            self.mlp_warp.append(linear)

    def forward(self,opt,uvf):
        feat = uvf
        for li,layer in enumerate(self.mlp_warp):
            if li in opt.arch.skip_warp: feat = torch.cat([feat,uvf],dim=-1)
            feat = layer(feat)
            if li!=len(self.mlp_warp)-1:
                feat = torch_F.relu(feat)
        warp = feat
        return warp
