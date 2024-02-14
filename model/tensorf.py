from . import nerf
from . import barf
import wandb
from util import log, interp_schedule
import util
import torch
import os, sys, time
import tqdm
import wandb
import camera
from easydict import EasyDict as edict
import numpy as np
import torch
from icecream import ic
from typing import List
from . import tensorf_repr
import pdb

class Model(nerf.Model):
    def __init__(self, opt):
        super().__init__(opt)

    def setup_optimizer(self, opt):
        self.optim = self.graph.nerf._get_optimizer(opt)
        # setup hooks for nerf to update optimizer params
        self.graph.nerf.get_current_optimizer = lambda : self.optim
        def register_new_optimizer(optimizer):
            self.optim = optimizer
        self.graph.nerf.register_new_optimizer = register_new_optimizer

    def summarize_loss(self, opt, var, loss):
        loss_all = 0.
        assert("all" not in loss)
        # weigh losses
        for key in loss:
            assert(key in opt.loss_weight), f"loss {key} not in opt.loss_weight"
            assert(loss[key].shape==()), f"loss \"{key}\" has shape {loss[key].shape}"
            if key == "L1":
                first_alpha_update = opt.train_schedule.update_alphamask_iters[0]
                weight = float(opt.loss_weight.L1.rest if hasattr(self,"it") and self.it > first_alpha_update else opt.loss_weight.L1.init)
                loss_all += weight * loss["L1"]
            elif opt.loss_weight[key] is not None:
                assert not torch.isinf(loss[key]),"loss {} is Inf".format(key)
                assert not torch.isnan(loss[key]),"loss {} is NaN".format(key)
                loss_all += float(opt.loss_weight[key])*loss[key]
        loss.update(all=loss_all)
        return loss

    def log_scalars(self, opt, var, loss, metric=None, step=0, split="train"):
        # call add_scalars of base.Model
        super(nerf.Model, self).log_scalars(opt, var, loss, metric, step, split)
        # log lr_basis and lr_index
        lr_basis = self.graph.nerf.lr_basis
        lr_index = self.graph.nerf.lr_index
        self.tb.add_scalar("{0}/{1}".format(split,"lr_basis"),lr_basis,step)
        wandb.log({f"{split}.{'lr_basis'}": lr_basis}, step=step)
        self.tb.add_scalar("{0}/{1}".format(split,"lr_index"),lr_index,step)
        wandb.log({f"{split}.{'lr_index'}": lr_index}, step=step)

        # log psnr
        psnr = -10*loss.render.log10()
        self.tb.add_scalar("{0}/{1}".format(split,"PSNR"),psnr,step)
        wandb.log({f"{split}.{'PSNR'}": psnr}, step=step)

        # log diff kernel parameter
        if opt.model == "bat" and opt.c2f_mode in ["uniform-gaussian","uniform-average"]:
            if opt.c2f_mode in ["uniform-gaussian","uniform-average"]:
                color_p = interp_schedule(self.graph.nerf.progress, opt.c2f_schedule_color)
                density_p = interp_schedule(self.graph.nerf.progress, opt.c2f_schedule_density)
            wandb.log({f"{split}.color_diff_param_mean": color_p}, step=step)
            wandb.log({f"{split}.density_diff_param_mean": density_p}, step=step)
            if hasattr(opt, "blur_2d") and opt.blur_2d == True:
                diff_2d_param = interp_schedule(self.graph.nerf.progress, opt.blur_2d_c2f_schedule)
                wandb.log({f"{split}.2D_diff_param": diff_2d_param}, step=step)

    # states that need to be saved besides mondel parameters state_dict
    def save_param_state(self):
        ckpt = self.graph.save_param_state()
        return ckpt
    # states that need to be saved besides mondel parameters state_dict
    def load_param_state(self, opt, ckpt):
        # reset self
        self.graph.load_param_state(opt, ckpt)


class Graph(nerf.Graph):
    def __init__(self, opt):
        super().__init__(opt)
        self.tvloss = tensorf_repr.TVLoss()

    def forward(self, opt, var, mode=None):

        var = super().forward(opt, var, mode)
        return var

    def compute_loss(self,opt,var,mode=None):
        loss = edict()
        batch_size = len(var.idx)
        image = var.image.view(batch_size,3,opt.H*opt.W).permute(0,2,1)
        if mode in ["train","test-optim"]:
            image = image[:,var.ray_idx]
        # compute render losses
        if opt.loss_weight.render is not None:
            if hasattr(opt, "edge_mask_on_render_loss") and opt.edge_mask_on_render_loss:
                if hasattr(opt, "alternate_edge_loss") and opt.alternate_edge_loss:
                    edge_loss_on = self.it % 2 == 0
                else:
                    edge_loss_on = True
            else:
                edge_loss_on = False

            if edge_loss_on and  mode in ["train"] and self.it < opt.edge_mask_before_iter:
                edge_mask = var.train_edge_masks[:, var.ray_idx].view(batch_size, len(var.ray_idx), 1)
                if hasattr(opt, "soft_edge_loss") and opt.soft_edge_loss:
                    edge_mask = edge_mask.expand(-1, -1, 3) * opt.edge_loss_factor + opt.non_edge_loss_factor
                    loss.render = self.MSE_loss(var.rgb*edge_mask, image*edge_mask)

                else:
                    edge_mask = edge_mask.expand(-1, -1, 3)
                    edge_loss = self.MSE_loss(var.rgb*edge_mask, image*edge_mask)
                    non_edge_loss = self.MSE_loss(var.rgb * (1-edge_mask), image * (1-edge_mask))
                    loss.render = opt.edge_loss_factor*edge_loss + opt.non_edge_loss_factor * non_edge_loss
            else:
                loss.render = self.MSE_loss(var.rgb,image)

        # calculate TV_Loss / Ortho_Loss / L1_Loss
        loss.L1 = self.nerf.tensorf.density_L1()
        ic(loss.L1)
        loss.TV_density = self.nerf.tensorf.TV_loss_density(self.tvloss)
        loss.TV_color = self.nerf.tensorf.TV_loss_app(self.tvloss)


        if mode =="train" and opt.nerf.ray_sampling_strategy == "all_view_rand_grid":
            if "TV_depth" in opt.loss_weight:
                depth_map = var.depth.reshape(batch_size, var.grid_H, var.grid_W)
                h_tv = torch.pow(depth_map[:,1:,:]-depth_map[:,:-1,:],2).sum() / var.grid_H
                w_tv = torch.pow(depth_map[:,:,1:]-depth_map[:,:,:-1],2).sum() / var.grid_W
                loss.TV_depth = h_tv + w_tv
                if self.it > opt.loss_weight.TV_depth_until_iters:
                    opt.loss_weight.TV_depth = 0.0

        return loss

    def render(self,opt,pose,intr_inv=None,ray_idx=None,mode=None, intr=None):

        batch_size = len(pose) # B = n_views
        if pose.isnan().any():
            print("Error: some pose is nan")
            print("Error on SE3 conversion!!")
            print(f"Pose = {pose}")
            raise Exception("Get NaN in pose")


        center,ray = camera.get_center_and_ray(opt,pose,intr_inv=intr_inv) # [B,HW,3]
        while ray.isnan().any(): # TODO: weird bug, ray becomes NaN arbitrarily if batch_size>1, not deterministic reproducible
            center,ray = camera.get_center_and_ray(opt,pose,intr_inv=intr_inv) # [B,HW,3]
            print("stuck in tensorf.py line 136, possibly because taylor series calculation out of bounds")

        if ray_idx is not None:
            # consider only subset of rays
            center,ray = center[:,ray_idx],ray[:,ray_idx]
        if opt.camera.ndc:
            # convert center/ray representations to NDC
            assert intr != None
            center,ray = camera.convert_NDC(opt,center,ray,intr=intr)

        return self.render_rays(opt, center, ray, mode, n_views=len(pose), n_pixels_per_view=center.shape[1])

    def render_rays(self,opt, center, ray, mode=None, n_views=None, n_pixels_per_view=None):

        batch_size = n_views if n_views else center.shape[0]# B = n_views
        dim1 = n_pixels_per_view if n_pixels_per_view else center.shape[1] # this can actually be H*W or len(ray_idx)
        #---------above are same as nerf.Graph.render ---------------------------
        # parameters used on barf only
        if opt.model == "bat" and opt.c2f_mode != "None":
                c2f_mode = opt.c2f_mode #mode for applying kernel on tensorf ( VM CP architecture should implement each of the c2f mode)
                c2f_kernel_size = opt.c2f_kernel_size
                if c2f_mode in ["uniform-gaussian","uniform-average"]:
                    c2f_parameter_density = interp_schedule(self.nerf.progress, opt.c2f_schedule_density)
                    c2f_parameter_color = interp_schedule(self.nerf.progress, opt.c2f_schedule_color)
                else:
                        raise Exception("unknown c2f_mode")
        else:
            c2f_parameter_density = None
            c2f_parameter_color = None
            c2f_mode = None
            c2f_kernel_size = None

        if mode == "vis":
            # don't distube density blur on visualization
            pass
        # randomly sample scale factor
        elif opt.model == "bat" and hasattr(opt, "c2f_random_density_blur") and opt.c2f_random_density_blur:
            if opt.c2f_mode != "None" and opt.c2f_mode != "diff":
                if mode == "train" and hasattr(opt, "sync_2d_3d_scales") and opt.sync_2d_3d_scales:
                    scale = self.scale
                else:
                    scale = np.random.choice(opt.c2f_random_density_scale_pool)
                c2f_parameter_density *= scale
        else:
            pass

        if mode == "test-optim" and opt.model == "bat" and opt.data.dataset == "llff":
            c2f_parameter_density = interp_schedule(self.nerf.test_time_progress, opt.optim.test_kernel_schedule)

        # remove kernel if max of all parameter less than threshold
        # this increase PSNR a little bit
        eps = 0.001
        if c2f_mode in ["uniform-gaussian","uniform-average"]:
            maxes_d = [c2f_parameter_density]
            maxes_c = [c2f_parameter_color]
        else:
            maxes_d = [100]
            maxes_c = [100]

        if max(maxes_d + maxes_c) < eps:
                c2f_parameter_density = None
                c2f_parameter_color = None
                c2f_mode = None
                c2f_kernel_size = None

        # convert camera rays to float16 for speedup and memory effeciency
        # not effective on RTX3090 and RTX2080Ti
        if hasattr(opt, "half_tensor") and opt.half_tensor == True:
            center = center.to(torch.float16)
            ray = ray.to(torch.float16)


        # near plane schedule
        if opt.data.dataset != "blender" and opt.model == "bat":
            self.nerf.tensorf.near_far[0] = interp_schedule(self.nerf.progress, opt.tensorf_near_plane_schedule)
            opt.nerf.depth.range[0] = self.nerf.tensorf.near_far[0]
            # slowly extend near plane to the front (from 0.4 to -1.0 ) to handle extremely close objects in LLFF dataset

        # Positional Encoding schedule
        if hasattr(opt, "c2f_view_pe_schedule"):
            view_pe_progress = interp_schedule(self.nerf.progress, opt.c2f_view_pe_schedule)
        else:
            view_pe_progress = 1.0
        if hasattr(opt, "c2f_fea_pe_schedule"):
            fea_pe_progress = interp_schedule(self.nerf.progress, opt.c2f_fea_pe_schedule)
        else:
            fea_pe_progress = 1.0

        # render with tensorf
        tensorf_output =self.nerf.tensorf.forward(
            opt
            ,center=center.view(-1,3) # ray centers [B*H*W, 3]
            ,ray_dir=ray.view(-1,3) # ray directions [B*H*W, 3]
            ,white_bg=opt.nerf.setbg_opaque
            ,is_train=mode == "train" and opt.nerf.sample_stratified # in training mode, tensorf depth sampling is stratified
            ,is_test_optim=(mode=="test-optim") and (opt.data.dataset=="llff")
            ,ndc_ray=opt.camera.ndc
            ,N_samples=self.nerf.n_samples # this opt value is updated along training schedule by nerf._update_num_samples
            ,c2f_parameter_density=c2f_parameter_density #bat only
            ,c2f_parameter_color=c2f_parameter_color #bat only
            ,c2f_mode=c2f_mode #bat only
            ,c2f_kernel_size=c2f_kernel_size
            ,fea_pe_progress=fea_pe_progress
            ,view_pe_progress=view_pe_progress
        )
        rgb, depth, opacity = tensorf_output
        rgb = rgb.view(batch_size, dim1,3)
        depth = depth.view(batch_size, dim1, 1)
        opacity = opacity.view(batch_size, dim1, 1)
        ret = edict(rgb=rgb,depth=depth,opacity=opacity) # [B,HW,K]
        return ret

    # states that need to be saved besides mondel parameters state_dict
    def save_param_state(self):
        ckpt = self.nerf.save_param_state()
        return ckpt
    # states that need to be saved besides mondel parameters state_dict
    def load_param_state(self, opt, ckpt):
        # reset self
        self.nerf.load_param_state(opt, ckpt)

class NeRF(nerf.NeRF):
    def __init__(self, opt):
        self.device = opt.device
        # callback for updating optimizer used by up stream tensorf.Model
        self.register_new_optimizer = None
        self.get_current_optimizer = None
        # lr_decay_factor
        self.lr_decay_duration = opt.max_iter if opt.optim.lr_decay_iters < 0 else opt.optim.lr_decay_iters
        ic(self.lr_decay_duration)
        self.lr_decay_factor = opt.optim.lr_decay_target_ratio ** (1 / self.lr_decay_duration)
        ic(self.lr_decay_factor)
        # iterations for update alpha mask
        self.update_alphamask_iters = opt.train_schedule.update_alphamask_iters
        ic(self.update_alphamask_iters)
        # iterations for doing upsample
        self.upsample_list = opt.train_schedule.upsample_iters
        ic(self.upsample_list)

        #preset bbox to make _find_resolution works
        bbox = opt.data.scene_bbox
        self.bbox = torch.tensor(bbox).to(torch.float).to(self.device).view(2,3)


        self.reset(
            opt
            ,bbox=opt.data.scene_bbox # [xyz_min, xyz_max]
            ,n_voxel_list=(torch.round(torch.exp(torch.linspace(np.log(opt.train_schedule.n_voxel_init), np.log(opt.train_schedule.n_voxel_final), len(self.upsample_list)+1))).long()).tolist()[1:]
            ,n_voxels=opt.train_schedule.n_voxel_init
            ,alphamask_resolution=self._find_resolution(opt,opt.train_schedule.n_voxel_init) # initialize to be same as self.resolution
            ,lr_basis=opt.optim.lr_basis
            ,lr_index=opt.optim.lr_index
            ,TV_weight_color=opt.loss_weight.TV_color
            ,TV_weight_density=opt.loss_weight.TV_density
        )

        super(NeRF,self).__init__(opt)

    def reset( self,
               opt,
               bbox,
               n_voxel_list,
               n_voxels,
               alphamask_resolution,
               lr_basis, lr_index,
               TV_weight_color, TV_weight_density,
        ):

        self.device = opt.device
        # bbox may shrink latter on alphamask update
        self.bbox = torch.tensor(bbox).to(torch.float).to(self.device).view(2,3)
        ic(self.bbox)

        # upsampling voxels schedule (n_voxel is linear in logrithmic space)
        self.n_voxel_list = n_voxel_list
        ic(self.n_voxel_list)
        # current n_voxels might changes during upsample schedule
        self.n_voxels = n_voxels
        ic(self.n_voxels)
        # find resolution and n_samples from n_voxels
        self.resolution = self._find_resolution(opt, self.n_voxels)
        ic(self.resolution)
        self.n_samples = self._find_n_samples(opt, self.resolution)
        ic(self.n_samples)

        # alpha mask update, may be update on alphamaks update (if self.resolution changes after last update)
        self.alphamask_resolution = alphamask_resolution


        # simulate lr for logging purpose only , will scaled by lr_factor and maybe reset on upsample during training
        self.lr_basis = lr_basis
        self.lr_index = lr_index

        # simulate TV weight for color and density, will scaled by lr_factor during training
        self.TV_weight_color = TV_weight_color
        self.TV_weight_density = TV_weight_density

        # change opt.loss_weight.TV_density and opt.loss_weight.TV_color
        # because Model.summarize_loss use opt.loss_weight
        opt.loss_weight.TV_color = self.TV_weight_color
        opt.loss_weight.TV_density = self.TV_weight_density


    def define_network(self, opt):
        # tensorf module
        if hasattr(opt, "half_tensor") and opt.half_tensor == True:
            tensorf_dtype = torch.float16
        else:
            tensorf_dtype = torch.float32

        # convert density components to integer
        density_n_comp = list(map(int,opt.arch.tensorf.density_components))

        appearance_n_comp = list(map(int,opt.arch.tensorf.color_components))
        assert opt.arch.tensorf.model in ["BAT_VMSplit", "TensorVMSplit"] or (max(density_n_comp) == min(density_n_comp) and max(appearance_n_comp) == min(appearance_n_comp)), "CP split components must be same in all dimensions"
        opt.arch.tensorf.density_components = density_n_comp
        opt.arch.tensorf.color_components = appearance_n_comp

        self.tensorf = getattr(tensorf_repr, opt.arch.tensorf.model)(
            self.bbox
            ,self.resolution
            ,opt.device
            ,density_n_comp=density_n_comp # number of components in tensor decomposition
            ,appearance_n_comp=appearance_n_comp # number of components in tensor decomposition
            ,app_dim=3 if opt.arch.shading=="RGB" else opt.arch.shading.app_dim  # input feature dimension to Renderer , 27 is requried for SH (spherical harmonic) , we also use 27 for MLP Renderer for convinience
            ,near_far = opt.nerf.depth.range #near and far range of each camera
            ,shadingMode= opt.arch.shading.model # can be "SH" or "MLP_PE" or "MLP_Fea" or "MLP" or "RGB"
            ,alphaMask_thres=opt.train_schedule.alpha_mask_threshold # threshold for updating alpha mask volume
            ,density_shift=opt.arch.density_shift # shift density in softplus; making density = 0  when feature == 0
            ,distance_scale=opt.arch.distance_scale # scale up distance for sigma to alpha computation
            ,pos_pe=opt.arch.shading.pose_pe # positional encoding for position (MLP shader)
            ,view_pe=opt.arch.shading.view_pe # positional encoding for view direction (MLP shader)
            ,fea_pe=opt.arch.shading.fea_pe # positional encoding for input feature (MLP shader)
            ,featureC=opt.arch.shading.mlp_hidden_dim # hidden dimension for MLP shader
            ,step_ratio=opt.nerf.step_ratio # ratio between resolution and sampling_step_size # this ratio will compute and estimate sampling interval using current resolution,  overwrite nerf.sample_intvs if smaller
            ,fea2denseAct=opt.arch.feature_to_density_activation # activation used to convert raw tensor density value to real densoity
            ,dtype=tensorf_dtype
            ,volume_init_scale = opt.arch.tensorf.volume_init_scale
            ,rayMarch_weight_thres = opt.arch.tensorf.rayMarch_weight_thres
            ,volume_init_bias = opt.arch.tensorf.volume_init_bias
        )

    def update_schedule(self, opt, it):
        assert self.register_new_optimizer is not None
        assert self.get_current_optimizer is not None

        if it in self.upsample_list:
            if it  == self.upsample_list[0]:
                # stop modifying resolution scale
                opt.train_schedule.resolution_scale_init = [1.0, 1.0, 1.0]
                if hasattr(opt.train_schedule,"reset_pose_on_first_upsample") and opt.train_schedule.reset_pose_on_first_upsample:
                    with torch.no_grad():
                        # get graph
                        self.get_parent().se3_refine.weight *= 0.0
                        print("reset pose se3 weights !!!!!!!!!!!!!!!!!!!")
            #  upsample voxels  --> update resolution --> upsample tensorf --> update_num_samples --> get_new_optimizer --> register new optmizer
            self.n_voxels = self.n_voxel_list.pop(0)
            self.resolution = self._find_resolution(opt,self.n_voxels)
            self.tensorf.upsample_volume_grid(self.resolution)
            if hasattr(opt.train_schedule,"reset_on_last_upsample") \
               and it == self.upsample_list[-1]:
                self.tensorf.init_svd_volume(
                    self.resolution,
                    opt.device,
                    init_density=opt.train_schedule.reset_on_last_upsample.density,
                    init_app=opt.train_schedule.reset_on_last_upsample.appearance,
                    init_basis=opt.train_schedule.reset_on_last_upsample.basis,
                    init_scale=opt.arch.tensorf.volume_init_scale
                )

            self.n_samples = self._find_n_samples(opt, self.resolution)
            optimizer = self._get_optimizer(opt,it)
            self.register_new_optimizer(optimizer)
        else:
            # update lr by lr_factor
            optimizer = self.get_current_optimizer()
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self.lr_decay_factor
            self.lr_basis *= self.lr_decay_factor
            self.lr_index *= self.lr_decay_factor

        if it in self.update_alphamask_iters:
            self._update_alphamask(it)

        if opt.loss_weight.TV_density > 0:
            opt.loss_weight.TV_density *= self.lr_decay_factor
            self.TV_weight_density = opt.loss_weight.TV_density

        if opt.loss_weight.TV_color > 0:
            opt.loss_weight.TV_color *= self.lr_decay_factor
            self.TV_weight_color = opt.loss_weight.TV_color

    def _find_resolution(self, opt, n_voxels: int):
        # find current resolution given total number of voxels
        xyz_min, xyz_max = self.bbox[0,:], self.bbox[1,:]
        dim = len(xyz_min)
        voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
        # scale the original resolution by a option scale (only effective before first upsample)
        scale = torch.tensor(opt.train_schedule.resolution_scale_init,device=self.bbox.device)
        return ((xyz_max - xyz_min) / voxel_size * scale).long().tolist()

    def _find_n_samples(self, opt,  resolution: List[int]):
        auto_sample_number = int(np.linalg.norm(resolution)/opt.nerf.step_ratio) # auto adjustment sampling step with
        n_samples = min(int(opt.nerf.sample_intvs), auto_sample_number)
        return n_samples

    def _get_optimizer(self, opt, it=0, lr_basis=None , lr_index=None):
        # reset lr if lr_upsample_reset is true (default) , else, continue exponential lr decay schedule
        if lr_basis==None and lr_index==None:
            lr_scale = 1.0 if (opt.optim.lr_upsample_reset and it in self.upsample_list) else opt.optim.lr_decay_target_ratio ** (it / opt.max_iter)
            self.lr_basis = opt.optim.lr_basis * lr_scale
            self.lr_index = opt.optim.lr_index * lr_scale
        else:
            # when restoring lr from a checkpoint
            self.lr_basis = lr_basis
            self.lr_index = lr_index
        grad_vars = self.tensorf.get_optparam_groups(self.lr_index, self.lr_basis)
        if opt.optim.algo == "Adam":
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99)) # different from default betas(0.9, 0.999)
        else:
            optimizer = getattr(torch.optim, opt.optim.algo)(grad_vars)
        return optimizer

    def _update_alphamask(self, it):
        if it not in self.update_alphamask_iters:
            return
        if self.resolution[0] * self.resolution[1] * self.resolution[2] <256**3:# update volume resolution
            self.alphamask_resolution = self.resolution
            new_aabb_bbox = self.tensorf.updateAlphaMask(tuple(self.alphamask_resolution))
            if it == self.update_alphamask_iters[0]:
                # update bbox when we first update alpha mask
                self.tensorf.shrink(new_aabb_bbox)
                self.bbox = new_aabb_bbox

    def get_reset_kwargs(self):
        return {
            'bbox': self.bbox
            ,'n_voxel_list': self.n_voxel_list
            ,'n_voxels': self.n_voxels
            ,'alphamask_resolution': self.alphamask_resolution
            ,'lr_basis': self.lr_basis
            ,'lr_index': self.lr_index
            ,'TV_weight_color': self.TV_weight_color
            ,'TV_weight_density': self.TV_weight_density
        }

    # states that need to be saved besides mondel parameters state_dict
    def save_param_state(self):
        ckpt = self.tensorf.save_param_state()
        reset_kwargs = self.get_reset_kwargs()
        ckpt.update({"nerf_reset_kwargs": reset_kwargs})
        return ckpt
    # states that need to be saved besides mondel parameters state_dict
    def load_param_state(self, opt, ckpt):
        # reset self
        reset_kwargs = ckpt["nerf_reset_kwargs"]
        self.reset(opt=opt, **reset_kwargs)
        # reset tensorf
        self.tensorf.load_param_state(ckpt)
        # reset optimizer with new learning rate
        if self.register_new_optimizer is not None:
            it = ckpt["iter"]
            # because the optimizer hood is initialized
            # we are in resume training  traninig mode
            lr_basis = reset_kwargs["lr_basis"]
            lr_index = reset_kwargs["lr_index"]
            optimizer = self._get_optimizer(opt, it, lr_basis, lr_index)
            self.register_new_optimizer(optimizer)
