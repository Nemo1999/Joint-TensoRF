import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
from icecream import ic
from .tensorBase import TensorBase, raw2alpha
from .. import kernels
from icecream import ic

class BatBase(TensorBase):
    def get_kernel(self, opt, c2f_mode, c2f_parameter, c2f_kernel_size = 25):
        scale = torch.mean(self.gridSize / (self.aabb[1,:] - self.aabb[0,:]))
        if c2f_mode in ["uniform-gaussian"]:
            # all samples use the same kernel
            # c2f_parameter represent kernel width in camera coordinate
            kernel =  kernels.get_gaussian_kernel(scale * c2f_parameter, c2f_kernel_size)
            kernel = kernel.to(device=self.device, dtype=self.dtype)
        elif c2f_mode =="uniform-average":
            kernel = kernels.get_average_kernel(scale*c2f_parameter, c2f_kernel_size)
            kernel = kernel.to(device=self.device, dtype=self.dtype)
        else:
            raise RuntimeError(f"invalid c2f_mode {c2f_mode}")
        return kernel

    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled, self.kernel_density, self.c2f_mode)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma
        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])

        return alpha

    def forward(self,opt,  center, ray_dir, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1, c2f_parameter_density=None, c2f_parameter_color=None, c2f_mode=None, c2f_kernel_size=None, is_test_optim=False, view_pe_progress=1.0, fea_pe_progress=1.0):
        # sample points ( same as TensorBase)
        self.opt = opt
        two_stage_renderer =  self.opt.arch.shading.model in ["MLP_Fea_TwoStage2_2", "MLP_Fea_TwoStage1_3"]
        mlp_predict_density = hasattr(opt.arch.shading, "predict_density") and opt.arch.shading.predict_density
        viewdirs = ray_dir
        # cache MLP input formatting information for internal query
        self.abs_components = opt.arch.abs_components
        self.mlp_predict_density = mlp_predict_density
        self.view_pe_progress = view_pe_progress
        self.fea_pe_progress = fea_pe_progress
        self.component_wise_feature2density = opt.arch.component_wise_feature2density
        self.plane_feature2density = opt.arch.plane_feature2density
        self.convolve_plane_only = opt.arch.convolve_plane_only
        self.convolve_positive_only = opt.arch.convolve_positive_only
        self.ignore_negative_split = opt.arch.ignore_negative_split

        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(center, viewdirs, is_train=is_train,N_samples=N_samples, simulate_euclid_sample=opt.camera.ndc_simulate_euclid_sample ,simulate_euclid_depth=opt.camera.ndc_simulate_euclid_depth, ndc_near_plane=opt.arch.ndc_near_plane if hasattr(opt.arch,"ndc_near_plane") else 1.0)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(center, viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)


        mlp_viewdirs = viewdirs.detach() if self.opt.arch.shading.detach_viewdirs else viewdirs
        mlp_xyz_sampled = xyz_sampled.detach() if self.opt.arch.shading.detach_xyz else xyz_sampled

        blur_active = c2f_parameter_density!=None or c2f_parameter_color!=None
        if self.alphaMask is not None and blur_active == False:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid



        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        if two_stage_renderer:
            albedo = torch.clone(rgb)
        # get kernels for bluring
        self.c2f_mode = c2f_mode
        if c2f_mode != None:
            if is_test_optim == True:
                self.kernel_density = self.get_kernel(opt, "uniform-gaussian", c2f_parameter_density, c2f_kernel_size)
            else:
                self.kernel_density = self.get_kernel(opt, c2f_mode, c2f_parameter_density, c2f_kernel_size)
            color_c2f_mode = c2f_mode
            self.kernel_color = self.get_kernel(opt, color_c2f_mode, c2f_parameter_color, c2f_kernel_size)
        else:
            self.kernel_density = None
            self.kernel_color = None

        # compute features (same as tensorBase, except that we pass additional kernels)
        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            if mlp_predict_density:
                # get feature for both density and rgb
                app_features = self.compute_appfeature(xyz_sampled[ray_valid], self.kernel_color, c2f_mode, interp_mode=opt.arch.tensorf.grid_sample_interp_mode)
                valid_rgbs, sigma_feature = self.renderModule(mlp_xyz_sampled[ray_valid], mlp_viewdirs[ray_valid], app_features, view_pe_progress, fea_pe_progress)
            else:
                if is_test_optim and c2f_mode != None:
                    sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid], self.kernel_density, "uniform-gaussian", interp_mode=opt.arch.tensorf.grid_sample_interp_mode)
                else:
                    sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid], self.kernel_density, c2f_mode, interp_mode=opt.arch.tensorf.grid_sample_interp_mode)

            if self.component_wise_feature2density:
                validsigma = sigma_feature
            else:
                validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        if mlp_predict_density:
            app_mask = ray_valid
        else:
            app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            if mlp_predict_density:
                pass # `valid_rgbs` is already computed along with density
            else:
                app_features = self.compute_appfeature(xyz_sampled[app_mask], self.kernel_color, c2f_mode, interp_mode=self.opt.arch.tensorf.grid_sample_interp_mode)
                if two_stage_renderer:
                    valid_abedos, valid_rgbs = self.renderModule(mlp_xyz_sampled[app_mask], mlp_viewdirs[app_mask], app_features)
                    albedo[app_mask] = valid_abedos
                else:
                    valid_rgbs = self.renderModule(mlp_xyz_sampled[app_mask], mlp_viewdirs[app_mask], app_features, view_pe_progress, fea_pe_progress)
            rgb[app_mask] = valid_rgbs


        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)
        if two_stage_renderer:
            albedo_map = torch.sum(weight[..., None] * albedo, -2)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * ray_dir[..., -1]
            depth_map = depth_map  -  self.near_far[0] + 0.05
        opacity = acc_map


        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])
            if two_stage_renderer:
                albedo_map = albedo_map + (1. - acc_map[..., None])

        rgb_map = rgb_map.clamp(0,1)
        if two_stage_renderer:
            albedo_map = albedo_map.clamp(0,1)
            return albedo_map, rgb_map, depth_map, opacity

        else:
            return rgb_map, depth_map, opacity # rgb, sigma, alpha, weight, bg_weight
