from . import planar
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
from matplotlib import pyplot as plt
from . import kernels
# training cycle is same as planar


class Model(planar.Model):
    def __init__(self, opt):
        super().__init__(opt)

    def log_scalars(self, opt, var, loss, metric=None, step=0, split="train"):
        super().log_scalars(opt, var, loss, metric, step, split)
        if opt.arch.kernel_type == "seperate":
            content_kernel_type = opt.arch.content.kernel_type
        else:
            content_kernel_type = opt.arch.kernel_type

        if opt.arch.kernel_type != "none":
            # log scheduled kerenl sigma
            sigma = self.graph.neural_image.get_scheduled_sigma()
            self.tb.add_scalar(f"{split}/{'kernel_param'}", sigma , step)
            wandb.log({f"{split}.{'kernel_param'}": sigma}, step=step)
        # log rank
        rank = self.graph.neural_image.get_scheduled_rank()
        self.tb.add_scalar(f"{split}/{'rank'}", rank , step)
        wandb.log({f"{split}.{'rank'}": rank}, step=step)

        # log the gradient of sigma w.r.t the reconstruction Loss
        if content_kernel_type in ["gaussian"] and opt.log_multi_scale:
            sigma_scales = [2.0**i for i in range(-3, 8)]
            weights = torch.arange(1,11+1).float().to(opt.device)
            weights = weights / weights.sum()
            for sigma_scale, weight, exponent in zip(sigma_scales, weights, range(-3, 8)):
                sigma = torch.tensor(1.0).to(torch.float).to(opt.device).requires_grad_()
                xy_grid = warp.get_normalized_pixel_grid_crop(opt)
                xy_grid_warped = warp.warp_grid(opt,xy_grid,self.graph.warp_param.weight)
                # render images
                kernel = self.graph.neural_image.get_kernel(kernel_type="gaussian_diff", kernel_size=sigma_scale*6, external_diff_sigma=sigma*sigma_scale)
                rgb_warped = self.graph.neural_image.forward(opt,xy_grid_warped, external_kernel=kernel) # [B,HW,3]
                image_pert = var.image_pert.view(opt.batch_size, 3, opt.H_crop*opt.W_crop).permute(0, 2, 1)
                l2_loss = ((rgb_warped - image_pert)**2).mean(axis=2, keepdim=False).mean(axis=1, keepdim=False)

                # log all-patch grad w.r.t sigma
                total_grad_sigma = torch.autograd.grad(l2_loss.mean(), sigma, retain_graph=True)[0]
                #total_grad_sigma *= weight
                self.tb.add_scalar(f"P_all_sigma'_2^{exponent}", total_grad_sigma, step)
                wandb.log({f"P_all_grad_sigma'_2^{exponent}": total_grad_sigma}, step=step)

                # log all-patch grad w.r.t warp parameters
                total_grad_warp = torch.autograd.grad(l2_loss.mean(), self.graph.warp_param.weight, retain_graph=opt.log_per_patch_loss)[0]
                #total_grad_warp *= weight
                total_grad_warp_norm = torch.norm(total_grad_warp, dim=1)
                total_warp_delta = (self.graph.warp_param.weight - self.warp_pert)  # current warp - GT warp
                total_grad_warp_cosine = torch.nn.functional.cosine_similarity(total_grad_warp, total_warp_delta , dim=1)

                self.tb.add_scalar(f"P_all_warp'_norm_2^{exponent}", total_grad_warp_norm.mean(), step)
                wandb.log({f"P_all_warp'_norm_2^{exponent}": total_grad_warp_norm.mean()}, step=step)

                self.tb.add_scalar(f"P_all_warp'_cosine_2^{exponent}", total_grad_warp_cosine.mean(), step)
                wandb.log({f"P_all_warp'_cosine_2^{exponent}": total_grad_warp_cosine.mean()}, step=step)

                if opt.log_per_patch_loss:
                    # log per-patch loss
                    for b in range(opt.batch_size):
                        # log per-patch grad w.r.t sigma
                        retain_graph = b != opt.batch_size - 1
                        patch_grad = torch.autograd.grad(l2_loss[b], sigma, retain_graph=retain_graph)[0]
                        #patch_grad *= weight
                        self.tb.add_scalar(f"P_{b}_sigma'_2^{exponent}", patch_grad, step)
                        wandb.log({f"P_{b}_sigma'_2^{exponent}": patch_grad}, step=step)

                        # log per-patch grad w.r.t warp parameters
                        self.tb.add_scalar(f"P_{b}_warp'_norm_2^{exponent}", total_grad_warp_norm[b], step)
                        wandb.log({f"P_{b}_warp'_norm_2^{exponent}": total_grad_warp_norm[b]}, step=step)
                        self.tb.add_scalar(f"P_{b}_warp'_cosine_2^{exponent}", total_grad_warp_cosine[b], step)
                        wandb.log({f"P_{b}_warp'_cosine_2^{exponent}": total_grad_warp_cosine[b]}, step=step)

    def visualize(self,opt,var,step=0,split="train"):
        super().visualize(opt,var,step,split)
        def process_kernel(kernel_sample):
            # padd kernel
            max_kernel_size = self.graph.neural_image.kernel_size
            padd_len = (max_kernel_size - kernel_sample.shape[0]) // 2
            kernel_sample = torch.nn.functional.pad(kernel_sample, (padd_len, padd_len))
            # compute fft
            kernel_spectrum = torch.abs(torch.fft.fftshift(torch.fft.fft(kernel_sample)))
            # convert to numpy
            kernel_sample = kernel_sample.detach().cpu().numpy()
            kernel_spectrum = kernel_spectrum.detach().cpu().numpy()
            return kernel_sample, kernel_spectrum

        if opt.arch.kernel_type == "seperate":
            warp_kernel_sample, warp_kernel_spectrum = process_kernel(var.kernel_warp)
            content_kernel_sample, content_kernel_spectrum = process_kernel(var.kernel_content)
            # log kernel
            fig = plt.figure()
            plt.title(f"seperate kernel: warp:{opt.arch.warp.kernel_type}, content:{opt.arch.content.kernel_type}")
            plt.plot(warp_kernel_sample)
            plt.plot(content_kernel_sample)
            wandb.log({f"{split}.{'kernel'}": wandb.Image(fig)}, step=step)
            plt.close(fig)
            # log fft transform of kernel
            fig = plt.figure()
            plt.title(f"seperate kernel: warp:{opt.arch.warp.kernel_type}, content:{opt.arch.content.kernel_type}")
            plt.plot(warp_kernel_spectrum)
            plt.plot(content_kernel_spectrum)
            wandb.log({f"{split}.{'kernel_fft'}": wandb.Image(fig)}, step=step)
            plt.close(fig)
        else:
            kernel_sample, kernel_spectrum = process_kernel(var.kernel)
            # log kernel
            fig = plt.figure()
            plt.title(f"{opt.arch.kernel_type}")
            plt.plot(kernel_sample)
            wandb.log({f"{split}.{'kernel'}": wandb.Image(fig)}, step=step)
            plt.close(fig)
            # log fft transform of kernel
            fig = plt.figure()
            plt.title(f"{opt.arch.kernel_type}")
            plt.plot(kernel_spectrum)
            wandb.log({f"{split}.{'kernel_fft'}": wandb.Image(fig)}, step=step)
            plt.close(fig)




# ============================ computation graph for forward/backprop ============================


class Graph(planar.Graph):

    def __init__(self, opt):
        super().__init__(opt)
    def compute_loss(self, opt, var, mode=None):
        loss = edict()
        if opt.loss_weight.render is not None:
            image_pert = var.image_pert.view(
                opt.batch_size, 3, opt.H_crop*opt.W_crop).permute(0, 2, 1)
            if opt.arch.kernel_type == "seperate":
                loss.render = self.MSE_loss(var.rgb_warped, image_pert) # MSE loss when learning neural image content
                loss.render_warp = self.MSE_loss(var.rgb_warped_additional, image_pert) # MSE loss when learning warping parameters
            else:
                loss.render = self.MSE_loss(var.rgb_warped, image_pert)
        if opt.loss_weight.total_variance is not None:
            loss.total_variance = self.TV_loss(self.neural_image)
        return loss

    def forward(self,opt,var,mode=None):
        if opt.arch.kernel_type == "seperate":
            assert opt.arch.kernel_type == "seperate"
            # the blurness of kernel depends on scheduled t parameter

            # learn content
            self.warp_param.requires_grad = False # freeze warping parameters
            for param in self.neural_image.parameters(): # learn neural_image content, including rank1, rank2
                param.requires_grad = True
            content_setting = opt.arch.content
            content_kernel_size = content_setting.kernel_size
            content_kernel_type = content_setting.kernel_type

            sigma = interp_schedule(self.neural_image.progress, content_setting.c2f_kernel)
            sigma = sigma* content_setting.sigma_scale
            kernel_content = self.neural_image.get_kernel(content_kernel_type, content_kernel_size, sigma=sigma)

            var.rgb_warped, var.rgb_warped_map, var.kernel_content = self.neural_image_forward(opt, external_kernel=kernel_content, return_kernel=True) # keep var entry name for compatibility with visuialization and logging
            var.kernel_content = kernel_content.detach()

            # learn warp
            self.warp_param.requires_grad = True # learn warping parameters
            for param in self.neural_image.parameters(): # freeze neural_image content , including rank1, rank2, diff_sigma
                param.requires_grad = False
            warp_setting = opt.arch.warp
            warp_kernel_size = warp_setting.kernel_size
            warp_kernel_type = warp_setting.kernel_type

            sigma = interp_schedule(self.neural_image.progress, warp_setting.c2f_kernel)
            sigma = sigma *  warp_setting.sigma_scale
            kernel_warp = self.neural_image.get_kernel(warp_kernel_type, warp_kernel_size, sigma=sigma)

            var.rgb_warped_additional, var.rgb_warped_map_additional, var.kernel_warp = self.neural_image_forward(opt, external_kernel=kernel_warp, return_kernel=True)
            var.kernel_warp = var.kernel_warp.detach()

            # turn on the grad of content for backprop
            for param in self.neural_image.parameters():
                param.requires_grad = True

        else:
            var.rgb_warped, var.rgb_warped_map, var.kernel = self.neural_image_forward(opt, return_kernel=True)
        return var

    def neural_image_forward(self,opt, external_kernel=None, return_kernel=False):
        xy_grid = warp.get_normalized_pixel_grid_crop(opt)
        xy_grid_warped = warp.warp_grid(opt,xy_grid,self.warp_param.weight)
        # render images
        if return_kernel:
            rgb_warped, kernel = self.neural_image.forward(opt,xy_grid_warped, external_kernel=external_kernel, return_kernel=True) # [B,HW,3]
        else:
            rgb_warped = self.neural_image.forward(opt,xy_grid_warped, external_kernel=external_kernel)# [B,HW,3]

        rgb_warped_map = rgb_warped.view(opt.batch_size,opt.H_crop,opt.W_crop,3).permute(0,3,1,2) # [B,3,H,W]

        if return_kernel:
            return rgb_warped, rgb_warped_map, kernel
        return rgb_warped, rgb_warped_map

    def TV_loss(self, svdImage):
        # Total Variance Loss
        r1, r2 = svdImage.rank1, svdImage.rank2

        N1 = svdImage.resolution[0] * svdImage.max_ranks
        tv1 = (r1[...,1:] - r1[...,:-1])
        tv1 = tv1 * tv1
        tv1 = torch.sum(tv1) / N1
        N2 = svdImage.resolution[1] * svdImage.max_ranks
        tv2 = (r2[...,1:] - r2[...,:-1])
        tv2 = tv2 * tv2
        tv2 = torch.sum(tv2) / N2

        return tv1 + tv2


class NeuralImageFunction(torch.nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.initialize_param(opt)
        self.define_network(opt)
        # use Parameter so it could be checkpointed
        self.progress = torch.nn.Parameter(torch.tensor(0.))
        self.opt = opt

    def initialize_param(self,opt):
        # arch options
        self.kernel_type = opt.arch.kernel_type
        self.kernel_size = opt.arch.kernel_size
        self.resolution = opt.arch.resolution # W, H

        if self.kernel_type == "seperate":
            self.kernel_content = opt.arch.content
            self.kernel_warp = opt.arch.warp

        self.device = opt.device
        # c2f_schedule options
        self.c2f_kernel = opt.c2f_schedule.kernel_t
        self.c2f_rank = opt.c2f_schedule.rank

    def get_kernel(self, kernel_type, kernel_size, sigma=None, external_diff_sigma=None):
        if kernel_type in ["gaussian", "average"]:
            assert sigma is not None, "sigma should be provided for kernel_type={kernel_type}"
            sigma = torch.tensor((sigma), device=self.device)

        # smaller kernel size for small t, for faster computation
        if sigma!=None:
            kernel_size = min(int(sigma * 6) , kernel_size)

        # smaller kernel size for small t, for faster computation
        if external_diff_sigma !=None:
            kernel_size = min(int(external_diff_sigma * 6) , kernel_size)

        else:
            kernel_size = self.kernel_size
        if kernel_size % 2 == 0 :
            kernel_size += 1


        match kernel_type:
            case "gaussian":
                kernel = kernels.get_gaussian_kernel(sigma, kernel_size)
            case "average":
                kernel =  kernels.get_average_kernel(sigma, kernel_size)
            case _:
                raise ValueError(f"invalid kernel type at \"{kernel_type}\"")

        return kernel.to(self.device)

    def define_network(self, opt):
        self.max_ranks = opt.arch.max_ranks
        rank1 = torch.zeros(3, self.max_ranks, self.resolution[0]).float()
        rank1 = (torch.normal(rank1, 0.1))
        rank2 = torch.zeros(3, self.max_ranks, self.resolution[1]).float()
        rank2 = (torch.normal(rank2, 0.1))
        self.register_parameter(name='rank1', param=torch.nn.Parameter(rank1))
        self.register_parameter(name='rank2', param=torch.nn.Parameter(rank2))

    def get_scheduled_rank(self):
        return int(interp_schedule(self.progress, self.c2f_rank))

    def get_scheduled_sigma(self):
        return interp_schedule(self.progress, self.c2f_kernel)

    def forward(self, opt, coord_2D, external_kernel=None, mode=None, return_kernel=False):  # [B,...,3]

        cur_rank = self.get_scheduled_rank()
        # the blurness of kernel depends on scheduled t parameter
        sigma = self.get_scheduled_sigma()

        if external_kernel is not None:
            kernel = external_kernel
        else:
            kernel = self.get_kernel(self.kernel_type, self.kernel_size, sigma=sigma)

        kernel_expand = kernel.expand(cur_rank, 1, -1)

        r1_blur = torch_F.conv1d(self.rank1[:, :cur_rank, :], kernel_expand,
                                 bias=None, stride=1, padding="same", dilation=1, groups=cur_rank) # 3, n_comp, resolution
        r2_blur = torch_F.conv1d(self.rank2[:, :cur_rank, :], kernel_expand,
                                 bias=None, stride=1, padding="same", dilation=1, groups=cur_rank) # 3, n_comp, resolution

        B = coord_2D.shape[0]

        if opt.interp_before_product:
            x_coord = coord_2D[:,:,0].expand(3, -1, -1) # 3, B,  H*W,
            y_coord = coord_2D[:,:,1].expand(3, -1, -1) # 3, B,  H*W
            x_coord = torch.stack([x_coord, torch.zeros_like(x_coord)], dim=-1)
            y_coord = torch.stack([torch.zeros_like(y_coord), y_coord], dim=-1)

            x_rbgs = torch_F.grid_sample(r1_blur.unsqueeze(2), x_coord, align_corners=False, mode=opt.arch.grid_interp) #(3, n_component, B ,H*W)
            y_rbgs = torch_F.grid_sample(r2_blur.unsqueeze(3), y_coord, align_corners=False, mode=opt.arch.grid_interp)  #(3, n_component, B ,H*W)
            rbg = torch.sum(x_rbgs * y_rbgs, dim=1, keepdim=False) # 3, B, H*W
            sampled_rbg = rbg.permute(1,0,2)

        else:

            rbg = torch.sum(r1_blur.unsqueeze(
                2) * r2_blur.unsqueeze(3), dim=1, keepdim=False)
            assert rbg.shape == (
                3, self.resolution[1], self.resolution[0]), f"rbg image has shape {rbg.shape}"


            sampled_rbg = torch_F.grid_sample(rbg.expand(B, -1, -1, -1), coord_2D.unsqueeze(1), align_corners=False, mode=opt.arch.grid_interp).squeeze(2) #B, 3, H*W

        if return_kernel:
            return sampled_rbg.permute(0, 2, 1), kernel

        return sampled_rbg.permute(0, 2, 1)
