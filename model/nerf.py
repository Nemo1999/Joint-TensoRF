import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict

import lpips
from external.pohsun_ssim import pytorch_ssim

import util,util_vis
from util import log,debug
from . import base
import camera
from camera import angle_to_rotation_matrix
import wandb
from icecream import  ic
import importlib
import os
import matplotlib.pyplot as plt
import util
from . import kernels
import math

# ============================ main engine for training and evaluation ============================

class Model(base.Model):

    def __init__(self,opt):
        super().__init__(opt)
        self.lpips_loss = lpips.LPIPS(net="alex").to(opt.device)

    def load_dataset(self,opt,eval_split="val", train_split="train"):
        super().load_dataset(opt,eval_split=eval_split, train_split=train_split)
        # prefetch all training data
        self.train_data.prefetch_all_data(opt)
        self.train_data.all = edict(util.move_to_device(self.train_data.all,opt.device))
        self.n_train_views = len(self.train_data.all.idx)

    def setup_optimizer(self,opt):
        log.info("setting up optimizers...")
        optimizer = getattr(torch.optim,opt.optim.algo)
        self.optim = optimizer([dict(params=self.graph.nerf.parameters(),lr=opt.optim.lr)])
        if opt.nerf.fine_sampling:
            self.optim.add_param_group(dict(params=self.graph.nerf_fine.parameters(),lr=opt.optim.lr))
        # set up scheduler
        if opt.optim.sched not in ["None", None]:
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched.type)
            if opt.optim.lr_end:
                assert(opt.optim.sched.type=="ExponentialLR")
                opt.optim.sched.gamma = (opt.optim.lr_end/opt.optim.lr)**(1./opt.max_iter)
            kwargs = { k:v for k,v in opt.optim.sched.items() if k!="type" }
            self.sched = scheduler(self.optim,**kwargs)
    @torch.no_grad()
    def process_GT_images(self,opt):
        """apply 2d Blur on Training GT image """

        # get kernel scales
        batch_size = len(self.train_data.all.image)
        if opt.c2f_alternate_2D_mode == "sample":
            scales = opt.c2f_alternate_2D_scale_pool
        else:
            scales = [0.0, 1.0]
        # get kernels
        kernels_dict = dict() # dictionary from scale to kernel
        for sc in scales:
            blur_param = util.interp_schedule(float(self.it/opt.max_iter), opt.blur_2d_c2f_schedule)
            blur_param = torch.tensor(blur_param, device=opt.device)
            blur_param *= sc
            # get kernel
            if opt.blur_2d_mode == "uniform-gaussian":
                kernel_width = blur_param * (opt.W + opt.H)/2
                kernel_1d = kernels.get_gaussian_kernel(kernel_width, opt.blur_2d_c2f_kernel_size)
            elif opt.blur_2d_mode == "uniform-average":
                kernel_width = blur_param * (opt.W + opt.H)/2
                kernel_1d = kernels.get_average_kernel(kernel_width, opt.blur_2d_c2f_kernel_size)
            else:
                raise NotImplementedError("illegal blur_2d_mode")
            kernel_1d = kernel_1d.to(device=opt.device,dtype=torch.float32)
            kernel_1d_log = kernel_1d.cpu().detach().numpy()
            fig = plt.figure()
            plt.title(f"2D GT Blur Kernel {sc}")
            plt.plot(kernel_1d_log)
            wandb.log({f"2D_GT_Blur_Kernel {sc}":wandb.Image(fig)}, step=self.it )
            plt.close(fig)
            kernel_1d = kernel_1d.expand(1,1,-1)

            kernels_dict[sc] = (kernel_1d, kernel_width)

        # generte blurred GT images
        blurred_gt_cached_images = dict()

        for sc, k in kernels_dict.items():
            kernel_1d, kernel_width = k
            # skip kernel if kernel_width too small
            if kernel_width < 0.01:
                images = self.train_data.all.image
            else:
                # perform 2D seperated convolution
                images = self.train_data.all.image.reshape(batch_size*3, opt.H,  opt.W)
                kernel_size = kernel_1d.shape[-1]
                pad_size= (kernel_size //2, kernel_size //2)
                images = torch_F.pad(images, pad_size, mode="replicate")
                images = torch_F.conv1d(images, kernel_1d.expand(opt.H,1,-1), bias=None, stride=1, padding=0, dilation=1, groups=opt.H)
                images = images.permute(0,2,1)
                images = torch_F.pad(images, pad_size, mode="replicate")
                images = torch_F.conv1d(images, kernel_1d.expand(opt.W,1,-1), bias=None, stride=1, padding=0, dilation=1, groups=opt.W)
                images = images.permute(0,2,1).reshape(batch_size, 3, opt.H, opt.W).contiguous()
            blurred_gt_cached_images[sc] = images
            util_vis.tb_wandb_image(opt, self.tb, self.it, "train", f"2d_blurred_GT_{sc}", images[0:opt.tb.num_images[0]*opt.tb.num_images[1]])
        return blurred_gt_cached_images

    @torch.no_grad()
    def get_edge_mask(self, opt, blurred_gt_cached_images):
        # get kernel scales
        batch_size = len(self.train_data.all.image)
        if opt.c2f_alternate_2D_mode == "sample":
            scales = opt.c2f_alternate_2D_scale_pool
        else:
            scales = [0.0, 1.0]
        # get kernels
        mask_dict = dict() # dictionary from scale to kernel
        # get sobel kernel
        Kx = torch.tensor([[1,0,-1],
                           [2,0,-2],
                           [1,0,-1]], device=opt.device, dtype=torch.float32)[None,None,...].expand(1,3,-1,-1)

        Ky = torch.tensor([[1,2,1],
                           [0,0,0],
                           [-1,-2,-1]], device=opt.device, dtype=torch.float32)[None, None,...].expand(1,3,-1,-1)
        for sc in scales:
            images = blurred_gt_cached_images[sc]
            pad_size = (1,1,1,1)
            images = torch_F.pad(images, pad_size, mode="replicate")
            Gx = torch_F.conv2d(images, Kx, padding=0)
            Gy = torch_F.conv2d(images, Ky, padding=0)
            GG = torch.sqrt(Gx**2 + Gy**2).view(batch_size,opt.H*opt.W)
            if hasattr(opt,"soft_edge_mask") and opt.soft_edge_mask:
                GG_max, _ = GG.max(dim=1,keepdim=True)
                GG_normalized = GG / GG_max
                mask_dict[sc] = GG_normalized
            else:
                GG_mean = GG.mean(dim=(1), keepdim=True)
                thresh = opt.hard_edge_mask_mean_thresh if hasattr(opt, "hard_edge_mask_mean_thresh") else 1.25
                GG_bool = (GG>GG_mean*thresh).to(torch.uint8)
                mask_dict[sc] = GG_bool # batch_size , H*W
        return mask_dict
    def train(self,opt):
        # before training
        log.title("TRAINING START")

        self.timer = edict(start=time.time(),it_mean=None)
        self.graph.train()
        self.ep = 0 # dummy for timer
        # training
        if self.iter_start==0: self.validate(opt,0)
        loader = tqdm.trange(opt.max_iter,desc="training",leave=False)

        # fix sampling groups to ensure all poses are equally supervised
        if hasattr(opt,"view_sampling_n_groups"):
            ng = opt.view_sampling_n_groups
            all_views = torch.randperm(self.n_train_views,device=opt.device)
            self.group_idx = [all_views[i::ng] for i in range(ng)]

        for self.it in loader:
            if hasattr(opt, "early_stop_iter") and opt.early_stop_iter == self.it:
                print("early stop!! EXIT EXIT EXIT")
                exit(0)
            self.graph.it = self.it
            if self.it % 500 == 0 and hasattr(opt,"blur_2d") and opt.blur_2d == True:
                # update blur image cache
                self.blurred_gt_cached_images = self.process_GT_images(opt)
                self.blurred_edge_masks = self.get_edge_mask(opt, self.blurred_gt_cached_images)
            if self.it<self.iter_start: continue
            if hasattr(opt.train_schedule, "change_n_rays_after_n_iters"):
                if self.it < opt.train_schedule.change_n_rays_after_n_iters:
                    opt.nerf.n_rays = opt.train_schedule.n_rays_init
                else:
                    opt.nerf.n_rays = opt.train_schedule.n_rays_rest

            if hasattr(opt.train_schedule, "change_n_AccumPoseGrad_after_n_iters"):
                if self.it < opt.train_schedule.change_n_AccumPoseGrad_after_n_iters:
                    opt.optim.pose_grad_accum_iter = opt.train_schedule.n_AccumPoseGrad_init
                else:
                    opt.optim.pose_grad_accum_iter = opt.train_schedule.n_AccumPoseGrad_rest

            if hasattr(opt.train_schedule, "change_n_AccumGrad_after_n_iters"):
                if self.it < opt.train_schedule.change_n_AccumGrad_after_n_iters:
                    opt.optim.grad_accum_iter = opt.train_schedule.n_AccumGrad_init
                else:
                    opt.optim.grad_accum_iter = opt.train_schedule.n_AccumGrad_rest
            if hasattr(opt.train_schedule,"reset_pose_on_iter") and opt.train_schedule.reset_pose_on_iter == self.it:
                with torch.no_grad():
                    # get graph
                    self.interrupt_pose(opt)

            if hasattr(opt.train_schedule,"reset_pose_on_iters") and self.it in opt.train_schedule.reset_pose_on_iters :
                with torch.no_grad():
                    # get graph
                    self.interrupt_pose(opt)

            if hasattr(opt.train_schedule, "all_view_sample_after_n_iters") and self.it == opt.train_schedule.all_view_sample_after_n_iters:
                opt.nerf.ray_sampling_strategy = "all_view_rand_rays"
            if hasattr(opt.train_schedule, "single_view_sample_after_n_iters") and self.it == opt.train_schedule.single_view_sample_after_n_iters:
                opt.nerf.ray_sampling_strategy = "single_view_rand_rays"
                            # blur 2D supervising images

            if hasattr(opt,"blur_2d") and opt.blur_2d == True:
                gt_blurred = self.blurred_gt_cached_images
                edge_masks = self.blurred_edge_masks
                # alternate blur kernel
                if hasattr(opt, "c2f_alternate_2D_blur") and opt.c2f_alternate_2D_blur == True:
                    if opt.c2f_alternate_2D_mode == "sample":
                        sc = np.random.choice(opt.c2f_alternate_2D_scale_pool)
                        train_images = gt_blurred[sc]
                        if hasattr(opt, "edge_mask_use_scale"):
                            train_edge_masks = edge_masks[opt.edge_mask_use_scale]
                        else:
                            train_edge_masks = edge_masks[sc]
                else:
                    train_images = gt_blurred[1.0]
                    train_edge_masks = edge_masks[1.0]
            else:
                train_images = self.train_data.all.image
                train_edge_masks = None


            # create new dictionary wrapper to prevent overwrite
            var = edict({**self.train_data.all})
            var.image = train_images
            var.train_edge_masks = train_edge_masks
            if hasattr(opt,"sync_2d_3d_scales") and opt.sync_2d_3d_scales:
                var.scale = sc
                self.graph.scale = sc

            # generate subsample view index
            if opt.nerf.ray_sampling_strategy == "single_view_rand_rays":
                # extract single view from taining data
                self.view_index = self.it % self.n_train_views
                self.graph.view_index = self.view_index
                view_sample_index = [self.view_index]
            elif hasattr(opt,"view_sampling_n_groups"):
                view_sample_index = self.group_idx[self.it% opt.view_sampling_n_groups]
            else:
                view_sample_index = None # sample all views

            # subsample view by index
            if view_sample_index != None:
                var.image = var.image[view_sample_index,...]
                var.pose  = var.pose[view_sample_index,...]
                var.intr_inv = var.intr_inv[view_sample_index,...]
                var.idx = var.idx[view_sample_index,...]
                var.intr = var.intr[view_sample_index,...]

            self.train_iteration(opt,var,loader)
            if hasattr( self.graph.nerf, "update_schedule"):
                # update schedule according to iterations ( used for tensorf voxel upsampling and alphaMask updating )
                self.graph.nerf.update_schedule(opt, self.it)
            if hasattr(opt.optim,"sched") and opt.optim.sched not in ["None",None]: self.sched.step()
            if self.it%opt.freq.val==0: self.validate(opt,self.it)
            if hasattr(opt.freq, "vis_pose") and self.it%opt.freq.vis_pose == 0: self.visualize_pose(opt, self.it)
            if hasattr(opt.freq, "vis_train") and self.it%opt.freq.vis_train == 0: self.visualize_train(opt, self.it)
            if self.it%opt.freq.ckpt==0: self.save_checkpoint(opt,ep=None,it=self.it)
            if opt.visualize_gradient and opt.nerf.ray_sampling_strategy=="single_view_rand_rays" and self.view_index == 0:
                if (self.it // self.n_train_views) % opt.freq.vis_grad_every_n_cycles == 0:
                    self.visualize_gradient( opt, var, self.it)
            if self.it in opt.generate_video_iters:
                self.generate_videos_synthesis(opt,eps=1e-10, it=self.it)

        self.save_checkpoint(opt,ep=None,it=self.it,latest=True)
        # after training
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        if opt.visdom: self.vis.close()
        log.title("TRAINING DONE")

    @torch.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train"):
        super().log_scalars(opt,var,loss,metric=metric,step=step,split=split)
        # log learning rate
        if split=="train":
            lr = self.optim.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split,"lr"),lr,step)
            wandb.log({f"{split}.{'lr'}": lr}, step=step)
            if opt.nerf.fine_sampling:
                lr = self.optim.param_groups[1]["lr"]
                self.tb.add_scalar("{0}/{1}".format(split,"lr_fine"),lr,step)
                wandb.log({f"{split}.{'lr_fine'}": lr}, step=step)
        # compute PSNR
        psnr = -10*loss.render.log10()
        self.tb.add_scalar("{0}/{1}".format(split,"PSNR"),psnr,step)
        wandb.log({f"{split}.{'PSNR'}": psnr}, step=step)
        if opt.nerf.fine_sampling:
            psnr = -10*loss.render_fine.log10()
            self.tb.add_scalar("{0}/{1}".format(split,"PSNR_fine"),psnr,step)
            wandb.log({f"{split}.{'PSNR_fine'}": psnr}, step=step)

    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train",eps=1e-10):
        if opt.tb:
            if step == opt.freq.vis:
                # only load training image on 0 iteration, saves alot of spaces
                util_vis.tb_wandb_image(opt,self.tb,step,split,"GT_images",var.image)
            if split!="train":
                invdepth = (var.depth) if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
                rgb_map = var.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2).expand(-1,3,-1,-1) # [B,1,H,W]
                util_vis.tb_wandb_image(opt,self.tb,step,split,"rgb",rgb_map, log_wandb=True)
                util_vis.tb_wandb_image(opt,self.tb,step,split,"invdepth",invdepth_map, log_wandb=True, from_range=(0.05**(-1),0))

                vis_number = step // opt.freq.vis
                vis_path = os.path.join(opt.output_path, "vis_images")
                os.makedirs(vis_path, exist_ok=True)
                torchvision_F.to_pil_image(rgb_map.cpu()[0]).save(os.path.join(vis_path, f"rgb_{vis_number}.png"))
                if opt.camera.ndc:
                    min_r, max_r = 0.05, opt.nerf.depth.range[1] - opt.nerf.depth.range[0]
                    invdepth_map_normalized = (invdepth_map - min_r) / (max_r - min_r)
                else:
                    invdepth_map_normalized = invdepth_map
                torchvision_F.to_pil_image(invdepth_map_normalized.cpu()[0]).save(os.path.join(vis_path, f"depth_{vis_number}.png"))
                if opt.nerf.fine_sampling:
                    invdepth = (1-var.depth_fine)/var.opacity_fine if opt.camera.ndc else 1/(var.depth_fine/var.opacity_fine+eps)
                    rgb_map = var.rgb_fine.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                    invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                    util_vis.tb_wandb_image(opt,self.tb,step,split,"rgb_fine",rgb_map, log_wandb=False)
                    util_vis.tb_wandb_image(opt,self.tb,step,split,"invdepth_fine",invdepth_map.expand(-1,3,-1,-1), log_wandb=False,from_range=(opt.nerf.depth.range[0]**(-1),0))

                if vis_number == ((opt.max_iter-1) // opt.freq.vis):
                    print("writing visualization videos...")
                    rgb_vid_fname = os.path.join(opt.output_path, "vis_rgb.webm")
                    depth_vid_fname = os.path.join(opt.output_path, "vis_invdepth.webm")
                    os.system("ffmpeg -y -framerate 5 -i {0}/rgb_%d.png -vcodec libvpx-vp9 -pix_fmt yuv420p {1} >/dev/null 2>&1".format(vis_path,rgb_vid_fname))
                    wandb.log({"val_rgb_vid": wandb.Video(rgb_vid_fname)})
                    os.system("ffmpeg -y -framerate 5 -i {0}/depth_%d.png -vcodec libvpx-vp9 -pix_fmt yuv420p {1} >/dev/null 2>&1".format(vis_path,depth_vid_fname))
                    wandb.log({"val_depth_vid": wandb.Video(depth_vid_fname)})

    @torch.no_grad()
    def visualize_pose(self, opt, step, split="train"):
        self.graph.eval()
        if not opt.tb: return
        if opt.model not in ["barf", "bat" ]: return
        cam_path = os.path.join(opt.output_path, "vis_cam")
        os.makedirs(cam_path, exist_ok=True)

        fig = plt.figure(figsize=(10,10) if opt.data.dataset in ["blender","t2"] else (16,8))

        pose, pose_ref = self.get_all_training_poses(opt)

        if opt.data.dataset in ["blender", "llff", "t2"]:
            if split=="train":
                pose_aligned , _ = self.prealign_cameras(opt, pose, pose_ref)
                pose_aligned = pose_aligned.detach().cpu()
            else:
                pose_aligned = pose.detach().cpu()
            pose_ref = pose_ref.detach().cpu()
            dict(
                blender=util_vis.plot_save_poses_blender,
                t2=util_vis.plot_save_poses_t2,
                llff=util_vis.plot_save_poses
            )[opt.data.dataset](opt, fig, pose_aligned, pose_ref=pose_ref, path=cam_path, ep=step)
        else:
            pose = pose.detach().cpu()
            util_vis.plot_save_poses(opt, fig, pose, pose_ref=None, path=cam_path, ep=step)
        plt.close(fig)

        # the test-time pose visualization will passed with step = opt.max_iter + 1
        if split == "test":
            wandb.log({"final_raw_pose_visualization": wandb.Image(os.path.join(cam_path, f'{opt.max_iter+1}.png'))}, step=opt.max_iter+1)
            return


        # log training pose visualization into wandb
        wandb.log({"camera_visualization": wandb.Image(os.path.join(cam_path, f'{step}.png'))}, step=step)

        list_file = os.path.join(cam_path, "cam_vis_image_list.txt")
        # write image name saved by `util_vis.plot_save_poses` into a temp file, which will be fed into ffmpeg latter
        with open(list_file, "a") as f:
            f.write(f"file {step}.png\n")
        # log training pose video
        if step // opt.freq.vis_pose == (opt.max_iter-1) // opt.freq.vis_pose:
            # last time calling visualize_pose, we collect all the images into a video
            cam_vid_fname = os.path.join(cam_path, "cam_poses.webm")
            os.system("ffmpeg -y -r 12 -f concat -i {0} -vcodec libvpx-vp9 -pix_fmt yuv420p {1} >/dev/null 2>&1".format(list_file ,cam_vid_fname))
            wandb.log({"camera_visualization_video": wandb.Video(cam_vid_fname)})
            # os.remove(cam_vid_fname)

    @torch.no_grad()
    def visualize_train(self, opt, step, eps=1.e-6):
        self.graph.eval()
        # only visualize if we are in barf or bat mode
        if not opt.tb: return
        if opt.model not in ["barf", "bat", "tensorf"]: return
        train_vis_path = os.path.join(opt.output_path, "train_vis")
        os.makedirs(train_vis_path, exist_ok=True)
        train_vis_number = step // opt.freq.vis_train

        # get current training poses
        if opt.model in ["barf", "bat"]:
            current_training_poses , _ = self.get_all_training_poses(opt)
        else:
            current_training_poses = self.get_all_training_poses(opt)[1]


        current_training_poses = current_training_poses[: opt.tb.num_images[0]*opt.tb.num_images[1]] # render images for visualization
        intr_inv = edict(next(iter(self.test_loader))).intr_inv[:1].to(opt.device) # grab intrinsics
        intr = edict(next(iter(self.test_loader))).intr[:1].to(opt.device) # grab intrinsics
        current_training_poses_tqdm = tqdm.tqdm(current_training_poses, desc="rendering current trianing views", leave=True)
        test_rgbs = []
        test_invdepths = []
        for i, pose in enumerate(current_training_poses_tqdm):
            ret = self.graph.render_by_slices(opt, pose[None], intr_inv=intr_inv, intr=intr)
            invdepth = (ret.depth)  if opt.camera.ndc else 1/(ret.depth/ret.opacity+eps)
            rgb_map = ret.rgb.view(opt.H, opt.W, 3).permute(2, 0, 1)
            invdepth_map = invdepth.view(opt.H, opt.W, 1).permute(2, 0, 1)
            test_rgbs.append(rgb_map.cpu())
            test_invdepths.append(invdepth_map.cpu())
        test_rgbs = torch.stack(test_rgbs)
        test_invdepths = torch.stack(test_invdepths)
        grid_rgb = util_vis.tb_wandb_image(opt, self.tb, step, "train", "trianing_poses_supervised_images", test_rgbs)
        if opt.camera.ndc:
            min_r, max_r = 0.05, opt.nerf.depth.range[1] - opt.nerf.depth.range[0]
            test_invdepths = (test_invdepths - min_r) / (max_r - min_r)

        grid_invdepth = util_vis.tb_wandb_image(opt, self.tb, step, "train", "trianing_poses_supervised_invdepths", test_invdepths.expand(-1,3,-1,-1))
        torchvision_F.to_pil_image(grid_rgb.cpu()).save(os.path.join(train_vis_path, f"rgb_{train_vis_number}.png"))

        std = torch.std(grid_invdepth,dim=(1,2))
        std = torch.max(std, torch.full_like(std, 0.00001))
        grid_invdepth = torchvision_F.normalize(grid_invdepth, mean=torch.mean(grid_invdepth,dim=(1,2)), std=std)
        torchvision_F.to_pil_image(grid_invdepth.cpu()).save(os.path.join(train_vis_path, f"depth_{train_vis_number}.png"))

        if train_vis_number == (opt.max_iter-1) // opt.freq.vis_train:
            print("writing training view videos")
            rgb_vid_fname = os.path.join(opt.output_path, "train_vis_rgb.webm")
            depth_vid_fname = os.path.join(opt.output_path, "train_vis_depth.webm")
            os.system("ffmpeg -y -framerate 5 -i {0}/rgb_%d.png -vcodec libvpx-vp9 -pix_fmt yuv420p {1} >/dev/null 2>&1".format(train_vis_path,rgb_vid_fname))
            wandb.log({"train_rgb_vid": wandb.Video(rgb_vid_fname)})
            os.system("ffmpeg -y -framerate 5 -i {0}/depth_%d.png -vcodec libvpx-vp9 -pix_fmt yuv420p {1} >/dev/null 2>&1".format(train_vis_path,depth_vid_fname))
            wandb.log({"train_depth_vid": wandb.Video(depth_vid_fname)})

    @torch.no_grad()
    def visualize_gradient(self,opt ,var ,step):
        # plot pose gradient on training view (currently only plot the first training view)
        grad_rotation = edict(X=[],Y=[],Z=[])
        grad_translation = edict(X=[],Y=[],Z=[])
        pbar = tqdm.tqdm(range(0, opt.H * opt.W, opt.vis_grad_n_rays))
        pbar.set_description("Visualizing Pose Grad")
        for c in pbar:
            ray_idx = torch.arange(c,min(c+opt.vis_grad_n_rays, opt.H*opt.W), device=opt.device)
            def render_mse(angles, translations):
                #angles = torch.zeros(3).to(opt.device)
                #translations = torch.zeros(3).to(opt.device)
                rmats = [angle_to_rotation_matrix(angles[i], ax) for i, ax in enumerate(["X", "Y", "Z"])]
                pose = self.graph.get_pose(opt, var, mode="train")
                delta_pose = camera.pose(R=rmats[0]@rmats[1]@rmats[2], t=translations)
                pose = camera.pose.compose([delta_pose, pose])
                # the pose is same for every iteration

                # render pose with current ray_idx
                ret = self.graph.render(opt, pose, ray_idx=ray_idx, mode="train", intr_inv=var.intr_inv, intr=var.intr)
                image = var.image.view(1,3,opt.H*opt.W).permute(0,2,1) # num_images, H*W, 3
                image = image[:,ray_idx,:]
                per_pixel_mse = (ret.rgb.contiguous() - image)**2
                return per_pixel_mse # 1, len(ray_idx) , 3

            # find gradient w.r.t R and t (jacobian is better )
            # angle_grads, translation_grads = torch.autograd.grad(
            #     per_pixel_mse,
            #     [angles, translations],
            #     grad_outputs=torch.ones_like(per_pixel_mse),
            #     create_graph=False,
            #     retain_graph=False,
            #    is_grads_batched=False)

            angles = torch.zeros(3).to(opt.device)
            translations = torch.zeros(3).to(opt.device)
            angle_grads, trans_grads = torch.autograd.functional.jacobian(render_mse, (angles, translations), create_graph=False, strict=False, vectorize=True)
            #angle_grads.shape=torch.Size([1, 128, 3, 3])
            #trans_grads.shape=torch.Size([1, 128, 3, 3])

            for i, ax in enumerate(["X", "Y", "Z"]):
                grad_rotation[ax].append(angle_grads[..., i])
                grad_translation[ax].append(trans_grads[..., i])

        for i, ax in enumerate(["X", "Y", "Z"]):
            grad_rotation[ax] = torch.cat(grad_rotation[ax],axis=1) # 1, H*W, 3
            grad_translation[ax] = torch.cat(grad_translation[ax], axis=1) # 1, H*W, 3
            grad_rotation[ax] = grad_rotation[ax].view(1, opt.H, opt.W, 3).permute(0,3,1,2)
            grad_translation[ax] = grad_translation[ax].view(1, opt.H, opt.W, 3).permute(0,3,1,2)
        rot_grads = torch.cat([grad_rotation[ax] for ax in "XYZ"],axis=0)
        trans_grads = torch.cat([grad_translation[ax] for ax in "XYZ"],axis=0)

        # we plot heatmap to distinguish positive and negative
        rot_grads = torch.sum(rot_grads, axis=1, keepdim=True)
        trans_grads = torch.sum(trans_grads, axis=1, keepdim=True)
        print(f"{rot_grads.shape=}")
        print(f"{trans_grads.shape=}")
        # log image
        util_vis.tb_wandb_image(opt,
                                self.tb,
                                step,
                                group="train",
                                name="pose_R_jacob_XYZ",
                                images=rot_grads,
                                num_vis=[1,3]
        )
        util_vis.tb_wandb_image(opt,
                                self.tb,
                                step,
                                group="train",
                                name="pose_t_jacob_XYZ",
                                images=trans_grads,
                                num_vis=[1,3]
        )

    @torch.no_grad()
    def get_all_training_poses(self,opt):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device)
        return None,pose_GT

    @torch.no_grad()
    def evaluate_full(self,opt,eps=1e-10):
        self.graph.eval()
        loader = tqdm.tqdm(self.test_loader,desc="evaluating",leave=False)
        res = []
        test_path = "{}/test_view".format(opt.output_path)
        os.makedirs(test_path,exist_ok=True)
        for i,batch in enumerate(loader):
            var = edict(batch)
            var = util.move_to_device(var,opt.device)
            if opt.model in ["barf", "bat"] and opt.optim.test_photo:
                # run test-time optimization to factorize imperfection in optimized poses from view synthesis evaluation
                var = self.evaluate_test_time_photometric_optim(opt,var)
            with torch.no_grad():
                var = self.graph.forward(opt,var,mode="eval")
                # evaluate view synthesis
                invdepth = (var.depth) if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
                rgb_map = var.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                if opt.camera.ndc:
                    min_r, max_r = 0.05, opt.nerf.depth.range[1] - opt.nerf.depth.range[0]
                    invdepth_map_normalized = (invdepth_map - min_r) / (max_r - min_r)
                else:
                    invdepth_map_normalized = invdepth_map
                psnr = -10*self.graph.MSE_loss(rgb_map,var.image).log10().item()
                ssim = pytorch_ssim.ssim(rgb_map,var.image).item()
                lpips = self.lpips_loss(rgb_map*2-1,var.image*2-1).item()
            res.append(edict(psnr=psnr,ssim=ssim,lpips=lpips))
            # dump novel views
            torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/rgb_{}.png".format(test_path,i))
            torchvision_F.to_pil_image(var.image.cpu()[0]).save("{}/rgb_GT_{}.png".format(test_path,i))
            torchvision_F.to_pil_image(invdepth_map_normalized.cpu()[0]).save("{}/depth_{}.png".format(test_path,i))
        # show results in terminal
        print("--------------------------")
        print("PSNR:  {:8.2f}".format(np.mean([r.psnr for r in res])))
        print("SSIM:  {:8.2f}".format(np.mean([r.ssim for r in res])))
        print("LPIPS: {:8.2f}".format(np.mean([r.lpips for r in res])))
        print("--------------------------")

        wandb.run.summary["TEST_PSNR"] = np.mean([r.psnr for r in res])
        wandb.run.summary["TEST_SSIM"] = np.mean([r.ssim for r in res])
        wandb.run.summary["TEST_LPIPS"] = np.mean([r.lpips for r in res])

        # dump numbers to file
        quant_fname = "{}/quant.txt".format(opt.output_path)
        with open(quant_fname,"w") as file:
            for i,r in enumerate(res):
                file.write("{} {} {} {}\n".format(i,r.psnr,r.ssim,r.lpips))

    @torch.no_grad()
    def generate_videos_synthesis(self,opt,eps=1e-10, it=None):

        split = "eval" if it == None else "train"

        self.graph.eval()
        if opt.data.dataset in ["blender", "t2"]:
            pose_pred,pose_GT = self.get_all_training_poses(opt)
            poses = pose_pred if opt.model in ["barf", "bat"] else pose_GT
            if opt.model in ["barf", "bat"]:
                _,sim3 = self.prealign_cameras(opt,pose_pred,pose_GT)
                scale = sim3.s1/sim3.s0
            else: scale = 1
            pose_novel = camera.get_novel_view_around_bbox(opt, N=120,scale=scale).to(opt.device)

        else:
            pose_pred,pose_GT = self.get_all_training_poses(opt)
            poses = pose_pred if opt.model in ["barf", "bat"] else pose_GT
            if opt.model in ["barf", "bat"] and opt.data.dataset=="llff":
                _,sim3 = self.prealign_cameras(opt,pose_pred,pose_GT)
                scale = sim3.s1/sim3.s0
            else: scale = 1
            # rotate novel views around the "center" camera of all poses
            idx_center = (poses-poses.mean(dim=0,keepdim=True))[...,3].norm(dim=-1).argmin()
            pose_novel = camera.get_novel_view_poses(opt,poses[idx_center],N=60,scale=scale).to(opt.device)

        # render the novel views
        novel_path = "{}/novel_view".format(opt.output_path)
        os.makedirs(novel_path,exist_ok=True)
        pose_novel_tqdm = tqdm.tqdm(pose_novel,desc="rendering novel views",leave=False)
        intr_inv = edict(next(iter(self.test_loader))).intr_inv[:1].to(opt.device) # grab intrinsics
        intr = edict(next(iter(self.test_loader))).intr[:1].to(opt.device) # grab intrinsics

        for i,pose in enumerate(pose_novel_tqdm):
            ret = self.graph.render_by_slices(opt,pose[None],intr_inv=intr_inv, intr=intr)
            invdepth = (ret.depth)/ret.opacity if opt.camera.ndc else 1/(ret.depth/ret.opacity+eps)
            rgb_map = ret.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
            invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]

            if opt.camera.ndc:
                min_r, max_r = 0.05, opt.nerf.depth.range[1] - opt.nerf.depth.range[0]
                invdepth_map_normalized = (invdepth_map - min_r) / (max_r - min_r)
            else:
                invdepth_map_normalized = invdepth_map

            torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/rgb_{}.png".format(novel_path,i))
            torchvision_F.to_pil_image(invdepth_map_normalized.cpu()[0]).save("{}/depth_{}.png".format(novel_path,i))
        # write videos

        print("writing videos...")
        rgb_vid_fname = "{}/novel_view_rgb_{}.webm".format(opt.output_path, it)
        depth_vid_fname = "{}/novel_view_depth_{}.webm".format(opt.output_path, it)
        os.system("ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -vcodec libvpx-vp9 -pix_fmt yuv420p {1} >/dev/null 2>&1".format(novel_path,rgb_vid_fname))
        os.system("ffmpeg -y -framerate 30 -i {0}/depth_%d.png -vcodec libvpx-vp9 -pix_fmt yuv420p {1} >/dev/null 2>&1".format(novel_path,depth_vid_fname))

        if split == "train":
            wandb.log({"test_rgb_vid": wandb.Video(rgb_vid_fname)}, step=it)
            wandb.log({"test_depth_vid": wandb.Video(depth_vid_fname)}, step=it)
        else:
            # evaluation
            wandb.log({"evaluation_rgb_vid": wandb.Video(rgb_vid_fname)})
            wandb.log({"evaluation_depth_vid": wandb.Video(depth_vid_fname)})

# ============================ computation graph for forward/backprop ============================

class Graph(base.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        self.it = 0
        model_module = importlib.import_module("model.{}".format(opt.model))
        log.info("building nerf...")
        self.nerf = model_module.NeRF(opt).to(opt.device)
        if opt.nerf.fine_sampling:
            self.nerf_fine = model_module.NeRF(opt)

    def forward(self,opt,var,mode=None):
        batch_size = len(var.idx)
        pose = self.get_pose(opt,var,mode=mode)
        var.current_pose = pose
        # render images
        if mode in ["train","test-optim"]:
            # sample random rays for optimization
            if opt.nerf.ray_sampling_strategy == "all_view_rand_rays":
                var.ray_idx = torch.randperm(opt.H*opt.W,device=opt.device)[:opt.nerf.n_rays//batch_size]
            elif opt.nerf.ray_sampling_strategy == "all_view_rand_grid":
                rays_per_view = opt.nerf.n_rays // batch_size
                area_per_ray = opt.H * opt.W // rays_per_view
                step_size = math.ceil(area_per_ray ** 0.5)
                offset_x, offset_y = np.random.randint(step_size), np.random.randint(step_size)
                sample_x = torch.arange(offset_x,opt.W,step_size, device=opt.device)
                sample_y = torch.arange(offset_y,opt.H,step_size, device=opt.device)
                gY, gX = torch.meshgrid(sample_y, sample_x,indexing="ij")
                var.ray_idx = (gX + gY * opt.W).view(-1)
                var.ray_grid_step = step_size
                var.grid_H = len(sample_y)
                var.grid_W = len(sample_x)
            else:
                assert opt.nerf.ray_sampling_strategy == "single_view_rand_rays"
                var.ray_idx = torch.randperm(opt.H*opt.W,device=opt.device)[:opt.nerf.n_rays]
            ret = self.render(opt,pose,intr_inv=var.intr_inv,ray_idx=var.ray_idx,mode=mode, intr=var.intr) # [B,N,3],[B,N,1]
        else:
            # render full image (process in slices)
            ret = self.render_by_slices(opt,pose,intr_inv=var.intr_inv,mode=mode, intr=var.intr) # [B,HW,3],[B,HW,1]
        var.update(ret)
        return var

    def compute_loss(self,opt,var,mode=None):
        loss = edict()
        batch_size = len(var.idx)
        image = var.image.view(batch_size,3,opt.H*opt.W).permute(0,2,1)
        if mode in ["train","test-optim"]:
            image = image[:,var.ray_idx]
        # compute image losses
        if opt.loss_weight.render is not None:
            loss.render = self.MSE_loss(var.rgb,image)
        if opt.loss_weight.render_fine is not None:
            assert(opt.nerf.fine_sampling)
            loss.render_fine = self.MSE_loss(var.rgb_fine,image)
        return loss

    def get_pose(self,opt,var,mode=None):
        return var.pose

    def render(self,opt,pose,intr_inv=None,ray_idx=None,mode=None, intr=None):
        batch_size = len(pose) # B = n_views
        center,ray = camera.get_center_and_ray(opt,pose,intr_inv=intr_inv) # [B,HW,3]
        while ray.isnan().any(): # TODO: weird bug, ray becomes NaN arbitrarily if batch_size>1, not deterministic reproducible
            center,ray = camera.get_center_and_ray(opt,pose,intr_inv=intr_inv) # [B,HW,3]
            print("stuck in nerf.py line 469")
        if ray_idx is not None:
            # consider only subset of rays
            center,ray = center[:,ray_idx],ray[:,ray_idx]
        if opt.camera.ndc:
            assert intr!=None
            # convert center/ray representations to NDC
            center,ray = camera.convert_NDC(opt,center,ray,intr=intr)
        # render with main MLP
        depth_samples = self.sample_depth(opt,batch_size,num_rays=ray.shape[1]) # [B,HW,N,1]
        rgb_samples,density_samples = self.nerf.forward_samples(opt,center,ray,depth_samples,mode=mode)
        rgb,depth,opacity,prob = self.nerf.composite(opt,ray,rgb_samples,density_samples,depth_samples)
        ret = edict(rgb=rgb,depth=depth,opacity=opacity) # [B,HW,K]
        # render with fine MLP from coarse MLP
        if opt.nerf.fine_sampling:
            with torch.no_grad():
                # resample depth acoording to coarse empirical distribution
                depth_samples_fine = self.sample_depth_from_pdf(opt,pdf=prob[...,0]) # [B,HW,Nf,1]
                depth_samples = torch.cat([depth_samples,depth_samples_fine],dim=2) # [B,HW,N+Nf,1]
                depth_samples = depth_samples.sort(dim=2).values
            rgb_samples,density_samples = self.nerf_fine.forward_samples(opt,center,ray,depth_samples,mode=mode)
            rgb_fine,depth_fine,opacity_fine,_ = self.nerf_fine.composite(opt,ray,rgb_samples,density_samples,depth_samples)
            ret.update(rgb_fine=rgb_fine,depth_fine=depth_fine,opacity_fine=opacity_fine) # [B,HW,K]
        return ret

    def render_by_slices(self,opt,pose,intr_inv=None,mode=None, intr=None):
        ret_all = edict( rgb=[],depth=[],opacity=[])

        if opt.nerf.fine_sampling:
            ret_all.update( rgb_fine=[],depth_fine=[],opacity_fine=[])
        # render the image by slices for memory considerations
        for c in range(0,opt.H*opt.W,opt.nerf.n_rays):
            ray_idx = torch.arange(c,min(c+opt.nerf.n_rays,opt.H*opt.W),device=opt.device)
            ret = self.render(opt,pose,intr_inv=intr_inv,ray_idx=ray_idx,mode="vis", intr=intr) # [B,R,3],[B,R,1]
            for k in ret: ret_all[k].append(ret[k])
        # group all slices of images
        for k in ret_all: ret_all[k] = torch.cat(ret_all[k],dim=1)
        return ret_all

    def sample_depth(self,opt,batch_size,num_rays=None):
        depth_min,depth_max = opt.nerf.depth.range
        num_rays = num_rays or opt.H*opt.W
        rand_samples = torch.rand(batch_size,num_rays,opt.nerf.sample_intvs,1,device=opt.device) if opt.nerf.sample_stratified else 0.5
        rand_samples += torch.arange(opt.nerf.sample_intvs,device=opt.device)[None,None,:,None].float() # [B,HW,N,1]
        depth_samples = rand_samples/opt.nerf.sample_intvs*(depth_max-depth_min)+depth_min # [B,HW,N,1]
        depth_samples = dict(
            metric=depth_samples,
            inverse=1/(depth_samples+1e-8),
        )[opt.nerf.depth.param]
        return depth_samples

    def sample_depth_from_pdf(self,opt,pdf):
        depth_min,depth_max = opt.nerf.depth.range
        # get CDF from PDF (along last dimension)
        cdf = pdf.cumsum(dim=-1) # [B,HW,N]
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]),cdf],dim=-1) # [B,HW,N+1]
        # take uniform samples
        grid = torch.linspace(0,1,opt.nerf.sample_intvs_fine+1,device=opt.device) # [Nf+1]
        unif = 0.5*(grid[:-1]+grid[1:]).repeat(*cdf.shape[:-1],1) # [B,HW,Nf]
        idx = torch.searchsorted(cdf,unif,right=True) # [B,HW,Nf] \in {1...N}
        # inverse transform sampling from CDF
        depth_bin = torch.linspace(depth_min,depth_max,opt.nerf.sample_intvs+1,device=opt.device) # [N+1]
        depth_bin = depth_bin.repeat(*cdf.shape[:-1],1) # [B,HW,N+1]
        depth_low = depth_bin.gather(dim=2,index=(idx-1).clamp(min=0)) # [B,HW,Nf]
        depth_high = depth_bin.gather(dim=2,index=idx.clamp(max=opt.nerf.sample_intvs)) # [B,HW,Nf]
        cdf_low = cdf.gather(dim=2,index=(idx-1).clamp(min=0)) # [B,HW,Nf]
        cdf_high = cdf.gather(dim=2,index=idx.clamp(max=opt.nerf.sample_intvs)) # [B,HW,Nf]
        # linear interpolation
        t = (unif-cdf_low)/(cdf_high-cdf_low+1e-8) # [B,HW,Nf]
        depth_samples = depth_low+t*(depth_high-depth_low) # [B,HW,Nf]
        return depth_samples[...,None] # [B,HW,Nf,1]

class NeRF(torch.nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.define_network(opt)

    def define_network(self,opt):
        input_3D_dim = 3+6*opt.arch.posenc.L_3D if opt.arch.posenc else 3
        if opt.nerf.view_dep:
            input_view_dim = 3+6*opt.arch.posenc.L_view if opt.arch.posenc else 3
        # point-wise feature
        self.mlp_feat = torch.nn.ModuleList()
        L = util.get_layer_dims(opt.arch.layers_feat)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = input_3D_dim
            if li in opt.arch.skip: k_in += input_3D_dim
            if li==len(L)-1: k_out += 1
            linear = torch.nn.Linear(k_in,k_out)
            if opt.arch.tf_init:
                self.tensorflow_init_weights(opt,linear,out="first" if li==len(L)-1 else None)
            self.mlp_feat.append(linear)
        # RGB prediction
        self.mlp_rgb = torch.nn.ModuleList()
        L = util.get_layer_dims(opt.arch.layers_rgb)
        feat_dim = opt.arch.layers_feat[-1]
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = feat_dim+(input_view_dim if opt.nerf.view_dep else 0)
            linear = torch.nn.Linear(k_in,k_out)
            if opt.arch.tf_init:
                self.tensorflow_init_weights(opt,linear,out="all" if li==len(L)-1 else None)
            self.mlp_rgb.append(linear)

    def tensorflow_init_weights(self,opt,linear,out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out=="first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:],gain=relu_gain)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self,opt,points_3D,ray_unit=None,mode=None): # [B,...,3]
        if opt.arch.posenc:
            points_enc = self.positional_encoding(opt,points_3D,L=opt.arch.posenc.L_3D)
            points_enc = torch.cat([points_3D,points_enc],dim=-1) # [B,...,6L+3]
        else: points_enc = points_3D
        feat = points_enc
        # extract coordinate-based features
        for li,layer in enumerate(self.mlp_feat):
            if li in opt.arch.skip: feat = torch.cat([feat,points_enc],dim=-1)
            feat = layer(feat)
            if li==len(self.mlp_feat)-1:
                density = feat[...,0]
                if opt.nerf.density_noise_reg and mode=="train":
                    density += torch.randn_like(density)*opt.nerf.density_noise_reg
                density_activ = getattr(torch_F,opt.arch.density_activ) # relu_,abs_,sigmoid_,exp_....
                density = density_activ(density)
                feat = feat[...,1:]
            feat = torch_F.relu(feat)
        # predict RGB values
        if opt.nerf.view_dep:
            assert(ray_unit is not None)
            if opt.arch.posenc:
                ray_enc = self.positional_encoding(opt,ray_unit,L=opt.arch.posenc.L_view)
                ray_enc = torch.cat([ray_unit,ray_enc],dim=-1) # [B,...,6L+3]
            else: ray_enc = ray_unit
            feat = torch.cat([feat,ray_enc],dim=-1)
        for li,layer in enumerate(self.mlp_rgb):
            feat = layer(feat)
            if li!=len(self.mlp_rgb)-1:
                feat = torch_F.relu(feat)
        rgb = feat.sigmoid_() # [B,...,3]
        return rgb,density

    def forward_samples(self,opt,center,ray,depth_samples,mode=None):
        points_3D_samples = camera.get_3D_points_from_depth(opt,center,ray,depth_samples,multi_samples=True) # [B,HW,N,3]
        if opt.nerf.view_dep:
            ray_unit = torch_F.normalize(ray,dim=-1) # [B,HW,3]
            ray_unit_samples = ray_unit[...,None,:].expand_as(points_3D_samples) # [B,HW,N,3]
        else: ray_unit_samples = None
        rgb_samples,density_samples = self.forward(opt,points_3D_samples,ray_unit=ray_unit_samples,mode=mode) # [B,HW,N],[B,HW,N,3]
        return rgb_samples,density_samples

    def composite(self,opt,ray,rgb_samples,density_samples,depth_samples):
        ray_length = ray.norm(dim=-1,keepdim=True) # [B,HW,1]
        # volume rendering: compute probability (using quadrature)
        depth_intv_samples = depth_samples[...,1:,0]-depth_samples[...,:-1,0] # [B,HW,N-1]
        depth_intv_samples = torch.cat([depth_intv_samples,torch.empty_like(depth_intv_samples[...,:1]).fill_(1e10)],dim=2) # [B,HW,N]
        dist_samples = depth_intv_samples*ray_length # [B,HW,N]
        sigma_delta = density_samples*dist_samples # [B,HW,N]
        alpha = 1-(-sigma_delta).exp_() # [B,HW,N]
        T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]],dim=2).cumsum(dim=2)).exp_() # [B,HW,N]
        prob = (T*alpha)[...,None] # [B,HW,N,1]
        # integrate RGB and depth weighted by probability
        depth = (depth_samples*prob).sum(dim=2) # [B,HW,1]
        rgb = (rgb_samples*prob).sum(dim=2) # [B,HW,3]
        opacity = prob.sum(dim=2) # [B,HW,1]
        if opt.nerf.setbg_opaque:
            rgb = rgb+opt.data.bgcolor*(1-opacity)
        return rgb,depth,opacity,prob # [B,HW,K]

    def positional_encoding(self,opt,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=opt.device)*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc
