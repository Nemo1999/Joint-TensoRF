# Large portion of the code is copied from barf --> model/barf.py
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
import matplotlib.pyplot as plt

import util,util_vis
from util import log,debug
from . import tensorf
import camera
import wandb

from util import interp_schedule


# ============================ main engine for training and evaluation ============================

class Model(tensorf.Model):

    def __init__(self,opt):
        super().__init__(opt)

    def build_networks(self,opt):
        super().build_networks(opt)
        if opt.camera.noise:
            # pre-generate synthetic pose perturbation
            se3_noise = torch.randn(len(self.train_data),6 ,device=opt.device)* (opt.camera.noise)
            se3_noise = se3_noise.to(torch.float32)
            self.graph.pose_noise = torch.nn.parameter.Parameter(data=camera.lie.se3_to_SE3(se3_noise), requires_grad=False)
        self.graph.se3_refine = torch.nn.Embedding(len(self.train_data),6).to(opt.device)
        torch.nn.init.zeros_(self.graph.se3_refine.weight)
        if hasattr(opt, "start_with_GT_pose") and opt.data.dataset in ["llff", "t2"]:
            pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device, dtype=torch.float32)
            pose_se3 = camera.lie.SE3_to_se3(pose_GT)
            with torch.no_grad():
                #self.graph.se3_refine_weight *= 0
                print(f"resetting se3_refine weight to GT poses")
                print(f"Pose Embedding: {self.graph.se3_refine.weight.shape=}")
                print(f"GT Pose: {pose_se3.shape=}")
                self.graph.se3_refine.weight[...] = pose_se3

    def setup_optimizer(self,opt):
        super().setup_optimizer(opt)
        optimizer = getattr(torch.optim,opt.optim.pose_algo)
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
        optim_params = dict()
        if hasattr(opt.optim, "pose_algo_params"):
            optim_params = opt.optim.pose_algo_params
        self.optim_pose = optimizer([dict(params=self.graph.se3_refine.parameters(),lr=lr_init)], **optim_params)
        # set up scheduler
        if opt.optim.sched_pose:
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched_pose.type)
            if opt.optim.sched_pose.type=="ExponentialLR":
                assert(hasattr(opt.optim, "lr_pose_end"))
                gamma = (opt.optim.lr_pose_end/opt.optim.lr_pose)**(1./opt.max_iter)
                kwargs = {"gamma": gamma }
            else:
                assert (opt.optim.sched_pose.type == "LambdaLR")
                lr_lambda = lambda it: util.interp_schedule(float(it/opt.max_iter), opt.optim.sched_pose.lr_pose_schedule)
                kwargs = {"lr_lambda": lr_lambda}
            self.sched_pose = scheduler(self.optim_pose,**kwargs)
    @torch.no_grad()
    def interrupt_pose(self, opt):
        with torch.no_grad():
            self.graph.se3_refine.weight *= 0.0
            print("reset pose se3 weights !!!!!!!!!!!!!!!!!!!")
    def freeze_poses(self,opt):
        self.graph.se3_refine.requires_grad = False

    def unfreeze_poses(self, opt):
        self.graph.se3_refine.requires_grad = True

    def freeze_scene(self, opt):
        self.graph.nerf.freeze_scene(opt)

    def unfreeze_scene(self, opt):
        self.graph.nerf.unfreeze_scene(opt)

    def train_iteration(self,opt,var,loader):
        # simple linear warmup of pose learning rate
        if opt.optim.warmup_pose:
            self.optim_pose.param_groups[0]["lr_orig"] = self.optim_pose.param_groups[0]["lr"] # cache the original learning rate
            self.optim_pose.param_groups[0]["lr"] *= min(1,self.it/opt.optim.warmup_pose)
        loss = super().train_iteration(opt,var,loader)

        if (not hasattr(opt.optim,"pose_grad_accum_iter") ) or (self.it % opt.optim.pose_grad_accum_iter) == 0:
            # we don't normalized gradient by scaling down loss here because the tensorf volume didn't accumulate gradient, so we shrink our learning rate instead ( see setup_optimizer function)
            self.optim_pose.step()
            self.optim_pose.zero_grad()

        # simple linear warmup of pose learning rate
        if opt.optim.warmup_pose:
            self.optim_pose.param_groups[0]["lr"] = self.optim_pose.param_groups[0]["lr_orig"] # reset learning rate


        if opt.optim.sched_pose: self.sched_pose.step()
        self.graph.nerf.progress.data.fill_(self.it/opt.max_iter)

        return loss

    @torch.no_grad()
    def validate(self,opt,ep=None):
        pose,pose_GT = self.get_all_training_poses(opt)
        _, self.graph.sim3 = self.prealign_cameras(opt,pose,pose_GT)
        super().validate(opt,ep=ep)

    @torch.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train"):
        super().log_scalars(opt,var,loss,metric=metric,step=step,split=split)
        if split=="train":
            # log learning rate
            lr = self.optim_pose.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split,"lr_pose"),lr,step)
            wandb.log({f"{split}.{'lr_pose'}": lr}, step=step)
        # compute pose error
        if split=="train" and opt.data.dataset in ["blender","llff", "t2"]:
            pose,pose_GT = self.get_all_training_poses(opt)
            pose_aligned, _ = self.prealign_cameras(opt,pose,pose_GT)
            error = self.evaluate_camera_alignment(opt,pose_aligned,pose_GT)
            self.tb.add_scalar("{0}/error_R".format(split),error.R.mean(),step)
            self.tb.add_scalar("{0}/error_t".format(split),error.t.mean(),step)
            wandb.log({f"{split}.{'error_R'}": error.R.mean()}, step=step)
            wandb.log({f"{split}.{'error_t'}": error.t.mean()}, step=step)

        # log scattar plot of per-view PSNR-PoseError

        if opt.nerf.ray_sampling_strategy != "single_view_rand_rays":
            return
        if not hasattr(self,"train_view_mse"):
            self.train_view_mse= []
            self.train_view_error_R = []
            self.train_view_error_t = []

        assert opt.freq.scalar==1 and opt.nerf.ray_sampling_strategy=="single_view_rand_rays", "need make sure that these assertion are satisfied for meaningful scatter plotting per-view-PSNR vs PoseError"
        scatter_path = os.path.join(opt.output_path, "mse_poseerror_scatters")
        os.makedirs(scatter_path, exist_ok=True)
        if self.it == 4000:
            #pdb.set_trace(header="check poise_noise and pose_refinement")
            pass
        if split == "train":
            if self.it in [0,1]:
                self.train_view_mse= []
                self.train_view_error_R = []
                self.train_view_error_t = []

            elif self.it % (self.n_train_views*5) == 0:
                # log scalar plot
                fig = plt.figure(figsize=(16,8))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.set_title(f"train MSE-error_R, iters={self.it}")
                ax2.set_title(f"train MSE-error_t, iters={self.it}")
                ax1.scatter(self.train_view_mse, self.train_view_error_R)
                ax2.scatter(self.train_view_mse, self.train_view_error_t)
                ax1.set_xlabel("mse")
                ax1.set_ylabel("error_R")
                ax2.set_xlabel("mse")
                ax2.set_ylabel("error_t")
                fig_path = os.path.join(scatter_path, f"iters_{self.it}.png")
                plt.savefig(fig_path , dpi=75)
                plt.clf()
                plt.close(fig)
                wandb.log({"train.MSE-train.PoseError plots": wandb.Image(fig_path)}, step=step)
            if self.it % self.n_train_views == 0:
                # reset accumulator
                self.train_view_mse= []
                self.train_view_error_R = []
                self.train_view_error_t = []
            self.train_view_mse.append(loss.render.cpu().numpy())
            self.train_view_error_R.append(error.R[self.view_index].cpu().numpy())
            self.train_view_error_t.append(error.t[self.view_index].cpu().numpy())
    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train"):
        super().visualize(opt,var,step=step,split=split)
        if opt.visdom:
            if split=="val":
                pose,pose_GT = self.get_all_training_poses(opt)
                util_vis.vis_cameras(opt,self.vis,step=step,poses=[pose,pose_GT])
    @torch.no_grad()
    def get_all_training_poses(self,opt):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device, dtype=torch.float32)
        # add synthetic pose perturbation to all training data
        if opt.data.dataset=="blender":
            pose = pose_GT
            if opt.camera.noise:
                pose = camera.pose.compose([self.graph.pose_noise,pose])
        else: pose = self.graph.pose_eye
        # add learned pose correction to all training data
        pose_refine = camera.lie.se3_to_SE3(self.graph.se3_refine.weight)
        pose = camera.pose.compose([pose_refine,pose])
        return pose,pose_GT

    @torch.no_grad()
    def prealign_cameras(self,opt,pose,pose_GT):
        # compute 3D similarity transform via Procrustes analysis
        center = torch.zeros(1,1,3,device=opt.device,dtype=torch.float32)
        center_pred = camera.cam2world(center,pose)[:,0] # [N,3]
        center_GT = camera.cam2world(center,pose_GT)[:,0] # [N,3]
        try:
            sim3 = camera.procrustes_analysis(center_GT,center_pred)
        except:
            print("warning: SVD did not converge...")
            sim3 = edict(t0=0,t1=0,s0=1,s1=1,R=torch.eye(3,device=opt.device,dtype=torch.float32))
            # align the camera poses (predicted)"pose" with "pose_GT"
        center_aligned = (center_pred-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
        R_aligned = pose[...,:3]@sim3.R.t()
        t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
        pose_aligned = camera.pose(R=R_aligned,t=t_aligned)
        return pose_aligned,sim3

    @torch.no_grad()
    def evaluate_camera_alignment(self,opt,pose_aligned,pose_GT):
        # measure errors in rotation and translation
        R_aligned,t_aligned = pose_aligned.split([3,1],dim=-1)
        R_GT,t_GT = pose_GT.split([3,1],dim=-1)
        R_error = camera.rotation_distance(R_aligned,R_GT)
        t_error = (t_aligned-t_GT)[...,0].norm(dim=-1)
        error = edict(R=R_error,t=t_error)
        return error

    @torch.no_grad()
    def evaluate_full(self,opt):
        self.graph.eval()
        # evaluate rotation/translation
        pose,pose_GT = self.get_all_training_poses(opt)
        pose_aligned,self.graph.sim3 = self.prealign_cameras(opt,pose,pose_GT)
        #pdb.set_trace(header="Check evaluate pose")
        error = self.evaluate_camera_alignment(opt,pose_aligned,pose_GT)
        print("--------------------------")
        print("rot:   {:8.3f}".format(np.rad2deg(error.R.mean().cpu())))
        print("trans: {:10.5f}".format(error.t.mean().cpu()))
        print("--------------------------")

        wandb.run.summary["TEST_Rotation_Error(deg)"] = np.rad2deg(error.R.mean().cpu())
        wandb.run.summary["TEST_Translation_Error(cannon_coord)"] = (error.t.mean().cpu())

        # dump numbers
        quant_fname = "{}/quant_pose.txt".format(opt.output_path)
        with open(quant_fname,"w") as file:
            for i,(err_R,err_t) in enumerate(zip(error.R,error.t)):
                file.write("{} {} {}\n".format(i,err_R.item(),err_t.item()))
        # evaluate novel view synthesis

        self.visualize_pose(opt, opt.max_iter+1, split="test")
        super().evaluate_full(opt)

    @torch.enable_grad()
    def evaluate_test_time_photometric_optim(self,opt,var):
        # use another se3 Parameter to absorb the remaining pose errors
        var.se3_refine_test = torch.nn.Parameter(torch.zeros(1,6,device=opt.device))
        optimizer = torch.optim.Adam
        optim_pose = optimizer([dict(params=[var.se3_refine_test],lr=opt.optim.lr_pose)])
        gamma = (opt.optim.lr_pose_test_end/opt.optim.lr_pose_test)**(1./opt.optim.test_iter)
        sched_pose = torch.optim.lr_scheduler.ExponentialLR(optim_pose, gamma=gamma)


        if opt.data.dataset == "llff":
            iterator = tqdm.trange(opt.optim.test_iter,desc="test-time optim.",leave=False,position=1)
        else:
            # remove tqdm progress bar for better log file
            iterator = range(opt.optim.test_iter)
        print("evaluting test_time photometric loss. . .")
        for it in iterator:
            self.graph.nerf.test_time_progress.data.fill_(it / opt.optim.test_iter)
            optim_pose.zero_grad()
            var.pose_refine_test = camera.lie.se3_to_SE3(var.se3_refine_test)
            var = self.graph.forward(opt,var,mode="test-optim")
            loss = self.graph.compute_loss(opt,var,mode="test-optim")
            loss = self.summarize_loss(opt,var,loss)
            loss.all.backward()
            optim_pose.step()
            sched_pose.step()
            #iterator.set_postfix(loss="{:.3f}".format(loss.all))
        return var

    @torch.no_grad()
    def generate_videos_pose(self,opt):
        self.graph.eval()
        fig = plt.figure(figsize=(10,10) if opt.data.dataset in["blender","t2"] else (16,8))
        cam_path = "{}/poses".format(opt.output_path)
        os.makedirs(cam_path,exist_ok=True)
        ep_list = []
        for ep in range(0,opt.max_iter+1,opt.freq.ckpt):
            # load checkpoint (0 is random init)
            if ep!=0:
                try: util.restore_checkpoint(opt,self,resume=ep)
                except: continue
            # get the camera poses
            pose,pose_ref = self.get_all_training_poses(opt)
            if opt.data.dataset in ["blender","llff", "t2"]:
                pose_aligned,_ = self.prealign_cameras(opt,pose,pose_ref)
                pose_aligned,pose_ref = pose_aligned.detach().cpu(),pose_ref.detach().cpu()
                dict(
                    blender=util_vis.plot_save_poses_blender,
                    llff=util_vis.plot_save_poses,
                    t2=util_vis.plot_save_poses_t2
                )[opt.data.dataset](opt,fig,pose_aligned,pose_ref=pose_ref,path=cam_path,ep=ep)
            else:
                pose = pose.detach().cpu()
                util_vis.plot_save_poses(opt,fig,pose,pose_ref=None,path=cam_path,ep=ep)
            ep_list.append(ep)
        plt.close()
        # write videos
        print("writing videos...")
        list_fname = "{}/temp.list".format(cam_path)
        with open(list_fname,"w") as file:
            for ep in ep_list: file.write("file {}.png\n".format(ep))
        cam_vid_fname = "{}/poses.mp4".format(opt.output_path)
        os.system("ffmpeg -y -r 2 -f concat -i {0} -pix_fmt yuv420p {1} >/dev/null 2>&1".format(list_fname, cam_vid_fname))
        wandb.log({"campose_vid": wandb.Video(cam_vid_fname)})
        os.remove(list_fname)

# ============================ computation graph for forward/backprop ============================

class Graph(tensorf.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        self.nerf = NeRF(opt)
        setattr(self.nerf, "get_parent", lambda : self)
        self.pose_eye = torch.eye(3,4).to(opt.device)

    def get_pose(self,opt,var,mode=None):
        if mode=="train":
            # add the pre-generated pose perturbations
            if opt.data.dataset=="blender":
                if opt.camera.noise:
                    var.pose_noise = self.pose_noise[var.idx]
                    pose = camera.pose.compose([var.pose_noise,var.pose]) # apply pose_noise on GT_pose (i.e. var.pose)
                else: pose = var.pose
            else: pose = self.pose_eye
            # add learnable pose correction
            var.se3_refine = self.se3_refine.weight[var.idx]
            pose_refine = camera.lie.se3_to_SE3(var.se3_refine)
            pose = camera.pose.compose([pose_refine,pose])
        elif mode in ["val","eval","test-optim"]:
            # align test pose to refined coordinate system (up to sim3)
            sim3 = self.sim3
            center = torch.zeros(1,1,3,device=opt.device, dtype=torch.float32)
            center = camera.cam2world(center,var.pose)[:,0] # [N,3]  # origin of each camear coordinate
            center_aligned = (center-sim3.t0)/sim3.s0@sim3.R*sim3.s1+sim3.t1  # transform using self.sim(generated from bat.Model.evaluate_camera_alignment)
            R_aligned = var.pose[...,:3]@self.sim3.R
            t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
            pose = camera.pose(R=R_aligned,t=t_aligned)
            # additionally factorize the remaining pose imperfection
            if opt.optim.test_photo and mode!="val":
                pose = camera.pose.compose([var.pose_refine_test,pose])
        else: pose = var.pose
        return pose

class NeRF(tensorf.NeRF):

    def __init__(self,opt):
        super().__init__(opt)
        self.progress = torch.nn.Parameter(torch.tensor(0.)) # use Parameter so it could be checkpointed
        self.test_time_progress = torch.nn.Parameter(torch.tensor(0.))
    def update_schedule(self, opt, it):
        super().update_schedule(opt, it)

    def freeze_scene(self, opt):
        self.tensorf.freeze_scene(opt)

    def unfreeze_scene(self,opt):
        self.tensorf.unfreeze_scene(opt)
