import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import imageio
from easydict import EasyDict as edict
import json
import pickle
from . import base
import camera
from util import log,debug


# Tanks and Temple Dataset
class Dataset(base.Dataset):

    def __init__(self,opt,split="train",subset=None):
        self.raw_H,self.raw_W = 1080, 1920
        super().__init__(opt,split)
        self.root = opt.data.root or "data/t2"
        self.path = "{}/{}".format(self.root,opt.data.scene)
        self.path_image = "{}/images".format(self.path)
        self.focal = self.raw_W * 0.7
        image_fnames = sorted(os.listdir(self.path_image))
        poses_raw = self.parse_cameras_and_bounds(opt)
        self.list = list(zip(image_fnames,poses_raw))
        # manually split train/val subsets
        num_val_split = int(len(self)*opt.data.val_ratio)
        # val and test are same split here in llff
        self.list = self.list[:-num_val_split] if split=="train" else self.list[-num_val_split:]
        if subset: self.list = self.list[:subset]
        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt,self.get_image)
            self.cameras = self.preload_threading(opt,self.get_camera,data_str="cameras")

    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def parse_cameras_and_bounds(self,opt):
        fname = "{}/Ignatius_COLMAP_SfM.log".format(self.path)
        with open(fname,"r") as f:
            content = f.readlines()
            content = [line.strip() for line in content]
        assert len(content) % 5 == 0
        i = 0

        poses_raw = []
        while True:
            image_id = int(content[i].split(' ')[0])
            i += 1
            pose = []
            for _ in range(3):
                pose.append(list(map(float, content[i].split(' '))))
                i += 1
            i+=1
            poses_raw.append(pose)
            if i >= len(content):
                break
        poses_raw = torch.tensor(poses_raw)
        eye = torch.eye(3)
        eye[0,0] *= -1
        eye[1,1] *= -1
        pose_rotate_backward = camera.pose(R=eye)
        #poses_raw[...,0],poses_raw[...,1] = poses_raw[...,1],-poses_raw[...,0]
        poses_raw = camera.pose.compose_pair(poses_raw, pose_rotate_backward)
        print(poses_raw.shape)
        # roughly center camera poses
        poses_raw = self.center_camera_poses(opt,poses_raw)
        return poses_raw

    # norm of v0 is not necessarily 1, even though norms of v1 and v2 are 1
    # so det(pose_avg) is not necessarily 1
    # which means that it is not SO(3)
    # please see https://github.com/bmild/nerf/issues/34
    # also refer to https://github.com/kwea123/nerf_pl/blob/master/datasets/llff.py

    # original implementation
    # def center_camera_poses(self,opt,poses):
    #     # compute average pose
    #     center = poses[...,3].mean(dim=0)
    #     v1 = torch_F.normalize(poses[...,1].mean(dim=0),dim=0)
    #     v2 = torch_F.normalize(poses[...,2].mean(dim=0),dim=0)
    #     v0 = v1.cross(v2)
    #     pose_avg = torch.stack([v0,v1,v2,center],dim=-1)[None] # [1,3,4]
    #     # apply inverse of averaged pose
    #     poses = camera.pose.compose([poses,camera.pose.invert(pose_avg)])
    #     return poses

    # new implementation
    def center_camera_poses(self,opt,poses):
        # compute average pose
        center = poses[...,3].mean(dim=0)
        v1 = torch_F.normalize(poses[...,1].mean(dim=0),dim=0)
        v2 = torch_F.normalize(poses[...,2].mean(dim=0),dim=0)
        v0 = v1.cross(v2)
        pose_avg = torch.stack([v0,v1,v2,center],dim=-1)[None] # [1,3,4]
        vz = torch_F.normalize(poses[...,2].mean(dim=0),dim=0)
        vy_hat = poses[..., 1].mean(dim=0)
        # in the original implementation, v0
        vx = torch_F.normalize(vy_hat.cross(vz), dim=0)
        vy = vz.cross(vx)
        pose_avg = torch.stack([vx,vy,vz,center],dim=-1)[None] # [1,3,4]
        # apply inverse of averaged pose
        poses = camera.pose.compose([poses,camera.pose.invert(pose_avg)])
        return poses

    def get_all_camera_poses(self,opt):
        if not opt.data.preload and not hasattr(self,"all"):
            pose_raw_all = [tup[1] for tup in self.list]
            pose_all = torch.stack([self.parse_raw_camera(opt,p) for p in pose_raw_all],dim=0)
            return pose_all
        else:
            return self.all["pose"]

    def __getitem__(self,idx):
        opt = self.opt
        sample = dict(idx=idx)
        aug = self.generate_augmentation(opt) if self.augment else None
        image = self.images[idx] if opt.data.preload else self.get_image(opt,idx)
        image = self.preprocess_image(opt,image,aug=aug)
        intr,pose = self.cameras[idx] if opt.data.preload else self.get_camera(opt,idx)
        intr_inv ,pose, intr = self.preprocess_camera(opt,intr,pose,aug=aug)
        sample.update(
            image=image,
            intr_inv=intr_inv,
            intr=intr,
            pose=pose,

        )
        return sample

    def get_image(self,opt,idx):
        image_fname = "{}/{}".format(self.path_image,self.list[idx][0])
        image = PIL.Image.fromarray(imageio.imread(image_fname)) # directly using PIL.Image.open() leads to weird corruption....
        return image

    def get_camera(self,opt,idx):
        intr = torch.tensor([[self.focal,0,self.raw_W/2],
                             [0,self.focal,self.raw_H/2],
                             [0,0,1]]).float()
        pose_raw = self.list[idx][1]
        pose = self.parse_raw_camera(opt,pose_raw)
        return intr,pose

    def parse_raw_camera(self,opt,pose_raw):
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
        pose = camera.pose.compose([pose_flip,pose_raw[:3]])
        pose = camera.pose.invert(pose)
        pose = camera.pose.compose([pose_flip,pose])
        return pose
