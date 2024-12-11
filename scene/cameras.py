#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import random
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, getProjectionMatrix2, focal2fov

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, prcppoint,
                 image=torch.zeros([1, 1, 1]), 
                 gt_alpha_mask=None, image_name=None, uid=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 mono=None):
        super(Camera, self).__init__()

        self.uid = uid # idx in current set (train/test)
        self.colmap_id = colmap_id # global id
        # rotation of c2w
        self.R = R
        # translation vec. in w2c mat.
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device) #CHW
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # if gt_alpha_mask is not None:
        #     self.original_image *= gt_alpha_mask.to(self.data_device)
        # else:
        #     self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.prcppoint = torch.tensor(prcppoint).to(torch.float32).to(self.data_device)
        
        # left rotation because batch/n_points dim. is on the left side
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        ### from eye/view/cam coord. to clip coord.
        self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, FoVx, FoVy, self.image_width, self.image_height, prcppoint).transpose(0,1).to(self.data_device)
        ### from world coord. to clip coord.
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if gt_alpha_mask is None:
            self.mask = None
        elif type(gt_alpha_mask) == np.ndarray:
            self.mask = torch.from_numpy(gt_alpha_mask).to(self.data_device)
        else: # torch tensor
            self.mask = gt_alpha_mask.to(self.data_device)
        # binary mask to avoid floating problem when multiplied with depth in render.py
        self.mask = (self.mask > 0.5).float()
        self.mono = None if mono is None else torch.from_numpy(mono).to(self.data_device)
       
    def get_gtMask(self, with_mask=True):
        if self.mask is None or not with_mask:
            self.mask = torch.ones_like(self.original_image[:1], device="cuda")
        return self.mask#.to(torch.bool)

    def get_gtImage(self, bg, with_mask=True, mask_overwrite=None):
        if self.mask is None or not with_mask:
            return self.original_image
        mask = self.get_gtMask(with_mask) if mask_overwrite is None else mask_overwrite
        return self.original_image * mask + bg[:, None, None] * (1 - mask)
    
    def random_patch(self, h_size=float('inf'), w_size=float('inf')):
        h = self.image_height
        w = self.image_width
        h_size = min(h_size, h)
        w_size = min(w_size, w)
        h0 = random.randint(0, h - h_size)
        w0 = random.randint(0, w - w_size)
        h1 = h0 + h_size
        w1 = w0 + w_size
        return torch.tensor([h0, w0, h1, w1]).to(torch.float32).to(self.data_device)
    
    # only used for rendering mesh video
    def update_intrinsics(self, scale_f_len=None, rendered_h=None, rendered_w=None, d_cy=None):
        self.fx = fov2focal(self.FoVx, self.image_width)
        self.fy = fov2focal(self.FoVy, self.image_height)
        # breakpoint()
        if scale_f_len:
            self.fx *= scale_f_len
            self.fy *= scale_f_len
        if rendered_h:
            self.image_width = rendered_w
            self.image_height = rendered_h
            # self.prcppoint = torch.tensor(prcppoint).to(torch.float32).to(self.data_device)
            self.prcppoint[0] = 1 / 2
            self.prcppoint[1] = 1 / 2
        if d_cy:
            self.prcppoint[1] += d_cy / rendered_h
        self.FoVx, self.FoVy = focal2fov(self.fx, self.image_width), focal2fov(self.fy, self.image_height)
        
        # left rotation because batch/n_points dim. is on the left side
        # self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        ### from eye/view/cam coord. to clip coord.
        self.projection_matrix = getProjectionMatrix2(self.znear, self.zfar, self.fx, self.fy, self.image_width, self.image_height, self.prcppoint).transpose(0,1).to(self.data_device)
        ### from world coord. to clip coord.
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

