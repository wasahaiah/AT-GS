import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import sys
sys.path.append('submodules')

from RAFT.raft import RAFT
from RAFT.utils import flow_viz

DEVICE = "cuda"

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def warp_flow(img, flow):
    n, _, h, w = flow.shape
    flow_new = flow.clone()

    # Create a mesh grid for the pixel coordinates
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
    grid_x = grid_x.to(flow.device).float()
    grid_y = grid_y.to(flow.device).float()

    for i in range(n):
        flow_new[i, 0, :, :] += grid_x
        flow_new[i, 1, :, :] += grid_y

    # Normalize the coordinates to be in the range [-1, 1]
    flow_new[:, 0, :, :] = 2.0 * flow_new[:, 0, :, :] / (w - 1) - 1.0
    flow_new[:, 1, :, :] = 2.0 * flow_new[:, 1, :, :] / (h - 1) - 1.0

    flow_new = flow_new.permute(0, 2, 3, 1)  # Reshape for grid_sample
    img = img.unsqueeze(0) if len(img.shape) == 3 else img  # Add batch dimension if necessary

    res = F.grid_sample(img, flow_new, mode='bilinear', padding_mode='zeros', align_corners=True)
    return res

def compute_fwdbwd_mask(fwd_flow, bwd_flow):
    alpha_1 = 0.5
    alpha_2 = 0.5

    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)  # bw flow in img1
    fwd_lr_error = torch.norm(fwd_flow + bwd2fwd_flow, dim=1)  # cycle error of img1->img2->img1
    fwd_mask = fwd_lr_error < alpha_1 * (torch.norm(fwd_flow, dim=1) + torch.norm(bwd2fwd_flow, dim=1)) + alpha_2  # mask of img1

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = torch.norm(bwd_flow + fwd2bwd_flow, dim=1)

    bwd_mask = bwd_lr_error < alpha_1 * (torch.norm(bwd_flow, dim=1) + torch.norm(fwd2bwd_flow, dim=1)) + alpha_2

    return fwd_mask, bwd_mask

def warp_image_using_bwd_flow(image1, flow_bwd):
    """
    Warp image1 to generate an image that aligns with image2 using the backward optical flow.

    Parameters:
    - image1 (torch.Tensor): The source image in BCHW format.
    - flow_bwd (torch.Tensor): The backward optical flow tensor with shape (B, 2, H, W).

    Returns:
    - warped_image (torch.Tensor): The image generated from image1 using the optical flow.
    """
    n, _, h, w = flow_bwd.shape

    # Create coordinate grid for the original image
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
    grid_x = grid_x.to(flow_bwd.device).float()
    grid_y = grid_y.to(flow_bwd.device).float()

    new_x = flow_bwd[:, 0, :, :] + grid_x
    new_y = flow_bwd[:, 1, :, :] + grid_y

    # Normalize the coordinates to be in the range [-1, 1]
    new_x = 2.0 * new_x / (w - 1) - 1.0
    new_y = 2.0 * new_y / (h - 1) - 1.0

    # Stack coordinates into a single array
    flow_map = torch.stack((new_x, new_y), dim=-1)  # Shape (B, H, W, 2)

    # Remap image1 to generate the warped image
    warped_image = F.grid_sample(image1, flow_map, mode='nearest', padding_mode='reflection', align_corners=True)
    # warped_image = F.grid_sample(image1, flow_map, mode='bilinear', padding_mode='reflection', align_corners=True)

    return warped_image

def append_mask_to_image(warped_image, mask):
    """
    Append the mask to warped_image to create an RGBA image.

    Parameters:
    - warped_image (torch.Tensor): The warped image in BCHW format with shape (B, C, H, W).
    - mask (torch.Tensor): The mask tensor with shape (B, H, W).

    Returns:
    - rgba_image (torch.Tensor): The RGBA image combining the warped image and mask.
    """
    # Check if warped_image and mask have compatible shapes
    if warped_image.shape[2:] != mask.shape[1:]:
        raise ValueError("The dimensions of warped_image and mask must match.")

    # Convert mask to uint8 format and scale to [0, 255]
    mask_uint8 = (mask * 255).byte()

    # Add a new channel dimension to the mask
    mask_uint8 = mask_uint8.unsqueeze(1)  # Shape (B, 1, H, W)

    # Combine the BGR image with the mask channel
    rgba_image = torch.cat((warped_image, mask_uint8), dim=1)  # Shape (B, C+1, H, W)

    return rgba_image

class C_RAFT:
    def __init__(self, model_path='models/raft-things.pth'):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default=model_path, help="restore RAFT checkpoint")
        parser.add_argument("--small", action="store_true", help="use small model")
        parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
        args = parser.parse_args([])
        
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(args.model))
        self.model = self.model.module
        self.model.to(DEVICE)
        self.model.eval()
        self.save_vis_idx = 0

    def raft_warp(self, image1_batch, image2_batch, normals1_batch, save_vis=False):
        with torch.no_grad():
            # image1/2 are BCHW in cuda
            image1_batch = image1_batch.to(DEVICE)*255 # BCHW
            image2_batch = image2_batch.to(DEVICE)*255 # BCHW

            padder = InputPadder(image1_batch.shape)
            image1_batch, image2_batch = padder.pad(image1_batch, image2_batch)

            _, flow_fwd_batch = self.model(image1_batch, image2_batch, iters=20, test_mode=True)
            _, flow_bwd_batch = self.model(image2_batch, image1_batch, iters=20, test_mode=True)

            flow_fwd_batch = padder.unpad(flow_fwd_batch)
            flow_bwd_batch = padder.unpad(flow_bwd_batch) # B2HW
            image1_batch = padder.unpad(image1_batch)
            image2_batch = padder.unpad(image2_batch)
            
            mask_fwd_batch, mask_bwd_batch = compute_fwdbwd_mask(flow_fwd_batch, flow_bwd_batch)

            image2_warpped_batch = warp_image_using_bwd_flow(image1_batch, flow_bwd_batch)
            image2_warpped_batch = append_mask_to_image(image2_warpped_batch, mask_bwd_batch)

            normals2_warpped_batch = warp_image_using_bwd_flow(normals1_batch, flow_bwd_batch) # B3HW

            mask_bwd_batch = mask_bwd_batch.unsqueeze(1).float()  # Add channel dimension, shape becomes Bx1xHxW
            normals2_warpped_batch = torch.cat((normals2_warpped_batch, mask_bwd_batch), dim=1)  # Shape becomes Bx4xHxW

            if save_vis:
                # os.makedirs('./flow_vis', exist_ok=True)
                # for i in range(flow_bwd_batch.shape[0]):
                #     masked_flow_bwd_batch = flow_bwd_batch * mask_bwd_batch.unsqueeze(1).float()
                #     Image.fromarray(flow_viz.flow_to_image(masked_flow_bwd_batch[i].cpu().numpy().transpose(1, 2, 0))).save(f"./flow_vis/{self.save_vis_idx}.png")
                #     self.save_vis_idx += 1
                return normals2_warpped_batch, flow_bwd_batch

            return normals2_warpped_batch
