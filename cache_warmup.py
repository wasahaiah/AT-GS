
import sys
import os
import tinycudann as tcnn
import torch
import json5
import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from argparse import ArgumentParser, Namespace

project_directory = '.'
sys.path.append(os.path.abspath(project_directory))
from ntc import NeuralTransformationCache


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def fetchXYZ(path):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    return torch.tensor(xyz, dtype=torch.float, device="cuda")

def get_xyz_bound():
    ## Hard-code the coordinate of the corners here!!
    return torch.tensor(aabb[0]).cuda(), torch.tensor(aabb[1]).cuda()

def get_contracted_xyz(xyz):
    xyz_bound_min, xyz_bound_max = get_xyz_bound()
    normalzied_xyz=(xyz-xyz_bound_min)/(xyz_bound_max-xyz_bound_min)
    return normalzied_xyz

@torch.compile
def quaternion_multiply(a, b):
    a_norm=nn.functional.normalize(a)
    b_norm=nn.functional.normalize(b)
    w1, x1, y1, z1 = a_norm[:, 0], a_norm[:, 1], a_norm[:, 2], a_norm[:, 3]
    w2, x2, y2, z2 = b_norm[:, 0], b_norm[:, 1], b_norm[:, 2], b_norm[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack([w, x, y, z], dim=1)

def quaternion_loss(q1, q2):
    cos_theta = F.cosine_similarity(q1, q2, dim=1)
    cos_theta = torch.clamp(cos_theta, -1+1e-7, 1-1e-7)
    return 1-torch.pow(cos_theta, 2).mean()

def l1loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def get_ply_path(directory):
    matching_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "point_cloud.ply":
                matching_files.append(os.path.join(root, file))
    matching_files.sort()    
    # Return the last one if the list is not empty
    return matching_files[-1] 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])

    # if aabb is known, we can hard-code it here
    # aabb = [[-1.5,-1.5,-1.5],[1.5,1.5,1.5]]
    aabb = None
    # otherwise, we can estimate it from the point cloud of the first frame
    if aabb is None:
        with open(args.config_path, 'r') as f:
            config = json5.load(f)
        pcd_path = get_ply_path(config["output_path"])
        point_cloud = o3d.io.read_point_cloud(pcd_path)
        aabb = point_cloud.get_axis_aligned_bounding_box()
        buffer_size = 1.5 
        aabb = [(aabb.min_bound - buffer_size).tolist(), (aabb.max_bound + buffer_size).tolist()]

    ntc_conf_path='configs/cache/cache_F_4.json' 
    with open(ntc_conf_path) as ntc_conf_file:
        ntc_conf = json5.load(ntc_conf_file)
    ntc=tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=8, encoding_config=ntc_conf["encoding"], network_config=ntc_conf["network"]).to(torch.device("cuda"))
    ntc_optimizer = torch.optim.Adam(ntc.parameters(), lr=1e-4)
    xyz=fetchXYZ(pcd_path)
    normalzied_xyz=get_contracted_xyz(xyz)
    mask = (normalzied_xyz >= 0) & (normalzied_xyz <= 1)
    mask = mask.all(dim=1)
    ntc_inputs=torch.cat([normalzied_xyz[mask]],dim=-1)
    noisy_inputs = ntc_inputs + 0.1 * (2 * torch.rand_like(ntc_inputs) - 1)
    d_xyz_gt=torch.tensor([0.,0.,0.]).cuda()
    d_rot_gt=torch.tensor([1.,0.,0.,0.]).cuda()
    dummy_gt=torch.tensor([1.]).cuda()
    def cacheloss(resi):
        masked_d_xyz=resi[:,:3]
        masked_d_rot=resi[:,3:7]
        masked_dummy=resi[:,7:8]
        loss_xyz=l1loss(masked_d_xyz,d_xyz_gt)
        loss_rot=quaternion_loss(masked_d_rot,d_rot_gt)
        loss_dummy=l1loss(masked_dummy,dummy_gt)
        loss=loss_xyz+loss_rot+loss_dummy
        return loss
    for iteration in range(0,10000):
        random_inputs = torch.cat([noisy_inputs, torch.rand_like(ntc_inputs)],dim=0)  
        ntc_output=ntc(random_inputs).to(torch.float64)
        loss=cacheloss(ntc_output)
        if iteration % 100 ==0:
            print(loss)
        loss.backward()
        ntc_optimizer.step()
        ntc_optimizer.zero_grad(set_to_none = True)

    # save_path='models/ntc.pth'
    save_path = config['ntc_path']
    ntc=NeuralTransformationCache(ntc,get_xyz_bound()[0],get_xyz_bound()[1])
    torch.save(ntc.state_dict(),save_path)
    print("NTC model saved at", save_path)
