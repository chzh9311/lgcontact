import numpy as np
import torch
import torch.nn.functional as F
from copy import copy
from sklearn.cluster import AgglomerativeClustering

## TMP import
import matplotlib.pyplot as plt
import open3d as o3d

from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_axis_angle
from common.utils.geometry import get_perpend_vecs, normalize_vec_tensor, cp_match
from scipy.sparse.csc import csc_matrix


def optimize_pose_wrt_local_grids(mano_layer, target_pts, target_W_verts, weights, n_iter=1200, lr=0.01):
    """
    :param mano_layer: Description
    :param target_pts: Description
    :param n_iter: Description
    :param lr: Description
    """
    batch_size = target_pts.shape[0]
    global_pose = torch.zeros((batch_size, 3), dtype=target_pts.dtype, device=target_pts.device)
    mano_trans = torch.zeros((batch_size, 3), dtype=target_pts.dtype, device=target_pts.device)
    
    mano_pose = torch.zeros((batch_size, mano_layer.ncomps), dtype=target_pts.dtype, device=target_pts.device)
    mano_shape = torch.zeros((batch_size, 10), dtype=target_pts.dtype, device=target_pts.device)

    ## Initialization
    handV, handJ, _ = mano_layer(torch.cat([global_pose, mano_pose], dim=1), th_betas=mano_shape, th_trans=mano_trans)
    root = handJ[:, 0:1]
    Rs, ts = cp_match(target_W_verts @ handV - root, target_pts - root, weights.unsqueeze(-1))
    global_pose = matrix_to_axis_angle(Rs)
    mano_trans = ts

    mano_pose.requires_grad = True
    mano_shape.requires_grad = True
    global_pose.requires_grad = True
    mano_trans.requires_grad = True 
    hand_opt_params = [global_pose, mano_pose, mano_shape, mano_trans]
    optimizer = torch.optim.Adam(hand_opt_params, lr=lr)
    for it in range(n_iter):
        handV, handJ, _ = mano_layer(torch.cat([global_pose, mano_pose], dim=1), th_betas=mano_shape, th_trans=mano_trans)
        optimizer.zero_grad()
        loss = torch.mean(weights.unsqueeze(-1) * F.mse_loss(target_W_verts @ handV, target_pts, reduction='none')) * 10000
        
        loss.backward()
        optimizer.step()
        if it % 100 == 99:
            print(f"Iter {it} | Loss: {loss.item():.6f}")
    
    return [p.detach() for p in hand_opt_params]
