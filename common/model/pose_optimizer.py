import numpy as np
import torch
import torch.nn.functional as F
from copy import copy

from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_axis_angle
from common.utils.geometry import (
        cp_match,
        calculate_contact_capsule,
        calculate_penetration_cost
    )


def optimize_pose_contactopt(mano_layer, obj_verts, obj_normals, obj_contact_target, obj_partition, n_iter=1200, lr=0.002, w_cont_obj=1,
                  save_history=False, ncomps=26, w_cont_asym=2, caps_top=0.0005, caps_bot=-0.001, caps_rad=0.001, caps_on_hand=False,
                  contact_norm_method=0, w_pen_cost=600, pen_it=0):
    """Runs differentiable optimization to align the hand with the target contact map.
    Minimizes the loss between ground truth contact and contact calculated with DiffContact"""
    batch_size = obj_contact_target.shape[0]
    device = obj_contact_target.device

    opt_vector = torch.zeros((batch_size, ncomps + 16), device=device)   # 3 hand rot, 10 hand shape, 3 hand trans
    opt_vector.requires_grad = True

    optimizer = torch.optim.Adam([opt_vector], lr=lr, amsgrad=True)  # AMSgrad helps
    loss_criterion = torch.nn.L1Loss(reduction='none')  # Benchmarked, L1 performs best vs MSE/SmoothL1
    opt_state = []

    for it in range(n_iter):
        optimizer.zero_grad()

        hand_verts, hand_joints, _ = mano_layer(opt_vector[:, :ncomps+3], th_betas=opt_vector[:, ncomps+3:ncomps+13], th_trans=opt_vector[:, ncomps+13:ncomps+16])

        if contact_norm_method != 0 and not caps_on_hand:
            with torch.no_grad():   # We need to calculate hand normals if using more complicated methods
                mano_mesh = Meshes(verts=hand_verts, faces=mano_layer.th_faces.repeat(batch_size, 1, 1))
                hand_normals = mano_mesh.verts_normals_padded()
        else:
            hand_normals = torch.zeros(hand_verts.shape, device=device)

        contact_obj, contact_hand, _ = calculate_contact_capsule(hand_verts, hand_normals, obj_verts, obj_normals,
                              caps_top=caps_top, caps_bot=caps_bot, caps_rad=caps_rad, caps_on_hand=caps_on_hand, contact_norm_method=contact_norm_method)

        contact_obj_sub = obj_contact_target.unsqueeze(-1) - contact_obj
        contact_obj_weighted = contact_obj_sub + torch.nn.functional.relu(contact_obj_sub) * w_cont_asym  # Loss for 'missing' contact higher
        loss_contact_obj = loss_criterion(contact_obj_weighted, torch.zeros_like(contact_obj_weighted)).mean(dim=(1, 2))

        # contact_hand_sub = hand_contact_target - contact_hand
        # contact_hand_weighted = contact_hand_sub + torch.nn.functional.relu(contact_hand_sub) * w_cont_asym  # Loss for 'missing' contact higher
        # loss_contact_hand = loss_criterion(contact_hand_weighted, torch.zeros_like(contact_hand_weighted)).mean(dim=(1, 2))

        loss = loss_contact_obj * w_cont_obj

        is_thin = torch.zeros(batch_size, device=device)  # For hands, we never use thin mode
        if w_pen_cost > 0 and it >= pen_it:
            pen_cost = calculate_penetration_cost(hand_verts, hand_normals, obj_verts, obj_normals, is_thin, contact_norm_method)
            loss += pen_cost.mean(dim=1) * w_pen_cost

        out_dict = {'loss': loss.detach().cpu()}
        if save_history:
            out_dict['hand_verts'] = hand_verts.detach().cpu()#.numpy()
            out_dict['hand_joints'] = hand_joints.detach().cpu()#.numpy()
            out_dict['contact_obj'] = contact_obj.detach().cpu()#.numpy()
            out_dict['contact_hand'] = contact_hand.detach().cpu()#.numpy()
        opt_state.append(out_dict)
        if it % 100 == 99:
            print(f"Iter {it} | Loss: {loss.mean().item():.6f} | Contact Obj Loss: {loss_contact_obj.mean().item():.6f}")
        loss.mean().backward()
        optimizer.step()

    return opt_vector.detach()


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
