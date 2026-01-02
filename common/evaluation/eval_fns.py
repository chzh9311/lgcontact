import coacd
import os.path as osp
import pickle
import json
import subprocess
import trimesh
import numpy as np
import scipy
import scipy.cluster
from scipy.stats import entropy
import torch
import open3d as o3d
from pysdf import SDF
from sklearn.cluster import KMeans
from tqdm import tqdm

import pybullet
import pybullet_utils.bullet_client as bc
from common.utils.vis import o3dmesh_from_trimesh

from evaluation.bullet_simulation import run_simulation
from common.utils.converter import transform_to_canonical, convert_joints


def diversity_legacy(params_list, cls_num=20):
    # k-means (original scipy implementation)
    params_list = scipy.cluster.vq.whiten(params_list)
    codes, dist = scipy.cluster.vq.kmeans(params_list, cls_num)  # codes: [20, 72], dist: scalar
    vecs, dist = scipy.cluster.vq.vq(params_list, codes)  # assign codes, vecs/dist: [1200]
    counts, bins = np.histogram(vecs, len(codes))  # count occurrences  count: [20]
    ee = entropy(counts)
    return ee, np.mean(dist)


def diversity(params_list, cls_num=20):
    # k-means using sklearn (more robust)
    # Whiten the data (normalize by standard deviation)
    params_std = params_list.std(axis=0)
    params_std[params_std == 0] = 1.0  # Avoid division by zero
    params_whitened = params_list / params_std

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=cls_num, max_iter=300, n_init=10, random_state=0)
    kmeans.fit(params_whitened)

    # Get cluster assignments
    labels = kmeans.labels_

    # Calculate distances to cluster centers
    distances = np.linalg.norm(params_whitened[:, np.newaxis] - kmeans.cluster_centers_, axis=2)
    min_distances = distances.min(axis=1)

    # Count occurrences in each cluster
    counts = np.bincount(labels, minlength=cls_num)

    # Calculate entropy
    ee = entropy(counts)

    return ee, np.mean(min_distances)


def parallel_calculate_metrics(params:dict):
    metrics = params['metrics']
    result = {}
    if "PyBullet SimuDisp" in metrics:
        ## decomposition
        if "obj_hulls" not in params:
            mesh = coacd.Mesh(params['obj_model'].vertices, params['obj_model'].faces)
            params['obj_hulls'] = coacd.run_coacd(mesh)
        pb_disp = pybullet_parallel_interface(params) * 100 # to cm
        result["PyBullet SimuDisp"] = pb_disp
        if "Pybullet Stable Rate" in metrics:
            result["Pybullet Stable Rate"] = pb_disp < 2
    if "Intersection Volume" in metrics:
        int_vol = intersect_vox(params['obj_model'], params['hand_model'], pitch=0.005) * 1000000 # turn to cm3
        result["Intersection Volume"] = int_vol
    if "Penetration Depth" in metrics:
        pen_depth = pene_depth(obj_mesh=params['obj_model'], hand_verts=params['hand_model'].vertices) * 100 # to cm
        result["Penetration Depth"] = pen_depth

    if "Pierce-Free Rate" in metrics:
        try:
            pierce, n_holes = determine_pierce(params['obj_model'], params['hand_model'])
            result["Pierce-Free Rate"] = pierce
        except Exception as e:
            print(f"Pierce check failed for {params['frame_name']} with error: {e}")
            result["Pierce-Free Rate"] = False
        # print(pierce)

    if "Contact Ratio" in metrics:
        penetration_tol = 0.005
        hand_verts = params['hand_model'].vertices
        obj_sdf = SDF(params['obj_model'].vertices, params['obj_model'].faces)
        hv_sds = obj_sdf(hand_verts)

        contact = hv_sds > - penetration_tol
        sample_contact = contact.sum() > 0
        result["Contact Ratio"] = sample_contact
    return result


def pybullet_parallel_interface(params:dict):
    client = bc.BulletClient(connection_mode=pybullet.DIRECT)
    hand_verts, hand_faces, obj_verts, obj_faces, fid, obj_hulls = (
        params['hand_model'].vertices, params['hand_model'].faces, params['obj_model'].vertices,
        params['obj_model'].faces, params['idx'], params['obj_hulls'])

    disp = run_simulation(hand_verts, hand_faces, obj_verts, obj_faces, indicator=fid, client=client, obj_hulls=obj_hulls, save_video=False)
    return disp


def determine_pierce(obj_mesh: trimesh.Trimesh, hand_mesh: trimesh.Trimesh):
    """
    The criteria to determine whether the hand penetrates through the object.
    The hand mesh should be watertight.
    Args:
        obj_mesh:
        hand_mesh:

    Returns:
    """
    for i in range(3):
        if obj_mesh.is_watertight:
            break
        else:
            omesh = o3dmesh_from_trimesh(obj_mesh)
            n_faces = len(omesh.triangles)
            mesh_smp = omesh.simplify_quadric_decimation(target_number_of_triangles=int(n_faces / 2))
            obj_mesh = trimesh.Trimesh(vertices=np.asarray(mesh_smp.vertices), faces=np.asarray(mesh_smp.triangles))

    if not hand_mesh.is_watertight:
        return np.nan, np.nan
    subtracted = trimesh.boolean.difference([hand_mesh, obj_mesh])

    # Check if the subtraction was successful
    if subtracted is None:
        raise ValueError("Boolean difference operation failed or resulted in no mesh.")

    # Check the connected components of the resulting mesh
    components = subtracted.split(only_watertight=False)

    # vis_geoms = [o3dmesh_from_trimesh(hand_mesh), o3dmesh_from_trimesh(obj_mesh)]
    # for comp in components:
    #     vis_geoms.append(o3dmesh_from_trimesh(comp).translate((0.3, 0, 0)))
    # Determine if the resulting mesh is continuous
    # o3d.visualization.draw_geometries(vis_geoms)
    no_pierce = len(components) == 1

    return no_pierce, len(components) - 1


def intersect_vox(obj_mesh, hand_mesh, pitch=0.5):
    """
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    """
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume


def calc_diversity(hand_joints):
    cluster = []
    cluster2 = []
    kps = hand_joints.copy()
    for count, kps_i in enumerate(kps):
        cluster.append(kps_i.flatten())

    """cluster2"""
    hand_kps = torch.as_tensor(kps.copy()).float()
    is_right_vec = torch.ones(hand_kps.shape[0], device=hand_kps.device)

    hand_kps = convert_joints(hand_kps, source="mano", target="biomech")

    hand_kps_after, _ = transform_to_canonical(hand_kps, is_right_vec)
    hand_kps_after = convert_joints(hand_kps_after, source="biomech", target="mano")

    for count, kps_flat in enumerate(hand_kps_after):
        cluster2.append(kps_flat.detach().reshape(-1).cpu().numpy())

    cluster_array = np.array(cluster)
    entropy, cluster_size = diversity(cluster_array, cls_num=20)

    cluster_array_2 = np.array(cluster2)
    entropy_2, cluster_size_2 = diversity(cluster_array_2, cls_num=20)

    return entropy, cluster_size, entropy_2, cluster_size_2

def pene_depth(obj_mesh, hand_verts):
    trimesh.repair.fix_normals(obj_mesh)

    # obj_triangles = obj_mesh.vertices[obj_mesh.faces]
    # exterior = batch_mesh_contains_points(torch.from_numpy(hand_verts[None, :, :]).float(),
    #                                                    torch.from_numpy(obj_triangles)[None, :, :, :].float())
    # penetr_mask = ~exterior.squeeze(dim=0)
    penetr_mask = obj_mesh.contains(hand_verts)

    if penetr_mask.sum() == 0:
        max_depth = 0
    else:
        (result_close, result_distance, _, ) = trimesh.proximity.closest_point(obj_mesh, hand_verts[penetr_mask == 1])
        max_depth = result_distance.max()

    return max_depth

def calculate_metrics(param_list, pool=None, metrics=[], reduction='mean'):
    for p in param_list:
        p["metrics"] = metrics

    if pool is not None:
        result_list = pool.map(parallel_calculate_metrics, param_list)
    else:
        result_list = []
        for p in tqdm(param_list):
             result_list.append(parallel_calculate_metrics(p))
        # result_list = [parallel_calculate_metrics(p) for p in param_list]
    result = {}

    if "Success Rate" in metrics:
        res = {'hand_verts': torch.stack([torch.as_tensor(p['hand_model'].vertices).float() for p in param_list], dim=0),
            'hand_joints': torch.stack([torch.as_tensor(p['hand_joints']).float() for p in param_list], dim=0),
            'obj_verts': [torch.as_tensor(p['obj_model'].vertices).float() for p in param_list],
            'obj_faces': [torch.as_tensor(p['obj_model'].faces).float() for p in param_list]
            }

        batch_idx = param_list[0]['frame_name']
        with open(osp.join('tmp', 'exp_output', f'sample_result_{batch_idx}.pkl'), 'wb') as f:
            pickle.dump(res, f)
        subprocess.run(
            ['/home/zxc417/anaconda3/envs/ugg/bin/python', '/home/zxc417/Projects/reproductions/ugg/test_success_rate.py',
             '-i', f'tmp/exp_output/sample_result_{batch_idx}.pkl', '-o', f'tmp/exp_output/success_rate_{batch_idx}.json'])
        with open(f'tmp/exp_output/success_rate_{batch_idx}.json', 'r') as f:
            success_rate = json.load(f)
        result["Success Rate"] = np.array([success_rate['6d']])

    for k in result_list[0].keys():
        result[k] = [rit[k] for rit in result_list]
    for m in result.keys():
        if reduction == 'mean':
            result[m] = np.mean(np.asarray(result[m])).item()
        elif reduction == 'sum':
            result[m] = np.sum(np.asarray(result[m])).item()
        elif reduction == "none":
            result[m] = np.asarray(result[m])
        else:
            raise ValueError(f"Unknown reduction {reduction}")
    return result
