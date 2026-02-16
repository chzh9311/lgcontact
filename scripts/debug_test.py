import torch
from torch.utils.data import DataLoader
import trimesh
import hydra
import open3d as o3d
from matplotlib import pyplot as plt
from common.manopth.manopth.manolayer import ManoLayer
from common.dataset_utils.grab_dataset import GRABDataset
from common.dataset_utils.datamodules import HOIDatasetModule, LocalGridDataModule
from common.utils.vis import visualize_local_grid, visualize_local_grid_with_hand, o3dmesh
from common.msdf.utils.msdf import get_grid
from common.dataset_utils.hoi4d_dataset import HOI4DHandDataModule
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
# from common.model.vae.grid_vae import MLCVAE
from tqdm import tqdm
import numpy as np
from omegaconf import OmegaConf


def visualize_grid_sdf(batch, cfg, dm):
    """
    Visualize grid SDFs from batch['objMsdf'] along with the object mesh.

    The objMsdf tensor has shape (B, N, K^3 + 3) where:
    - First K^3 dimensions are SDF values of the local grid
    - Last 3 dimensions are the grid centers

    Args:
        batch: dict containing 'objMsdf' tensor
        cfg: config with msdf.kernel_size and msdf.scale
        dm: datamodule with access to object mesh info
    """
    obj_msdf = batch['objMsdf']  # (B, N, K^3 + 3)
    kernel_size = cfg.msdf.kernel_size
    scale = cfg.msdf.scale

    B, N, _ = obj_msdf.shape
    k3 = kernel_size ** 3

    # Get normalized grid coordinates from get_grid function
    normalized_coords = get_grid(kernel_size).reshape(-1, 3).numpy()  # (K^3, 3)

    # Use colormap for SDF values
    cmap = plt.colormaps['coolwarm']

    for b in range(min(B, 1)):  # Visualize first sample in batch
        sdf_values = obj_msdf[b, :, :k3].cpu().numpy()  # (N, K^3)
        grid_centers = obj_msdf[b, :, k3:].cpu().numpy()  # (N, 3)

        # Collect all points and their SDF values
        all_points = []
        all_sdf = []

        for i in range(N):
            # Scale and translate grid points to world coordinates
            grid_points = grid_centers[i] + normalized_coords * scale  # (K^3, 3)
            all_points.append(grid_points)
            all_sdf.append(sdf_values[i])

        all_points = np.concatenate(all_points, axis=0)  # (N * K^3, 3)
        all_sdf = np.concatenate(all_sdf, axis=0)  # (N * K^3,)

        # Normalize SDF values for coloring (clip to reasonable range)
        sdf_min, sdf_max = np.percentile(all_sdf, [5, 95])
        sdf_normalized = np.clip((all_sdf - sdf_min) / (sdf_max - sdf_min + 1e-8), 0, 1)

        # Apply colormap
        colors = cmap(sdf_normalized)[:, :3]  # (N * K^3, 3)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Also create point cloud for grid centers
        centers_pcd = o3d.geometry.PointCloud()
        centers_pcd.points = o3d.utility.Vector3dVector(grid_centers)
        centers_pcd.paint_uniform_color([0, 1, 0])  # Green for centers

        # Create object mesh
        obj_name = batch['objName'][b]
        obj_verts = dm.val_set.obj_info[obj_name]['verts'].copy()
        obj_faces = dm.val_set.obj_info[obj_name]['faces']

        # Apply aug_rot if present
        if 'aug_rot' in batch:
            aug_rot = batch['aug_rot'][b].cpu().numpy()  # (3, 3)
            obj_verts = obj_verts @ aug_rot.T

        obj_mesh = o3d.geometry.TriangleMesh()
        obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts)
        obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces)
        obj_mesh.compute_vertex_normals()
        obj_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color

        print(f"\nVisualizing sample {b} (object: {obj_name}):")
        print(f"  Total grid points: {all_points.shape[0]}")
        print(f"  Number of grids: {N}")
        print(f"  SDF range: [{all_sdf.min():.4f}, {all_sdf.max():.4f}]")
        print(f"  Color range (clipped): [{sdf_min:.4f}, {sdf_max:.4f}]")
        print("  Blue = negative SDF (inside), Red = positive SDF (outside)")
        print("  Green points = grid centers, Gray mesh = object")

        o3d.visualization.draw_geometries([pcd, centers_pcd, obj_mesh],
                                          window_name=f"Grid SDF Visualization - Sample {b} ({obj_name})")

OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)

@hydra.main(config_path="../config", config_name="mlcdiff")
def vis_msdf_data_sample(cfg):
    dm = HOIDatasetModule(cfg)

    # Prepare data (preprocessing like M-SDF generation)
    print("Preparing data...")
    dm.prepare_data()

    # Setup validation dataset
    print("Setting up validation dataset...")
    dm.setup('validate')

    # Get validation dataloader (use num_workers=0 to avoid multiprocessing issues)
    val_loader = DataLoader(dm.val_set, batch_size=dm.val_batch_size, shuffle=False,
                            num_workers=4, collate_fn=dm.collate_fn)
    print(f"Validation dataset size: {len(dm.val_set)}")
    print(f"Number of batches: {len(val_loader)}")
    print(f"Batch size: {dm.val_batch_size}")

    # Track handTrans min/max per dimension across all batches
    hand_trans_min = None
    hand_trans_max = None

    # Iterate through validation dataloader
    print("\nTesting validation batches...")
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Processing batches")):
        # Track handTrans range
        if 'handTrans' in batch:
            hand_trans = batch['handTrans']  # (B, 3)
            batch_min = hand_trans.min(dim=0).values  # (3,)
            batch_max = hand_trans.max(dim=0).values  # (3,)

            if hand_trans_min is None:
                hand_trans_min = batch_min
                hand_trans_max = batch_max
            else:
                hand_trans_min = torch.minimum(hand_trans_min, batch_min)
                hand_trans_max = torch.maximum(hand_trans_max, batch_max)

    # Print handTrans range summary
    print(f"\n{'='*50}")
    print("handTrans range across all batches:")
    if hand_trans_min is not None:
        for dim in range(3):
            dim_name = ['X', 'Y', 'Z'][dim]
            print(f"  {dim_name}: min={hand_trans_min[dim].item():.6f}, max={hand_trans_max[dim].item():.6f}, range={hand_trans_max[dim].item() - hand_trans_min[dim].item():.6f}")
    else:
        print("  handTrans not found in batch")

    print("\nDataModule validation complete!")


@hydra.main(config_path="../config", config_name="gridae")
def vis_local_grid_interact(cfg):
    dm = LocalGridDataModule(cfg)
    mano_layer = ManoLayer(mano_root = cfg.data.mano_root, use_pca=False, side='right', flat_hand_mean=True, ncomps=45)
    hand_faces = mano_layer.th_faces.numpy()
    dm.prepare_data()
    dm.setup('validate')
    # train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    # test_loader = dm.test_dataloader()
    print(f"Validation dataset size: {len(dm.val_set)}")
    for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Visualizing validation data"):
        grid_data = torch.cat([batch['gridSDF'], batch['gridContact'], batch['gridHandCSE']], dim=-1)  # (B, K, K, K, C)
        batch_size = grid_data.shape[0]
        for b in range(10):
            # break
            # Extract data for this sample
            local_grid = grid_data[b].cpu().numpy()  # (K, K, K, C)
            contact_point = batch['objSamplePt'][b].cpu().numpy()  # (3,)
            nhand_verts = batch['nHandVerts'][b].cpu().numpy()  # (778, 3)
            obj_rot = batch['objRot'][b].cpu().numpy()  # (3, 3) or axis-angle (3,)
            obj_trans = batch['objTrans'][b].cpu().numpy()  # (3,)
            obj_name = batch['obj_name'][b]

            # Reconstruct object mesh
            obj_mesh_data = dm.val_set.simp_obj_mesh[obj_name]
            # Handle rotation - check if it's axis-angle or matrix
            if obj_rot.shape == (3,):
                from pytorch3d.transforms import axis_angle_to_matrix
                objR = axis_angle_to_matrix(torch.from_numpy(obj_rot)).numpy()
                if dm.val_set.dataset_name == 'grab':
                    objR = objR.T
            else:
                objR = obj_rot

            obj_verts = (objR @ obj_mesh_data['verts'].T).T + obj_trans
            obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_mesh_data['faces'], process=False)
            hand_verts = nhand_verts + contact_point[np.newaxis, :]

            # Visualize
            print(f"\nVisualizing sample {b} from batch {batch_idx} (object: {obj_name})")
            geoms = visualize_local_grid_with_hand(local_grid, hand_verts, hand_faces, dm.val_set.hand_cse, cfg.msdf.kernel_size,
                                           cfg.msdf.scale, contact_point=contact_point, obj_mesh=obj_mesh)
            o3d.visualization.draw_geometries(geoms)
            # break
        # Only visualize first batch
        # break


def test_obj():
    obj = 'toothpaste'
    # msdf = np.load(f'data/preprocessed/grab_msdf/{obj}.npz')['msdf']
    obj_info = GRABDataset.load_mesh_info('data/grab', msdf_path='data/preprocessed/grab_msdf')[obj]
    obj_mesh = trimesh.Trimesh(obj_info['verts'], obj_info['faces'], process=False)
    msdf = obj_info['msdf']
    kernel_size = 7

    # Visualize the first point's local grid
    visualize_local_grid(msdf, kernel_size, point_idx=0, obj_mesh=obj_mesh)


@hydra.main(config_path="../config", config_name="mlcontact_gen")
def test_pointvae(cfg):
    dummy_contact = torch.randn(4, 256, 5, 8, 8, 8)
    dummy_obj_msdf = torch.randn(4, 256, 1, 8, 8, 8)
    dummy_msdf_center = torch.randn(4, 256, 3)
    model = MLCVAE(cfg)
    recon, mu, logvar = model(dummy_contact, dummy_obj_msdf, dummy_msdf_center)


def compare_ckpt():
    ckpt1 = torch.load('logs/wandb_logs/LG3DContact/GRIDAE-128v1/checkpoints/last.ckpt')['state_dict']['handcse.embedding_tensor']
    ckpt2 = torch.load('common/model/hand_cse/hand_cse_4.ckpt')['state_dict']
    ckpt2 = ckpt2['embedding_tensor']
    print(ckpt1)
    print(ckpt2)


def test_manolayer():
    mano_layer = ManoLayer(mano_root = 'data/misc/mano_v1_2/models', use_pca=False, side='right', flat_hand_mean=True, ncomps=45)
    thetas = torch.randn(1, 48)
    betas = torch.randn(1, 10)
    trans = torch.randn(1, 3)
    faces = mano_layer.th_faces.detach().cpu().numpy()
    _, cano_joints, _ = mano_layer(torch.zeros_like(thetas), th_betas=betas)
    verts, joints, _ = mano_layer(thetas, th_betas=betas, th_trans=trans)
    root_j = cano_joints[:, 0]
    targetR = axis_angle_to_matrix(torch.randn(1, 3))
    targett = torch.randn(1, 3)
    verts = verts @ targetR.transpose(-1, -2) + targett.unsqueeze(1)
    root_rot = axis_angle_to_matrix(thetas[:, :3])
    new_root_rot = matrix_to_axis_angle(targetR @ root_rot)
    thetas[:, :3] = new_root_rot
    new_trans = (targetR @ trans.unsqueeze(-1) + targetR @ root_j.unsqueeze(-1)).squeeze(-1) + targett - root_j
    new_verts, new_joints, _ = mano_layer(thetas, th_betas=betas, th_trans=new_trans)

    ori_mesh = o3dmesh(verts[0].cpu().numpy(), faces, color=[0, 0, 1]) # blue
    new_mesh = o3dmesh(new_verts[0].cpu().numpy(), faces, color=[1, 0, 0]) # red
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([ori_mesh, new_mesh, coord_frame])


# Usage Example
def test_hoi4d_datamodule():
    from omegaconf import OmegaConf

    # Example config
    cfg = OmegaConf.create({
        'data': {
            'dataset_path': 'data/HOI4D',
            'preprocessed_dir': 'data/preprocessed',
            'release_file': 'release.txt',
            'num_workers': 8,
        },
        'train': {'batch_size': 32},
        'val': {'batch_size': 32},
        'test': {'batch_size': 32},
    })

    # Create datamodule
    dm = HOI4DHandDataModule(cfg)

    # Run preprocessing
    dm.prepare_data()

    # Setup for training
    dm.setup('fit')

    print(f"Train set size: {len(dm.train_set)}")
    print(f"Val set size: {len(dm.val_set)}")

    # Load a sample
    sample = dm.train_set[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Theta shape: {sample['theta'].shape}")
    print(f"Beta shape: {sample['beta'].shape}")
    print(f"Trans shape: {sample['trans'].shape}")
    print(f"Side: {'right' if sample['side'] == 1 else 'left'}")


@hydra.main(config_path="../config", config_name="handvae")
def fit_mano_beta(cfg):
    mano_layer = ManoLayer(mano_root = cfg.data.mano_root, use_pca=False, side='right', flat_hand_mean=True, ncomps=45)
    train_set = GRABDataset(cfg.data, 'train', load_msdf=cfg.load_msdf, load_grid_contact=cfg.load_grid_contact)


if __name__ == "__main__":
    # vis_msdf_data_sample()
    # test_obj()
    vis_local_grid_interact()
    # test_pointvae()
    # compare_ckpt()
    # test_hoi4d_datamodule()
    # test_manolayer()