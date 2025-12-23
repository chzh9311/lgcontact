import trimesh
import hydra
import open3d as o3d    
from common.manopth.manopth.manolayer import ManoLayer
from common.dataset_utils.grab_dataset import GRABDataset
from common.dataset_utils.datamodules import HOIDatasetModule, LocalGridDataModule
from common.utils.vis import visualize_local_grid, visualize_local_grid_with_hand
import numpy as np

@hydra.main(config_path="../config", config_name="mlcontact_gen")
def vis_msdf_data_sample(cfg):
    dm = HOIDatasetModule(cfg)

    # Prepare data (preprocessing like M-SDF generation)
    print("Preparing data...")
    dm.prepare_data()

    # Setup validation dataset
    print("Setting up validation dataset...")
    dm.setup('validate')

    # Get validation dataloader
    val_loader = dm.val_dataloader()
    print(f"Validation dataset size: {len(dm.val_set)}")
    print(f"Number of batches: {len(val_loader)}")
    print(f"Batch size: {dm.val_batch_size}")

    # Iterate through validation dataloader
    print("\nTesting validation batches...")
    for batch_idx, batch in enumerate(val_loader):
        print(f"\n{'='*50}")
        print(f"Batch {batch_idx}:")
        print(f"  Keys: {list(batch.keys())}")

        # Print shapes/info for each field in the batch
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
            else:
                print(f"  {key}: {value}")

        # Test first 5 batches only
        if batch_idx >= 4:
            print(f"\n{'='*50}")
            print("Successfully tested 5 batches!")
            break

    print("\nDataModule validation complete!")


@hydra.main(config_path="../config", config_name="mlcontact_vqvae")
def vis_local_grid_interact(cfg):
    dm = LocalGridDataModule(cfg)
    mano_layer = ManoLayer(mano_root = cfg.data.mano_root, use_pca=False, side='right', flat_hand_mean=True, ncomps=45)
    hand_faces = mano_layer.th_faces.numpy()
    dm.prepare_data()
    dm.setup('validate')
    val_loader = dm.val_dataloader()
    print(f"Validation dataset size: {len(dm.val_set)}")
    for batch_idx, batch in enumerate(val_loader):
        grid_data = batch['localGrid']  # (B, K, K, K, C)
        batch_size = grid_data.shape[0]
        for b in range(batch_size):
            # Extract data for this sample
            local_grid = grid_data[b].cpu().numpy()  # (K, K, K, C)
            contact_point = batch['obj_sample_pt'][b].cpu().numpy()  # (3,)
            hand_verts = batch['handVerts'][b].cpu().numpy()  # (778, 3)
            obj_rot = batch['objRot'][b].cpu().numpy()  # (3, 3) or axis-angle (3,)
            obj_trans = batch['objTrans'][b].cpu().numpy()  # (3,)
            obj_name = batch['obj_name'][b]

            # Reconstruct object mesh
            obj_mesh_data = dm.val_set.simp_obj_mesh[obj_name]
            # Handle rotation - check if it's axis-angle or matrix
            if obj_rot.shape == (3,):
                from pytorch3d.transforms import axis_angle_to_matrix
                import torch
                objR = axis_angle_to_matrix(torch.from_numpy(obj_rot)).numpy()
                if dm.val_set.dataset_name == 'grab':
                    objR = objR.T
            else:
                objR = obj_rot

            obj_verts = (objR @ obj_mesh_data['verts'].T).T + obj_trans
            obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_mesh_data['faces'], process=False)

            # Visualize
            print(f"\nVisualizing sample {b} from batch {batch_idx} (object: {obj_name})")
            visualize_local_grid_with_hand(local_grid, contact_point, obj_mesh, hand_verts,
                                          hand_faces, dm.val_set.hand_cse, cfg.msdf.kernel_size, cfg.msdf.scale)
        # Only visualize first batch
        break


def test_obj():
    obj = 'toothpaste'
    # msdf = np.load(f'data/preprocessed/grab_msdf/{obj}.npz')['msdf']
    obj_info = GRABDataset.load_mesh_info('data/grab', msdf_path='data/preprocessed/grab_msdf')[obj]
    obj_mesh = trimesh.Trimesh(obj_info['verts'], obj_info['faces'], process=False)
    msdf = obj_info['msdf']
    kernel_size = 7

    # Visualize the first point's local grid
    visualize_local_grid(msdf, kernel_size, point_idx=0, obj_mesh=obj_mesh)


if __name__ == "__main__":
    vis_msdf_data_sample()
    # test_obj()
    # vis_local_grid_interact()