import trimesh
import hydra
import open3d as o3d    
from common.dataset_utils.hoi_dataset import HOIDatasetModule
from common.dataset_utils.local_grid_dataset import LocalGridDataModule
from common.dataset_utils.grab_dataset import GRABDataset
from common.utils.vis import o3dmesh, o3dmesh_from_trimesh
import numpy as np

@hydra.main(config_path="../config", config_name="mlcontact")
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
    dm = LocalGridDataModule(cfg, split='val')
    dm.prepare_data()


def visualize_local_grid(msdf, kernel_size, point_idx, obj_mesh):
    """
    Visualize one local grid of MSDF with the object.

    Args:
        msdf: numpy array of shape (n_points, kernel_size**3 + 3 + 1)
        kernel_size: size of the cubic grid
        point_idx: which point's grid to visualize
        obj_mesh: trimesh mesh of the object
    """
    # Extract data for the selected point
    point_data = msdf[point_idx]
    sdf_values = point_data[:kernel_size**3]
    center = point_data[kernel_size**3:kernel_size**3+3]
    scale = point_data[-1]

    # Generate grid points (same as get_grid_points function)
    # Grid in range [-1, 1] for each dimension
    indices = np.array([(i, j, k) for i in range(kernel_size)
                        for j in range(kernel_size)
                        for k in range(kernel_size)])

    # Convert to coordinates in [-1, 1] range
    coords = 2 * indices - (kernel_size - 1)
    coords = coords / (kernel_size - 1)

    # Scale and translate to world coordinates
    grid_points = center + coords * scale

    # Create point cloud for grid points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid_points)

    # Color points based on SDF values
    sdf_min, sdf_max = sdf_values.min(), sdf_values.max()

    # Use a colormap: blue (negative/inside) -> white (zero) -> red (positive/outside)
    # Map to RGB where blue=negative, red=positive
    colors = np.zeros((len(sdf_values), 3))
    for i, sdf_val in enumerate(sdf_values):
        if sdf_val < 0:  # Inside object - blue shades
            intensity = min(abs(sdf_val) / abs(sdf_min) if sdf_min < 0 else 1, 1)
            colors[i] = [1-intensity, 1-intensity, 1]  # Blue
        else:  # Outside object - red shades
            intensity = min(sdf_val / sdf_max if sdf_max > 0 else 1, 1)
            colors[i] = [1, 1-intensity, 1-intensity]  # Red

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create a bounding box to show the grid extent
    bbox_points = np.array([
        center + scale * np.array([-1, -1, -1]),
        center + scale * np.array([1, -1, -1]),
        center + scale * np.array([1, 1, -1]),
        center + scale * np.array([-1, 1, -1]),
        center + scale * np.array([-1, -1, 1]),
        center + scale * np.array([1, -1, 1]),
        center + scale * np.array([1, 1, 1]),
        center + scale * np.array([-1, 1, 1]),
    ])

    bbox_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(bbox_points)
    line_set.lines = o3d.utility.Vector2iVector(bbox_lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in bbox_lines])  # Green bbox

    # Create center point marker for current grid (larger, yellow)
    center_pcd = o3d.geometry.PointCloud()
    center_pcd.points = o3d.utility.Vector3dVector([center])
    center_pcd.colors = o3d.utility.Vector3dVector([[1, 1, 0]])  # Yellow

    # Create point cloud for ALL sampled points (all grid centers)
    all_centers = msdf[:, kernel_size**3:kernel_size**3+3]
    all_centers_pcd = o3d.geometry.PointCloud()
    all_centers_pcd.points = o3d.utility.Vector3dVector(all_centers)
    # Color all centers in cyan, except the current one
    all_centers_colors = np.tile([0, 1, 1], (len(all_centers), 1))  # Cyan
    all_centers_pcd.colors = o3d.utility.Vector3dVector(all_centers_colors)

    # Convert object mesh to open3d
    obj_o3d_mesh = o3dmesh_from_trimesh(obj_mesh)

    # Print information
    print(f"\nVisualizing MSDF Grid for Point {point_idx}")
    print(f"Center: {center}")
    print(f"Scale: {scale}")
    print(f"SDF values range: [{sdf_min:.4f}, {sdf_max:.4f}]")
    print(f"Grid size: {kernel_size}x{kernel_size}x{kernel_size} = {kernel_size**3} points")
    print(f"Total number of sampled points: {len(all_centers)}")
    print(f"\nColor scheme:")
    print(f"  Blue: Negative SDF (inside object)")
    print(f"  Red: Positive SDF (outside object)")
    print(f"  Yellow: Current grid center (point {point_idx})")
    print(f"  Cyan: All other sampled points (grid centers)")
    print(f"  Green: Grid bounding box")

    # Visualize
    o3d.visualization.draw_geometries(
        [obj_o3d_mesh, pcd, center_pcd, all_centers_pcd, line_set],
        window_name=f"MSDF Grid Visualization - Point {point_idx}"
    )


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
    # vis_data_samples()
    # test_obj()
    vis_local_grid_interact()