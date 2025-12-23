import trimesh
import hydra
import open3d as o3d    
from common.manopth.manopth.manolayer import ManoLayer
from common.dataset_utils.grab_dataset import GRABDataset
from common.dataset_utils.datamodules import HOIDatasetModule, LocalGridDataModule
from common.utils.vis import o3dmesh, o3dmesh_from_trimesh
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


def visualize_local_grid_with_hand(local_grid, contact_point, obj_mesh, hand_verts, hand_faces, hand_cse, kernel_size, grid_scale):
    """
    Visualize local grid with object mesh, hand mesh, and grid points.

    Args:
        local_grid: numpy array of shape (kernel_size, kernel_size, kernel_size, C)
                   where C >= 18 (SDF, contact, CSE[16])
        contact_point: numpy array of shape (3,), the center of the grid
        obj_mesh: trimesh mesh of the object
        hand_verts: numpy array of shape (778, 3), hand vertices
        hand_faces: numpy array of shape (N, 3), hand face indices
        hand_cse: numpy array of shape (778, 16), hand contact surface embeddings
        kernel_size: int, size of the cubic grid
        grid_scale: float, scale of the grid
    """
    # Generate normalized grid coordinates
    indices = np.array([(i, j, k) for i in range(kernel_size)
                        for j in range(kernel_size)
                        for k in range(kernel_size)])

    # Convert to coordinates in [-1, 1] range
    coords = 2 * indices - (kernel_size - 1)
    coords = coords / (kernel_size - 1)

    # Scale and translate to world coordinates
    grid_points = contact_point[None, :] + coords * grid_scale

    # Extract SDF values (first channel)
    sdf_values = local_grid[:, :, :, 0].flatten()

    # Create point cloud for grid points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid_points)

    # Color points based on SDF values
    sdf_min, sdf_max = sdf_values.min(), sdf_values.max()
    colors = np.zeros((len(sdf_values), 3))
    for i, sdf_val in enumerate(sdf_values):
        if sdf_val < 0:  # Inside object - blue shades
            intensity = min(abs(sdf_val) / abs(sdf_min) if sdf_min < 0 else 1, 1)
            colors[i] = [1-intensity, 1-intensity, 1]  # Blue
        else:  # Outside object - red shades
            intensity = min(sdf_val / sdf_max if sdf_max > 0 else 1, 1)
            colors[i] = [1, 1-intensity, 1-intensity]  # Red

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create contact point marker (larger, yellow)
    contact_pcd = o3d.geometry.PointCloud()
    contact_pcd.points = o3d.utility.Vector3dVector([contact_point])
    contact_pcd.colors = o3d.utility.Vector3dVector([[1, 1, 0]])  # Yellow

    # Create bounding box for the grid
    bbox_points = np.array([
        contact_point + grid_scale * np.array([-1, -1, -1]),
        contact_point + grid_scale * np.array([1, -1, -1]),
        contact_point + grid_scale * np.array([1, 1, -1]),
        contact_point + grid_scale * np.array([-1, 1, -1]),
        contact_point + grid_scale * np.array([-1, -1, 1]),
        contact_point + grid_scale * np.array([1, -1, 1]),
        contact_point + grid_scale * np.array([1, 1, 1]),
        contact_point + grid_scale * np.array([-1, 1, 1]),
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

    # Convert object mesh to open3d
    obj_o3d_mesh = o3dmesh_from_trimesh(obj_mesh)

    # Create hand mesh
    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh.paint_uniform_color([0.8, 0.6, 0.4])  # Skin color
    hand_mesh.compute_vertex_normals()

    # ==================== DUPLICATE GEOMETRIES WITH OFFSET ====================
    # Calculate offset along x-axis (make it large enough to separate the scenes)
    bbox = obj_mesh.bounds
    offset_distance = (bbox[1][0] - bbox[0][0]) * 2  # 2x the object width
    offset = np.array([offset_distance, 0, 0])

    # Duplicate and offset grid points
    grid_points_offset = grid_points + offset

    # Extract contact likelihood values (channel 1)
    contact_values = local_grid[:, :, :, 1].flatten()

    # Create point cloud for offset grid with contact likelihood coloring
    pcd_offset = o3d.geometry.PointCloud()
    pcd_offset.points = o3d.utility.Vector3dVector(grid_points_offset)

    # Color using inferno colormap (blue=0 to red=1)
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('inferno')
    contact_colors = np.array([cmap(val)[:3] for val in contact_values])  # Get RGB, ignore alpha
    pcd_offset.colors = o3d.utility.Vector3dVector(contact_colors)

    # Duplicate contact point marker
    contact_pcd_offset = o3d.geometry.PointCloud()
    contact_pcd_offset.points = o3d.utility.Vector3dVector([contact_point + offset])
    contact_pcd_offset.colors = o3d.utility.Vector3dVector([[1, 1, 0]])  # Yellow

    # Duplicate bounding box
    bbox_points_offset = bbox_points + offset
    line_set_offset = o3d.geometry.LineSet()
    line_set_offset.points = o3d.utility.Vector3dVector(bbox_points_offset)
    line_set_offset.lines = o3d.utility.Vector2iVector(bbox_lines)
    line_set_offset.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in bbox_lines])

    # Duplicate object mesh
    obj_o3d_mesh_offset = o3dmesh_from_trimesh(obj_mesh)
    obj_o3d_mesh_offset.translate(offset)

    # Duplicate hand mesh
    hand_mesh_offset = o3d.geometry.TriangleMesh()
    hand_verts_offset = hand_verts + offset
    hand_mesh_offset.vertices = o3d.utility.Vector3dVector(hand_verts_offset)
    hand_mesh_offset.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh_offset.paint_uniform_color([0.8, 0.6, 0.4])
    hand_mesh_offset.compute_vertex_normals()

    # ==================== CREATE CORRESPONDENCE LINES ====================
    # Extract predicted CSE from local_grid (indices 2:18, which is 16 dimensions)
    predicted_cse = local_grid[:, :, :, 2:18].reshape(-1, 16)  # (kernel_size^3, 16)

    # Compute distance matrix between predicted CSE and hand CSE
    # predicted_cse: (K^3, 16), hand_cse: (778, 16)
    dist_matrix = np.linalg.norm(predicted_cse[:, None, :] - hand_cse[None, :, :], axis=2)  # (K^3, 778)
    nearest_hand_idx = np.argmin(dist_matrix, axis=1)  # (K^3,) - index of nearest hand vertex for each grid point

    # Filter to only high contact likelihood points (> 0.5)
    high_contact_mask = contact_values > 0.5
    high_contact_grid_points = grid_points_offset[high_contact_mask]
    high_contact_nearest_idx = nearest_hand_idx[high_contact_mask]

    # Create lines from high-contact grid points to their nearest hand vertices
    line_points = []
    line_indices = []
    for i, grid_point in enumerate(high_contact_grid_points):
        hand_vert_idx = high_contact_nearest_idx[i]
        hand_point = hand_verts_offset[hand_vert_idx]

        # Add both points
        line_points.append(grid_point)
        line_points.append(hand_point)

        # Add line index
        line_indices.append([len(line_points) - 2, len(line_points) - 1])

    # Create line set for correspondences
    correspondence_lines = o3d.geometry.LineSet()
    if len(line_points) > 0:
        correspondence_lines.points = o3d.utility.Vector3dVector(line_points)
        correspondence_lines.lines = o3d.utility.Vector2iVector(line_indices)
        correspondence_lines.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in line_indices])  # Green lines

    # Print information
    print(f"\nVisualizing Local Grid")
    print(f"Contact point: {contact_point}")
    print(f"Grid scale: {grid_scale}")
    print(f"SDF values range: [{sdf_min:.4f}, {sdf_max:.4f}]")
    print(f"Contact likelihood range: [{contact_values.min():.4f}, {contact_values.max():.4f}]")
    print(f"Grid size: {kernel_size}x{kernel_size}x{kernel_size} = {kernel_size**3} points")
    print(f"Hand vertices: {len(hand_verts)}")
    print(f"Hand faces: {len(hand_faces)}")
    print(f"High contact points (>0.5): {high_contact_mask.sum()}")
    print(f"Correspondence lines: {len(line_indices)}")
    print(f"\nLeft scene - Original (SDF-colored):")
    print(f"  Blue: Negative SDF (inside object)")
    print(f"  Red: Positive SDF (outside object)")
    print(f"  Yellow: Contact point (grid center)")
    print(f"  Green: Grid bounding box")
    print(f"  Skin color: Hand mesh")
    print(f"  Gray: Object mesh")
    print(f"\nRight scene - Contact & Correspondence:")
    print(f"  Inferno colormap: Blue (no contact) → Red (high contact)")
    print(f"  Green lines: Correspondence to nearest hand vertex (CSE-based)")

    # Visualize all geometries
    geometries = [
        # Original scene
        obj_o3d_mesh, pcd, contact_pcd, line_set, hand_mesh,
        # Offset scene with contact likelihood
        obj_o3d_mesh_offset, pcd_offset, contact_pcd_offset, line_set_offset, hand_mesh_offset
    ]

    # Add correspondence lines if any exist
    if len(line_points) > 0:
        geometries.append(correspondence_lines)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Local Grid with Hand and Object - Dual View"
    )


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
    vis_msdf_data_sample()
    # test_obj()
    # vis_local_grid_interact()