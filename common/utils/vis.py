import numpy as np
import trimesh
import open3d as o3d
from matplotlib import pyplot as plt
from copy import deepcopy
from sklearn.manifold import Isomap
from common.utils.geometry import point_set_register, get_v2v_rot
import os

def o3dmesh(vert, face, color=None):
    """
    Input: vert_list, face_list: list of numpy arrays of shape [N, 3]
           color_list: list of str
           writer: Tensorboard Summary Writer
    """
    vert = o3d.utility.Vector3dVector(vert)
    face = o3d.utility.Vector3iVector(face)
    mesh = o3d.geometry.TriangleMesh(vertices=vert, triangles=face)
    cmap = plt.colormaps['hsv']
    if color is None:
        # color = np.clip(np.random.randn(3), 0, 1)
        color = cmap(np.random.rand())[:3]
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()

    return mesh
    # o3d.visualization.draw_geometries(meshes, mesh_show_wireframe=True)


def o3dmesh_from_trimesh(mesh, color=None):
    return o3dmesh(mesh.vertices, mesh.faces, color)


def register_and_vis_hand_parts(hpc:np.ndarray, opc:np.ndarray) -> list:
    """
    register all object pointclouds to the root & rotate all hand parts.
    To test whether the algorithm works as expected.
    :param hpc: hand point cloud <n_parts x n1 x 3>
    :param opc: object point cloud <n_parts x n2 x 3>
    :return: a list of <o3d.geometry.PointCloud>
    """
    vis_geoms = []
    part_cmap = plt.colormaps['hsv']
    for i in range(16):
        R, t = point_set_register(opc[i], opc[0])
        vis_hand_pc = hpc[i] @ R.T + t.reshape(1, 3)
        vis_hand_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vis_hand_pc))
        vis_hand_pc.paint_uniform_color(part_cmap(i / 16)[:3])
        vis_geoms.append(vis_hand_pc)

    obj_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(opc[0]))
    vis_geoms.append(obj_pc)

    return vis_geoms


def visualize_latent(obj_shape, sample_pts, pos_feat, sdf):
    """
    Visualize the predicted latent position features by isomapping reduction. Only one sample
    :param obj_shape: trimesh
    :param sample_pts: <N1 x 3>
    :param pos_feat: <N1 x D>
    :param sdf: <N1>
    """
    isomap = Isomap(n_neighbors=10, n_components=2)
    obj_mesh = o3dmesh_from_trimesh(obj_shape)
    pos_feat_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample_pts))
    sdf_pc = deepcopy(pos_feat_pc)
    pos_isomap = isomap.fit_transform(pos_feat)
    up, bottom, right, left = pos_isomap[:, 0].max(), pos_isomap[:, 0].min(), pos_isomap[:, 1].max(), pos_isomap[:, 1].min()
    ## Normalize
    pos_isomap = (pos_isomap - np.array([[bottom, left]])) / np.array([[up - bottom], [right - left]])
    hsv_cmap = plt.get_cmap('hsv')
    pos_color = hsv_cmap(pos_isomap[:, 1])[:, :3]
    satr = 1 - np.abs(2 * pos_isomap[:, 0] - 1)
    pos_color[pos_isomap[:, 0] < 0.5] = pos_color[pos_isomap[:, 0] < 0.5] * satr + 1 * (1 - satr)
    pos_color[pos_isomap[:, 0] > 0.5] = pos_color[pos_isomap[:, 0] > 0.5] * satr
    hm_cmap = plt.get_cmap('inferno')

    pos_feat_pc.colors = o3d.utility.Vector3dVector(pos_color)
    sdf_pc.colors = o3d.utility.Vector3dVector(hm_cmap(sdf)[:, :3])
    sdf_pc.translate((0.4, 0, 0))
    o3d.visualization.draw_geometries([obj_mesh, pos_feat_pc, sdf_pc])


def o3d_arrow(anchor: np.ndarray, direction: np.ndarray,
              color: (np.ndarray, None) = None, scale: float = 1.0, log=False) -> o3d.geometry.TriangleMesh:
    arr_len = np.linalg.norm(direction, axis=-1)
    if log:
        arr_len = 4.12 * np.log(arr_len + 1) # 4.12 * log (9.8 + 1) = 9.8
    a = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=0.002,
        cone_height=0.002,
        cylinder_radius=0.001,
        cylinder_height=arr_len * scale,
    )
    # Transform to the right direction
    original_dire = np.array([0, 0, 1])
    R = get_v2v_rot(original_dire, direction)
    a.rotate(R, center=np.array([0, 0, 0]))
    a.translate(anchor)
    if color is not None:
        a.paint_uniform_color(color)
    a.compute_vertex_normals()

    return a

def o3d_point(pos, radius, color=None):
    pt = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    pt.translate(pos)
    if color is not None:
        pt.paint_uniform_color(color)
    pt.compute_vertex_normals()
    return pt


def geom_to_img(vis_geoms, w, h):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend

    # Collect all meshes
    all_vertices = []
    mesh_data = []

    for geom in vis_geoms:
        if type(geom) is dict:
            o3d_geom = geom['geometry']
        else:
            o3d_geom = geom

        if isinstance(o3d_geom, o3d.geometry.TriangleMesh):
            vertices = np.asarray(o3d_geom.vertices)
            faces = np.asarray(o3d_geom.triangles)

            if len(vertices) == 0 or len(faces) == 0:
                continue

            # Get vertex colors if available
            if o3d_geom.has_vertex_colors():
                colors = np.asarray(o3d_geom.vertex_colors)
            else:
                colors = np.ones((len(vertices), 3)) * 0.5  # Default gray

            all_vertices.append(vertices)
            mesh_data.append({'vertices': vertices, 'faces': faces, 'colors': colors})

    if len(all_vertices) == 0:
        print("Warning: No valid geometries to render")
        return np.ones((h, w * 4, 3))

    # Compute scene bounds
    all_vertices = np.vstack(all_vertices)
    center = np.mean(all_vertices, axis=0)
    extent = np.ptp(all_vertices, axis=0)
    # Tight framing for 80% coverage - use smaller multiplier
    max_extent = np.max(extent)
    half_range = max_extent * 0.7  # Even tighter for better zoom

    result_imgs = []
    for i in range(4):
        # Create figure
        dpi = 100
        fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')

        # Set camera position for each view
        angle = np.pi * i / 2
        elev = 10  # Slight elevation for better 3D view
        azim = np.degrees(angle)
        ax.view_init(elev=elev, azim=azim)

        # Collect all faces from all meshes with their colors for proper depth sorting
        all_faces = []
        all_face_colors = []

        for mesh in mesh_data:
            verts = mesh['vertices']
            faces = mesh['faces']
            colors = mesh['colors']

            # Create face colors (average vertex colors for each face)
            face_colors = colors[faces].mean(axis=1)

            # Add each face
            for face_idx, face in enumerate(faces):
                face_verts = [verts[face[0]], verts[face[1]], verts[face[2]]]
                all_faces.append(face_verts)
                all_face_colors.append(face_colors[face_idx])

        # Create single polygon collection with all faces for proper depth sorting
        collection = Poly3DCollection(all_faces, facecolors=all_face_colors,
                                     edgecolors='none', alpha=0.95, linewidths=0,
                                     zsort='average')  # Enable depth sorting
        ax.add_collection3d(collection)

        # Set tight limits around the object
        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)

        # Force equal aspect ratio and tight layout
        ax.set_box_aspect([1, 1, 1])
        ax.set_facecolor('white')
        ax.grid(False)
        ax.set_axis_off()

        # Additional zoom control - set camera distance
        ax.dist = 1  # Lower value = closer camera (default is 10)

        # Render to array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        result_imgs.append(img_array[:, :, :3] / 255.0)

        plt.close(fig)

    ret_img = np.concatenate(result_imgs, axis=1)
    return ret_img


def vis_nn_bps(hand_mesh, bps, nn_idx):
    """
    visualize the nearest point in the bps to the corresonding hand vertex.
    For all points in BPS that are in correspondence with the same hand vertex,
    selected only the nearest one. All seleted points are visualized as a point cloud, together with the hand mesh.

    :param hand_mesh: trimesh.Trimesh of 778 vertices
    :param bps: N^3 x 3 array of bps grid points
    :param nn_idx: N^3 array of nearest hand vertex indices for each bps point
    """
    # Convert hand mesh to open3d format
    o3d_hand_mesh = o3dmesh_from_trimesh(hand_mesh, color=[0.8, 0.8, 0.8])

    # For each hand vertex, find the nearest BPS point
    num_hand_verts = hand_mesh.vertices.shape[0]
    selected_bps_points = []
    corresponding_hand_verts = []

    for v_idx in range(num_hand_verts):
        # Find all BPS points that correspond to this hand vertex
        mask = (nn_idx == v_idx)
        if np.sum(mask) == 0:
            continue

        # Get the BPS points corresponding to this vertex
        corresponding_bps = bps[mask]  # M x 3

        # Calculate distances from these BPS points to the hand vertex
        hand_vert = hand_mesh.vertices[v_idx]  # 3
        distances = np.linalg.norm(corresponding_bps - hand_vert, axis=1)

        # Select the nearest BPS point
        nearest_idx = np.argmin(distances)
        selected_bps_points.append(corresponding_bps[nearest_idx])
        corresponding_hand_verts.append(hand_vert)

    # Convert selected BPS points to open3d point cloud
    geometries = [o3d_hand_mesh]

    if len(selected_bps_points) > 0:
        selected_bps_points = np.array(selected_bps_points)
        corresponding_hand_verts = np.array(corresponding_hand_verts)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(selected_bps_points)
        # Color the points red for visibility
        pcd.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(pcd)

        # Create arrows from BPS points to hand vertices
        for bps_pt, hand_vert in zip(selected_bps_points, corresponding_hand_verts):
            # Create line segment from BPS point to hand vertex
            points = np.array([bps_pt, hand_vert])
            lines = np.array([[0, 1]])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([0.0, 1.0, 0.0])  # Green arrows
            geometries.append(line_set)

    # Visualize
    o3d.visualization.draw_geometries(geometries,
                                     window_name="Hand Mesh with Nearest BPS Points and Arrows",
                                     width=1024,
                                     height=768)


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
