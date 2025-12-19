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