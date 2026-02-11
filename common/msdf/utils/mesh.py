import trimesh
from trimesh import Trimesh, Scene
import numpy as np

def normalize_mesh(mesh: Trimesh) -> Trimesh:
    """
    Normalize the mesh to have a unit bounding box [-1, 1] in all three dimensions.
    :param mesh:
    :return: normalized mesh
    """
    min_bound, max_bound = mesh.bounds
    center = (min_bound + max_bound) / 2
    scale = max(max_bound - min_bound) / 2
    mesh.vertices -= center
    mesh.vertices /= scale
    return mesh


def load_mesh(file_path: str) -> Trimesh:
    mesh = trimesh.load(file_path, process=False)
    if isinstance(mesh, Scene):
        scene = mesh
        meshes = []
        for name, val in scene.geometry.items():
            meshes.append(val)
        mesh = trimesh.util.concatenate(meshes)

    mesh = normalize_mesh(mesh)
    return mesh


def farthest_point_sampling(points, num_samples):
    """
    Farthest Point Sampling: iteratively select the point that is farthest
    from the already-selected set.
    points: numpy array of shape (N, 3)
    num_samples: the number of points to sample
    """
    N = points.shape[0]
    if num_samples >= N:
        return points.copy()

    selected = np.zeros(num_samples, dtype=np.int64)
    # Start from a random point
    selected[0] = np.random.randint(N)
    # min_dist[i] = min distance from points[i] to any selected point so far
    min_dist = np.full(N, np.inf)

    for i in range(1, num_samples):
        dist = np.linalg.norm(points - points[selected[i - 1]], axis=1)
        min_dist = np.minimum(min_dist, dist)
        selected[i] = np.argmax(min_dist)

    return points[selected]
