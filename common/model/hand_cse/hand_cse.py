"""
The Continuous Surface Embedding (CSE) for hand model
"""

import trimesh
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygeodesic.geodesic as geodesic
from common.manopth.manopth.manolayer import ManoLayer
from sklearn.decomposition import PCA
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm

class HandCSE(nn.Module):
    def __init__(self, n_verts=778, emb_dim=16, cano_verts=None, cano_faces=None):
        super(HandCSE, self).__init__()
        self.emb_dim = emb_dim
        self.n_verts = n_verts
        self.register_parameter('embedding_tensor', nn.Parameter(torch.randn(n_verts, emb_dim)))
        self.register_buffer("cano_faces", torch.as_tensor(cano_faces).long())
        # dist_mat_path = os.path.join('data', 'misc', 'hand_geo_dist_matrix.npy')
        # if os.path.exists(dist_mat_path):
        #     self.geo_dist_matrix = np.load(dist_mat_path)
        # else:
        #     self.geo_dist_matrix = self.geo_distance_matrix(cano_verts, cano_faces)
        #     np.save(dist_mat_path, self.geo_dist_matrix)
        
        # self.geo_dist_matrix = torch.as_tensor(self.geo_dist_matrix).float()

    def forward(self):
        Pemb = torch.exp(-torch.square(torch.cdist(self.embedding_tensor, self.embedding_tensor)))
        Pgt = torch.exp(-torch.square(self.geo_dist_matrix) / (2 * self.geo_dist_matrix.std()**2)).to(self.embedding_tensor.device)
        loss = F.binary_cross_entropy(Pemb, Pgt)
        return loss
    
    def vert2emb(self, verts_idx):
        """
        Get the CSE embedding features for given vertex indices
        Args:
            verts_idx: (B, ) tensor of vertex indices
        Returns:
            emb_features: (B, emb_dim) tensor of embedding features
        """
        emb_features = self.embedding_tensor[verts_idx]
        return emb_features
    
    def emb2Wvert(self, emb_features):
        """
        Find the nearest triangular surface and compute weights for each vertex on the triangle
        Args:
            emb_features: (B, n_pts, emb_dim) tensor of embedding features
        Returns:
            W: (B, n_pts, n_verts) tensor of weights for each vertex
        """
        triangle_cse = self.embedding_tensor[self.cano_faces].mean(dim=1)  # (n_faces, emb_dim)
        dists = torch.cdist(emb_features, triangle_cse)  # (B, n_pts, n_faces)
        face_idx = dists.argmin(dim=2)  # (B, n_pts)
        vert_idx = self.cano_faces[face_idx]  # (B, n_pts, 3)
        vert_cse = self.embedding_tensor[vert_idx]  # (B, n_pts, 3, emb_dim)
        # weights = F.softmax(-torch.cdist(emb_features.unsqueeze(2), vert_cse).squeeze(2) * 16, dim=2)  # (B, n_pts, 3)
        ## The weights should be inversely proportional to the distance in embedding space
        dists_to_verts = torch.cdist(emb_features.unsqueeze(2), vert_cse).squeeze(2)  # (B, n_pts, 3)
        weights = self.inv_proportional_weights(dists_to_verts)  # (B, n_pts, 3)
        W = torch.zeros(emb_features.shape[0], emb_features.shape[1], self.n_verts).to(emb_features.device)  # (B, n_pts, n_verts)
        W.scatter_add_(2, vert_idx, weights)
        # W = F.softmax(-dists*16, dim=2)
        return W

    
    @staticmethod
    def geo_distance_matrix(verts, faces):
        """
        Compute the geodesic distance matrix between hand vertices
        Args:
            hand_mesh: trimesh.Trimesh object of the hand mesh
        Returns:
            geo_dist_matrix: (N, N) numpy array of geodesic distances
        """
        geoalg = geodesic.PyGeodesicAlgorithmExact(verts, faces)
        dists = []
        for i in range(verts.shape[0]):
            src_idx = np.array([i])
            dist, _ = geoalg.geodesicDistances(src_idx)
            dists.append(dist)
        
        geo_dist_matrix = np.stack(dists, axis=0)

        return geo_dist_matrix
    
    @staticmethod
    def inv_proportional_weights(distances):
        """
        Compute weights inversely proportional to distances
        Args:
            distances: (B, n_pts, 3) tensor of distances
        Returns:
            weights: (B, n_pts, 3) tensor of weights
        """
        d1, d2, d3 = distances.unbind(dim=2) # (B, n_pts)
        bottom = d1 * d2 + d2 * d3 + d3 * d1  # (B, n_pts)
        w1 = (d2 * d3) / (bottom + 1e-8)
        w2 = (d3 * d1) / (bottom + 1e-8)
        w3 = (d1 * d2) / (bottom + 1e-8)
        weights = torch.stack([w1, w2, w3], dim=2)  # (B, n_pts, 3)
        return weights

def train():
    device = 'cuda:0'
    mano_layer = ManoLayer(mano_root='data/mano/models', use_pca=False, side='right', flat_hand_mean=True)
    handV, handJ, _ = mano_layer(th_pose_coeffs=torch.zeros(1, 48))
    handF = mano_layer.th_faces
    handcse = HandCSE(cano_verts=handV[0].cpu().numpy(), cano_faces=handF.cpu().numpy(), emb_dim=16).to(device)
    optimizer = torch.optim.Adam(handcse.parameters(), lr=1e-3)
    n_iters = 10000
    for it in range(n_iters):
        optimizer.zero_grad()
        loss = handcse()
        loss.backward()
        optimizer.step()

        if it % 100 == 0:
            print(f"Iter {it}/{n_iters}, Loss: {loss.item():.6f}")

    handcse.eval()

    # Save model checkpoint
    checkpoint = {
        'state_dict': handcse.state_dict(),
        'emb_dim': handcse.emb_dim,
    }
    checkpoint_path = os.path.join('data', 'misc', 'hand_cse.ckpt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")


def visualize_distance_from_point(vertices, faces, cse_features, source_idx, colormap='jet', temperature=10.0):
    """
    Visualize the embedding distance from a source point using heatmap colors

    Args:
        vertices: (N, 3) array of vertex positions
        faces: (F, 3) array of face indices
        cse_features: (N, D) array of CSE embedding features
        source_idx: index of the source vertex
        colormap: matplotlib colormap name (e.g., 'jet', 'hot', 'viridis')
        temperature: temperature parameter for softmax (higher = smoother)
    """
    # Compute Euclidean distances in embedding space
    source_embedding = cse_features[source_idx:source_idx+1]  # (1, D)
    distances = np.linalg.norm(cse_features - source_embedding, axis=1)  # (N,)

    # Convert distances to probabilities using softmax with temperature
    # p = softmax(-distance * t)
    logits = -distances * temperature
    exp_logits = np.exp(logits - logits.max())  # Subtract max for numerical stability
    probabilities = exp_logits / exp_logits.sum()  # (N,)

    # Normalize probabilities to [0, 1] for colormap
    prob_normalized = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min() + 1e-8)

    # Apply colormap (higher probability = closer = hotter color)
    cmap = cm.get_cmap(colormap)
    colors = cmap(prob_normalized)[:, :3]  # (N, 3) RGB values

    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()

    # Create a sphere at the source point
    source_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    source_sphere.translate(vertices[source_idx])
    source_sphere.paint_uniform_color([0, 0, 0])  # Black sphere

    return mesh, source_sphere, distances, probabilities


def test():
    # Load checkpoint
    checkpoint_path = os.path.join('data', 'misc', 'hand_cse.ckpt')
    checkpoint = torch.load(checkpoint_path)

    mano_layer = ManoLayer(mano_root='data/mano/models', use_pca=False, side='right', flat_hand_mean=True)
    handV, handJ, _ = mano_layer(th_pose_coeffs=torch.zeros(1, 48))
    handF = mano_layer.th_faces

    # Initialize model and load state
    handcse = HandCSE(n_verts=778, emb_dim=checkpoint['emb_dim'], cano_faces=handF.cpu().numpy()).to('cpu')
    handcse.load_state_dict(checkpoint['state_dict'])
    handcse.eval()

    # Convert to numpy
    vertices = handV[0].cpu().numpy()
    faces = handF.cpu().numpy()
    cse_features = handcse.embedding_tensor.detach().cpu().numpy()  # Shape: (778, 16)

    # ============================================
    # Part 1: General PCA visualization
    # ============================================
    print("=" * 50)
    print("Part 1: General PCA-based visualization")
    print("=" * 50)

    # Use PCA to reduce 16D features to 3D for RGB color mapping
    pca = PCA(n_components=3)
    cse_3d = pca.fit_transform(cse_features)  # Shape: (778, 3)

    # Normalize to [0, 1] range for RGB colors
    cse_colors = (cse_3d - cse_3d.min(axis=0)) / (cse_3d.max(axis=0) - cse_3d.min(axis=0))

    # Create Open3D mesh
    mesh_pca = o3d.geometry.TriangleMesh()
    mesh_pca.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_pca.triangles = o3d.utility.Vector3iVector(faces)
    mesh_pca.vertex_colors = o3d.utility.Vector3dVector(cse_colors)
    mesh_pca.compute_vertex_normals()

    # Print some statistics
    print(f"CSE tensor shape: {cse_features.shape}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # Visualize
    o3d.visualization.draw_geometries([mesh_pca], window_name="Hand CSE - PCA Visualization")

    # ============================================
    # Part 2: Distance heatmap from random points
    # ============================================
    print("\n" + "=" * 50)
    print("Part 2: Distance heatmap from random points")
    print("=" * 50)

    # Pick random source points
    np.random.seed(0)
    n_samples = 5
    source_indices = np.random.choice(len(vertices), size=n_samples, replace=False)

    print(f"Visualizing embedding distances from {n_samples} random points...")
    print(f"Source vertex indices: {source_indices}")

    # Visualize each source point
    for i, source_idx in enumerate(source_indices):
        mesh_heatmap, source_sphere, distances, probabilities = visualize_distance_from_point(
            vertices, faces, cse_features, source_idx, colormap='jet', temperature=10.0
        )

        print(f"\nSource point {i+1}/{n_samples} (vertex {source_idx}):")
        print(f"  Position: {vertices[source_idx]}")
        print(f"  Distance - Min: {distances.min():.4f}, Max: {distances.max():.4f}, Mean: {distances.mean():.4f}")
        print(f"  Probability - Min: {probabilities.min():.6f}, Max: {probabilities.max():.6f}, Sum: {probabilities.sum():.6f}")

        # Visualize
        o3d.visualization.draw_geometries(
            [mesh_heatmap, source_sphere],
            window_name=f"Distance Heatmap from Vertex {source_idx} ({i+1}/{n_samples})"
        )


if __name__ == '__main__':
    # train()
    test()