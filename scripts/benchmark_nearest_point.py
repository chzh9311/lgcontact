"""
Benchmark nearest point query: trimesh (CPU) vs Kaolin (GPU).
Uses realistic MANO hand mesh (778 verts, 1538 faces),
128 contact points * 8^3 grid = 65536 query points, batch_size=20.
"""
import time
import numpy as np
import torch
import trimesh

from common.manopth.manopth.manolayer import ManoLayer
from common.msdf.utils.msdf import (
    nn_dist_to_mesh,
    nn_dist_to_mesh_gpu,
    calc_local_grid_all_pts,
    calc_local_grid_all_pts_gpu,
    get_grid,
)


def benchmark_nn_dist(hand_verts_np, hand_faces_np, hand_verts_gpu, hand_faces_gpu, n_queries=65536):
    """Benchmark raw nearest-point query (no grid logic)."""
    query_pts_np = (np.random.randn(n_queries, 3) * 0.1).astype(np.float32)
    query_pts_gpu = torch.tensor(query_pts_np, device="cuda")

    mesh = trimesh.Trimesh(vertices=hand_verts_np, faces=hand_faces_np, process=False)

    # Warmup GPU
    nn_dist_to_mesh_gpu(query_pts_gpu[:100], hand_verts_gpu, hand_faces_gpu)
    torch.cuda.synchronize()

    # --- CPU (trimesh) ---
    t0 = time.perf_counter()
    dist_cpu, face_cpu, cp_cpu = nn_dist_to_mesh(query_pts_np, mesh)
    cpu_time = time.perf_counter() - t0

    # --- GPU (Kaolin) ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dist_gpu, face_gpu, cp_gpu = nn_dist_to_mesh_gpu(query_pts_gpu, hand_verts_gpu, hand_faces_gpu)
    torch.cuda.synchronize()
    gpu_time = time.perf_counter() - t0

    print(f"\n=== Raw nearest-point query ({n_queries} points, 1538 faces) ===")
    print(f"  CPU (trimesh):  {cpu_time*1000:>10.1f} ms")
    print(f"  GPU (Kaolin):   {gpu_time*1000:>10.1f} ms")
    print(f"  Speedup:        {cpu_time/gpu_time:>10.1f}x")

    # Verify correctness
    dist_diff = np.abs(dist_cpu - dist_gpu.cpu().numpy())
    cp_diff = np.linalg.norm(cp_cpu - cp_gpu.cpu().numpy(), axis=-1)
    print(f"  Distance max err:  {dist_diff.max():.2e}")
    print(f"  Closest pt max err: {cp_diff.max():.2e}")


def benchmark_calc_local_grid(hand_verts_np, hand_faces_np, hand_verts_gpu, hand_faces_gpu,
                               n_contacts=128, kernel_size=8, grid_scale=0.02):
    """Benchmark full calc_local_grid_all_pts pipeline."""
    K3 = kernel_size ** 3
    normalized_coords_np = get_grid(kernel_size).reshape(-1, 3).numpy().astype(np.float32)
    normalized_coords_gpu = torch.tensor(normalized_coords_np, device="cuda")

    # Generate contact points near the hand
    contact_pts_np = (hand_verts_np[np.random.choice(len(hand_verts_np), n_contacts)] +
                      np.random.randn(n_contacts, 3).astype(np.float32) * 0.01)
    contact_pts_gpu = torch.tensor(contact_pts_np, device="cuda")

    mesh = trimesh.Trimesh(vertices=hand_verts_np, faces=hand_faces_np, process=False)

    # Warmup GPU
    calc_local_grid_all_pts_gpu(
        contact_pts_gpu[:4], normalized_coords_gpu, hand_verts_gpu, hand_faces_gpu, kernel_size, grid_scale
    )
    torch.cuda.synchronize()

    # --- CPU ---
    t0 = time.perf_counter()
    res_cpu = calc_local_grid_all_pts(
        contact_pts_np, normalized_coords_np, obj_mesh=None, hand_mesh=mesh,
        kernel_size=kernel_size, grid_scale=grid_scale
    )
    cpu_time = time.perf_counter() - t0

    # --- GPU ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res_gpu = calc_local_grid_all_pts_gpu(
        contact_pts_gpu, normalized_coords_gpu, hand_verts_gpu, hand_faces_gpu, kernel_size, grid_scale
    )
    torch.cuda.synchronize()
    gpu_time = time.perf_counter() - t0

    n_total = n_contacts * K3
    print(f"\n=== calc_local_grid_all_pts ({n_contacts} centers x {kernel_size}^3 = {n_total} queries) ===")
    print(f"  CPU (trimesh):  {cpu_time*1000:>10.1f} ms")
    print(f"  GPU (Kaolin):   {gpu_time*1000:>10.1f} ms")
    print(f"  Speedup:        {cpu_time/gpu_time:>10.1f}x")

    return cpu_time, gpu_time


def benchmark_batched(hand_faces_gpu, mano_layer, batch_size=20, n_contacts=128, kernel_size=8, grid_scale=0.02):
    """Benchmark batched scenario: batch_size hands, each with n_contacts grids."""
    K3 = kernel_size ** 3
    normalized_coords_gpu = get_grid(kernel_size, device="cuda").reshape(-1, 3).float()

    # Generate batch of random MANO hands
    with torch.no_grad():
        pose = torch.randn(batch_size, 48, device="cuda") * 0.1
        betas = torch.randn(batch_size, 10, device="cuda") * 0.5
        trans = torch.randn(batch_size, 3, device="cuda") * 0.05
        handV, _, _ = mano_layer(pose, th_betas=betas, th_trans=trans)

    # Generate contact points near each hand
    contact_pts = []
    for b in range(batch_size):
        verts = handV[b]
        idx = torch.randint(0, verts.shape[0], (n_contacts,), device="cuda")
        pts = verts[idx] + torch.randn(n_contacts, 3, device="cuda") * 0.01
        contact_pts.append(pts)

    # Warmup
    calc_local_grid_all_pts_gpu(
        contact_pts[0][:4], normalized_coords_gpu, handV[0], hand_faces_gpu, kernel_size, grid_scale
    )
    torch.cuda.synchronize()

    # --- GPU batched ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for b in range(batch_size):
        calc_local_grid_all_pts_gpu(
            contact_pts[b], normalized_coords_gpu, handV[b], hand_faces_gpu, kernel_size, grid_scale
        )
    torch.cuda.synchronize()
    gpu_time = time.perf_counter() - t0

    n_total = batch_size * n_contacts * K3
    print(f"\n=== Batched ({batch_size} hands x {n_contacts} centers x {kernel_size}^3 = {n_total} queries) ===")
    print(f"  GPU total:      {gpu_time*1000:>10.1f} ms")
    print(f"  GPU per sample: {gpu_time/batch_size*1000:>10.1f} ms")

    # --- CPU batched (estimate from single) ---
    mesh = trimesh.Trimesh(vertices=handV[0].cpu().numpy(), faces=hand_faces_gpu.cpu().numpy(), process=False)
    t0 = time.perf_counter()
    calc_local_grid_all_pts(
        contact_pts[0].cpu().numpy(),
        normalized_coords_gpu.cpu().numpy(),
        obj_mesh=None, hand_mesh=mesh,
        kernel_size=kernel_size, grid_scale=grid_scale,
    )
    cpu_single = time.perf_counter() - t0
    cpu_est = cpu_single * batch_size

    print(f"  CPU estimated:  {cpu_est*1000:>10.1f} ms  (single={cpu_single*1000:.1f} ms x {batch_size})")
    print(f"  Speedup:        {cpu_est/gpu_time:>10.1f}x")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # Load MANO
    mano_layer = ManoLayer(
        mano_root="data/misc/mano_v1_2/models",
        use_pca=False, ncomps=45, flat_hand_mean=True
    ).cuda()

    hand_faces_gpu = mano_layer.th_faces  # (1538, 3)

    # Generate a sample hand
    with torch.no_grad():
        pose = torch.zeros(1, 48, device="cuda")
        betas = torch.zeros(1, 10, device="cuda")
        trans = torch.zeros(1, 3, device="cuda")
        handV, _, _ = mano_layer(pose, th_betas=betas, th_trans=trans)

    hand_verts_gpu = handV[0]  # (778, 3)
    hand_verts_np = hand_verts_gpu.cpu().numpy()
    hand_faces_np = hand_faces_gpu.cpu().numpy()

    print(f"MANO mesh: {hand_verts_np.shape[0]} verts, {hand_faces_np.shape[0]} faces")
    print(f"Query setup: 128 contact points x 8^3 grid = {128 * 512} points per sample")

    # 1. Raw nearest-point
    benchmark_nn_dist(hand_verts_np, hand_faces_np, hand_verts_gpu, hand_faces_gpu, n_queries=128 * 512)

    # 2. Full local grid pipeline (single sample)
    benchmark_calc_local_grid(hand_verts_np, hand_faces_np, hand_verts_gpu, hand_faces_gpu)

    # 3. Batched (batch_size=20)
    benchmark_batched(hand_faces_gpu, mano_layer, batch_size=20)


if __name__ == "__main__":
    main()
