"""
Profile each step of _project_latent / contact_grid_projection
to find the bottleneck in the ~0.9s overhead.
"""
import time
import numpy as np
import torch
from common.manopth.manopth.manolayer import ManoLayer
from common.msdf.utils.msdf import (
    calc_local_grid_all_pts_gpu,
    get_grid,
)
from common.model.handobject import recover_hand_verts_from_contact
from common.model.hand_cse.hand_cse import HandCSE


def timeit(name, fn, n_repeat=3):
    """Run fn, sync CUDA, report time."""
    # warmup
    result = fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = np.mean(times)
    print(f"  {name:45s} {avg*1000:>8.1f} ms")
    return result


def main():
    torch.manual_seed(42)
    device = "cuda"
    n_samples = 20
    n_grids = 128
    K = 8
    grid_scale = 0.02
    cse_dim = 16

    # Load MANO
    mano_layer = ManoLayer(
        mano_root="data/misc/mano_v1_2/models",
        use_pca=False, ncomps=45, flat_hand_mean=True
    ).to(device)
    hand_faces = mano_layer.th_faces

    # Load hand CSE embedding
    hand_cse = torch.load("common/model/hand_cse/hand_cse_16.pt", weights_only=False).to(device)  # (778, 16)
    cse_dim = hand_cse.shape[-1]

    # Generate fake hand data
    with torch.no_grad():
        pose = torch.randn(n_samples, 48, device=device) * 0.1
        betas = torch.randn(n_samples, 10, device=device) * 0.5
        trans = torch.randn(n_samples, 3, device=device) * 0.05
        handV, _, _ = mano_layer(pose, th_betas=betas, th_trans=trans)

    # Simulate grid data
    grid_centers = handV[:, :n_grids, :]  # (B, N, 3)
    normalized_coords = get_grid(kernel_size=K, device=device).reshape(-1, 3).float()
    grid_coords = grid_centers[:, :, None, :] + normalized_coords[None, None, :, :] * grid_scale
    grid_coords = grid_coords.reshape(n_samples, -1, 3)
    grid_contact = torch.rand(n_samples, n_grids * K**3, device=device) * 0.1
    grid_cse_data = torch.randn(n_samples, n_grids * K**3, cse_dim, device=device) * 0.1

    print(f"Setup: n_samples={n_samples}, n_grids={n_grids}, K={K}")
    print(f"Total query points per sample: {n_grids * K**3}")
    print(f"\nProfiling steps:\n")

    # Step 1: MANO forward
    def step_mano():
        return mano_layer(pose, th_betas=betas, th_trans=trans)
    timeit("mano_layer forward", step_mano)

    # Step 2: calc_local_grid_all_pts_gpu (loop over B)
    def step_kaolin_loop():
        for b in range(n_samples):
            calc_local_grid_all_pts_gpu(
                grid_centers[b], normalized_coords, handV[b], hand_faces, K, grid_scale
            )
    timeit("calc_local_grid_all_pts_gpu (loop B)", step_kaolin_loop)

    # Step 3: Barycentric CSE with torch.linalg.inv (loop over B)
    # First, check active grids
    gd0, vm0, gm0, hd0, nfi0, npt0 = calc_local_grid_all_pts_gpu(
        grid_centers[0], normalized_coords, handV[0], hand_faces, K, grid_scale
    )
    M = gm0.sum().item()
    print(f"\n  (Active grids per sample: M={M}/{n_grids}, total pts: {M * K**3})\n")

    def step_bary_inv_loop():
        for b in range(n_samples):
            gd, vm, gm, hd, nfi, npt = calc_local_grid_all_pts_gpu(
                grid_centers[b], normalized_coords, handV[b], hand_faces, K, grid_scale
            )
            if gm.any():
                nfi_f = nfi.reshape(-1)
                npt_f = npt.reshape(-1, 3)
                nvi = hand_faces[nfi_f]
                fv = handV[b, nvi]
                fc = hand_cse[nvi]
                w = torch.linalg.inv(fv.transpose(1, 2)) @ npt_f.unsqueeze(-1)
                w = torch.clamp(w, 0, 1)
                w = w / (torch.sum(w, dim=1, keepdim=True) + 1e-8)
                _ = torch.sum(fc * w, dim=1)
    timeit("kaolin + bary inv (loop B)", step_bary_inv_loop)

    # Step 4: Just the inv part on single sample
    nfi_f = nfi0.reshape(-1)
    npt_f = npt0.reshape(-1, 3)
    nvi = hand_faces[nfi_f]
    fv = handV[0, nvi]
    fc = hand_cse[nvi]

    def step_inv_only():
        w = torch.linalg.inv(fv.transpose(1, 2)) @ npt_f.unsqueeze(-1)
        w = torch.clamp(w, 0, 1)
        w = w / (torch.sum(w, dim=1, keepdim=True) + 1e-8)
        return torch.sum(fc * w, dim=1)
    timeit(f"torch.linalg.inv only (1 sample, {M*K**3})", step_inv_only)

    # Step 5: solve instead of inv
    def step_solve_only():
        w = torch.linalg.solve(fv.transpose(1, 2), npt_f.unsqueeze(-1))
        w = torch.clamp(w, 0, 1)
        w = w / (torch.sum(w, dim=1, keepdim=True) + 1e-8)
        return torch.sum(fc * w, dim=1)
    timeit(f"torch.linalg.solve only (1 sample, {M*K**3})", step_solve_only)

    # Step 6: Use Cramer's rule (no matrix ops)
    def step_cramer_only():
        # fv: (P, 3, 3), npt_f: (P, 3)
        a = fv[:, 0, :]  # (P, 3) — vertex 0
        b_ = fv[:, 1, :]
        c = fv[:, 2, :]
        # Barycentric via area ratios
        v0 = b_ - a
        v1 = c - a
        v2 = npt_f - a
        d00 = (v0 * v0).sum(-1)
        d01 = (v0 * v1).sum(-1)
        d11 = (v1 * v1).sum(-1)
        d20 = (v2 * v0).sum(-1)
        d21 = (v2 * v1).sum(-1)
        denom = (d00 * d11 - d01 * d01).clamp(min=1e-12)
        bary_v = (d11 * d20 - d01 * d21) / denom
        bary_w = (d00 * d21 - d01 * d20) / denom
        bary_u = 1.0 - bary_v - bary_w
        # Clamp
        bary_u = bary_u.clamp(0, 1)
        bary_v = bary_v.clamp(0, 1)
        bary_w = bary_w.clamp(0, 1)
        s = (bary_u + bary_v + bary_w).clamp(min=1e-8)
        bary_u = bary_u / s
        bary_v = bary_v / s
        bary_w = bary_w / s
        w = torch.stack([bary_u, bary_v, bary_w], dim=-1).unsqueeze(-1)  # (P, 3, 1)
        return torch.sum(fc * w, dim=1)
    timeit(f"cramer's rule only (1 sample, {M*K**3})", step_cramer_only)

    # Full loop comparison with Cramer
    def step_kaolin_cramer_loop():
        for b in range(n_samples):
            gd, vm, gm, hd, nfi, npt = calc_local_grid_all_pts_gpu(
                grid_centers[b], normalized_coords, handV[b], hand_faces, K, grid_scale
            )
            if gm.any():
                nfi_f = nfi.reshape(-1)
                npt_f = npt.reshape(-1, 3)
                nvi = hand_faces[nfi_f]
                fv = handV[b, nvi]
                fc = hand_cse[nvi]
                a = fv[:, 0, :]
                b_ = fv[:, 1, :]
                c = fv[:, 2, :]
                v0 = b_ - a
                v1 = c - a
                v2 = npt_f - a
                d00 = (v0 * v0).sum(-1)
                d01 = (v0 * v1).sum(-1)
                d11 = (v1 * v1).sum(-1)
                d20 = (v2 * v0).sum(-1)
                d21 = (v2 * v1).sum(-1)
                denom = (d00 * d11 - d01 * d01).clamp(min=1e-12)
                bary_v = (d11 * d20 - d01 * d21) / denom
                bary_w = (d00 * d21 - d01 * d20) / denom
                bary_u = 1.0 - bary_v - bary_w
                bary_u = bary_u.clamp(0, 1)
                bary_v = bary_v.clamp(0, 1)
                bary_w = bary_w.clamp(0, 1)
                s = (bary_u + bary_v + bary_w).clamp(min=1e-8)
                w = torch.stack([bary_u/s, bary_v/s, bary_w/s], dim=-1).unsqueeze(-1)
                _ = torch.sum(fc * w, dim=1)
    timeit("kaolin + cramer (loop B)", step_kaolin_cramer_loop)


if __name__ == "__main__":
    main()
