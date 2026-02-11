"""
Test script for evaluating StableLoss on GRAB dataset ground-truth grasps.
Loads data through HOIDatasetModule and HandObject, computes the local grid
contact online, and prints per-sample stable loss values.
"""
import sys

import trimesh

from common.utils.physics import StableLoss
sys.path.insert(0, '.')

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import open3d as o3d

from common.dataset_utils.datamodules import HOIDatasetModule
from common.model.handobject import HandObject


@hydra.main(config_path="../config", config_name="mlcdiff", version_base=None)
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build data module and prepare data (ensures MSDF npz files exist)
    dm = HOIDatasetModule(cfg)
    dm.prepare_data()
    dm.setup(stage='fit')

    loader = dm.train_dataloader()

    stable_loss = StableLoss()
    # Instantiate HandObject
    ho = HandObject(cfg.data, device=device, normalize=True)
    ho.stable_loss = stable_loss

    K = cfg.data.msdf.kernel_size
    K3 = K ** 3
    # optimizer = torch.optim.Adam(ho.stable_loss.parameters(), lr=1e-3)

    all_losses = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating StableLoss")):
        ho.load_from_batch(batch)

        # Flatten ml_contact from (B, N, K, K, K, 1+cse_dim) to (B, N*K^3)
        B, N = ho.obj_msdf.shape[:2]
        gt_contact = ho.ml_contact[:, :, :, :, :, 0]       # (B, N, K, K, K) — contact channel
        gt_contact = gt_contact.reshape(B, N * K3)          # (B, N*K^3)

        # Call StableLoss via HandObject helper
        loss = ho.calculate_stable_loss(gt_contact)
        # optimizer.zero_grad()
        # loss.mean().backward()
        # optimizer.step()

        # loss shape: (num_masked_points,) — one scalar per sample after masking
        if loss.numel() > 0:
            mean_loss = loss.mean().item()
        else:
            mean_loss = 0.0

        all_losses.append(mean_loss)

        obj_templates = [trimesh.Trimesh(dm.val_set.obj_info[name]['verts'], dm.val_set.obj_info[name]['faces'])
                         for name in batch['objName']]

        if mean_loss > 0:
            all_geoms = []
            valid_idx = torch.where(loss == 0)[0][0].item()
            all_geoms.extend(ho.vis_local_grid_with_hand(obj_templates=obj_templates, idx=valid_idx))
            idxs = torch.where(loss > 0)[0].tolist()  # Get the indices of all non-zero loss samples
            for i, idx in enumerate(idxs):  # Visualize only the first non-zero loss sample
                geoms = ho.vis_local_grid_with_hand(obj_templates=obj_templates, idx=idx)
                all_geoms.extend([g.translate((-0.4 + 0.25* (i // 3), -0.4 + 0.25 * (i % 3), 0)) for g in geoms])

            print(loss[idx])
            o3d.visualization.draw_geometries(all_geoms)

        # _, vis_geoms = ho.vis_all_grid_points(obj_templates=obj_templates, hue='overlap')
        # o3d.visualization.draw(vis_geoms, show_skybox=False)

        if (batch_idx + 1) % 10 == 0:
            running_avg = sum(all_losses) / len(all_losses)
            print(f"  Batch {batch_idx+1}: loss={mean_loss:.6f}  running_avg={running_avg:.6f}")
            print(f'  Stable rate: {(loss < 0.1).float().mean().item() * 100:.2f}%')
            print(f"  StableLoss k parameter: {ho.stable_loss.k:.6f}")

    avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
    print(f"\n{'='*50}")
    print(f"Total batches: {len(all_losses)}")
    print(f"Average StableLoss: {avg_loss:.6f}")
    print(f"Min: {min(all_losses):.6f}  Max: {max(all_losses):.6f}")


if __name__ == '__main__':
    main()
