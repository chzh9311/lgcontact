"""
Bayesian hyperparameter search for the hybrid pose optimizer.

Searches over:
  - w_repulsive       (log-uniform, 1e-4 … 1.0)
  - w_regularization  (log-uniform, 1e-4 … 1.0)
  - contact_th        (uniform,     0.01 … 0.5)

Objective: maximise Contact Ratio − Penetration Depth  (customise as needed).

Usage:
    conda run -n hoi_common python scripts/tune_pose_optimizer.py \
        n_trials=50 val_batches=20 run_phase=test
"""
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import hydra
import optuna
import trimesh
import numpy as np
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from common.dataset_utils.datamodules import HOIDatasetModule
from common.model.gridae.gridae import GRIDAEResidual
from common.model.vae.handvae import HandVAE
from common.model.graspdifftrainer import GraspDiffTrainer
from common.model.handobject import HandObject, recover_hand_verts_from_contact
from common.model.pose_optimizer import optimize_pose_wrt_local_grids
from common.evaluation.eval_fns import calculate_metrics
from common.utils.misc import load_pl_ckpt
import common.model.diff.mdm.gaussian_diffusion as mdm_gd
from common.model.diff.unet import DualUNetModel

OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)


def build_model(cfg):
    """Load the full GraspDiffTrainer from checkpoint (weights only, no Lightning Trainer)."""
    gridae  = GRIDAEResidual(cfg.ae)
    hand_ae = HandVAE(cfg.hand_ae)

    mdm_cfg   = cfg.generator.mdm
    diffusion = mdm_gd.space_timestep_diffusion(**mdm_gd.mdm_defaults(mdm_cfg))
    model     = DualUNetModel(cfg.generator.unet)

    trainer_model = GraspDiffTrainer(
        cfg, model=model, diffusion=diffusion,
        grid_ae=gridae, hand_ae=hand_ae,
    )

    ckpt = torch.load(cfg.ckpt_path, map_location='cpu')
    load_pl_ckpt(trainer_model, ckpt['state_dict'])
    trainer_model.eval()
    return trainer_model


def evaluate(trainer_model, dm, cfg, device):
    """
    Run inference on up to n_batches test samples and return per-sample metrics.
    Returns a dict of metric_name -> np.ndarray (one value per sample).
    """
    test_set = dm.test_set
    loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dm.collate_fn,
    )

    all_results = []
    mano_layer = trainer_model.mano_layer.to(device)

    for batch_idx, batch in enumerate(loader):

        obj_name     = batch['objName'][0]
        obj_mesh_dict = test_set.simp_obj_mesh
        obj_hulls    = test_set.obj_hulls[obj_name]
        obj_mesh     = trimesh.Trimesh(obj_mesh_dict[obj_name]['verts'],
                                       obj_mesh_dict[obj_name]['faces'])

        handobject = HandObject(cfg.data, device, mano_layer=mano_layer, normalize=True)
        n_samples  = cfg.test.n_samples
        handobject.load_from_batch_obj_only(batch, n_samples,
                                            obj_template=obj_mesh, obj_hulls=obj_hulls)

        msdf_k          = trainer_model.msdf_k
        n_grids         = batch['objMsdf'].shape[1]
        obj_msdf_grid   = handobject.obj_msdf[:, :, :msdf_k**3].view(-1, 1, msdf_k, msdf_k, msdf_k)
        obj_msdf_center = handobject.obj_msdf[:, :, msdf_k**3:]

        with torch.no_grad():
            obj_feat, multi_scale_obj_cond = trainer_model.grid_ae.encode_object(obj_msdf_grid.to(device))
            obj_pc      = torch.cat([obj_msdf_center, obj_feat.unsqueeze(0)], dim=-1).to(device)
            cat_noise   = torch.randn(n_samples,
                                      n_grids * cfg.ae.feat_dim + trainer_model.hand_ae.latent_dim,
                                      device=device)
            input_data  = {'x': cat_noise,
                           'obj_pc': obj_pc.permute(0, 2, 1),
                           'obj_msdf': handobject.obj_msdf.to(device)}
            samples     = trainer_model.diffusion.sample(trainer_model.model, input_data,
                                                         k=n_samples, proj_fn=None, progress=False)

        hand_latent, grid_latent = (samples[:, :cfg.generator.unet.d_y],
                                    samples[:, cfg.generator.unet.d_y:].view(n_samples, n_grids, -1))

        grid_latent        = grid_latent.reshape(n_samples * n_grids, -1)
        ms_cond_rep        = [c.repeat(n_samples, 1, 1, 1, 1) for c in multi_scale_obj_cond]
        ms_cond_rep.append(obj_feat.repeat(n_samples, 1))

        with torch.no_grad():
            recon_lg_contact = trainer_model.grid_ae.decode(grid_latent, ms_cond_rep)
        recon_lg_contact = recon_lg_contact.permute(0, 2, 3, 4, 1)
        recon_lg_contact = recon_lg_contact.view(n_samples, n_grids, msdf_k, msdf_k, msdf_k, -1)
        recon_lg_contact[..., 0][recon_lg_contact[..., 0] < cfg.pose_optimizer.contact_th] = 0

        pred_grid_contact = recon_lg_contact[..., 0].reshape(n_samples, -1)
        pred_grid_cse     = recon_lg_contact[..., 1:].reshape(n_samples, -1, trainer_model.cse_dim)

        grid_coords = trainer_model.grid_coords  # (K^3, 3) normalized
        scale       = cfg.msdf.scale
        centers_exp = obj_msdf_center.to(device).unsqueeze(2).expand(-1, -1, msdf_k**3, -1)
        grid_coords_world = (centers_exp + grid_coords[None, None] * scale).reshape(n_samples, -1, 3)

        pred_targetWverts = trainer_model.hand_cse.emb2Wvert(pred_grid_cse)

        recon_hand_verts, recon_verts_mask = recover_hand_verts_from_contact(
            trainer_model.hand_cse, None,
            pred_grid_contact, pred_grid_cse,
            grid_coords=grid_coords_world, chunk_size=10,
        )
        recon_params, _, _ = trainer_model.hand_ae.decode(hand_latent)

        with torch.enable_grad():
            params, _ = optimize_pose_wrt_local_grids(
                mano_layer,
                grid_centers=obj_msdf_center.to(device),
                target_pts=grid_coords_world,
                target_W_verts=pred_targetWverts,
                weights=pred_grid_contact,
                grid_sdfs=obj_msdf_grid.squeeze(1).to(device),
                dist2contact_fn=trainer_model.grid_dist_to_contact,
                recon_hand_verts=recon_hand_verts,
                recon_verts_mask=recon_verts_mask,
                n_iter=cfg.pose_optimizer.n_opt_iter,
                lr=cfg.pose_optimizer.opt_lr,
                grid_scale=scale,
                w_repulsive=cfg.pose_optimizer.w_repulsive,
                w_reg_loss=cfg.pose_optimizer.w_regularization,
                init_pose=recon_params,
            )
        mano_trans, global_pose, mano_pose, mano_shape = params
        handV, handJ, _ = mano_layer(torch.cat([global_pose, mano_pose], dim=1),
                                     th_betas=mano_shape, th_trans=mano_trans)
        handV = handV.detach().cpu().numpy()
        handJ = handJ.detach().cpu().numpy()

        closed_faces = trainer_model.closed_mano_faces
        param_list = [
            {'dataset_name': 'grab',
             'frame_name': f"{obj_name}_{i}",
             'hand_model': trimesh.Trimesh(handV[i], closed_faces),
             'obj_name': obj_name,
             'hand_joints': handJ[i],
             'obj_model': handobject.obj_models[0],
             'obj_hulls': handobject.obj_hulls[0],
             'idx': i}
            for i in range(handV.shape[0])
        ]
        result = calculate_metrics(param_list, metrics=cfg.test.criteria, reduction='none')
        all_results.append(result)

    # Aggregate: concatenate per-metric arrays across batches
    aggregated = {}
    for key in all_results[0]:
        aggregated[key] = np.concatenate([r[key] for r in all_results])
    return aggregated


@hydra.main(version_base=None, config_path="../config", config_name="mlcdiff")
def main(cfg: DictConfig):
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_trials   = cfg.get('n_trials', 50)
    val_batches = cfg.get('val_batches', 20)

    print(f"Building model from {cfg.ckpt_path} ...")
    trainer_model = build_model(cfg).to(device)

    print("Setting up test datamodule ...")
    dm = HOIDatasetModule(cfg)
    dm.prepare_data()
    dm.setup('validate')

    def objective(trial: optuna.Trial) -> float:
        w_rep = trial.suggest_float('w_repulsive',      1e-4, 1.0, log=True)
        w_reg = trial.suggest_float('w_regularization', 1e-4, 1.0, log=True)
        cth   = trial.suggest_float('contact_th',       0.01, 0.5)

        # Override pose-optimizer config in-place for this trial
        cfg.pose_optimizer.w_repulsive      = w_rep
        cfg.pose_optimizer.w_regularization = w_reg
        cfg.pose_optimizer.contact_th       = cth

        metrics = evaluate(trainer_model, dm, cfg, device, n_batches=val_batches)

        simu_disp = float(np.mean(metrics.get('Simulation Displacement', [0])))
        intersection_volume = float(np.mean(metrics.get('Intersection Volume', [0])))
        penetration_depth = float(np.mean(metrics.get('Penetration Depth', [0])))

        # Maximise contact quality, penalise penetration
        score = - simu_disp - penetration_depth - intersection_volume * 0.1
        print(f"  Trial {trial.number}: w_rep={w_rep:.4f}, w_reg={w_reg:.4f}, "
              f"contact_th={cth:.3f} → score={score:.4f} "
              f"(simu_disp={simu_disp:.4f}, penetr={penetration_depth:.4f})")
        return score

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name='pose_optimizer_search',
    )

    # Warm-start with current config values
    study.enqueue_trial({
        'w_repulsive':      float(cfg.pose_optimizer.w_repulsive),
        'w_regularization': float(cfg.pose_optimizer.w_regularization),
        'contact_th':       float(cfg.pose_optimizer.contact_th),
    })

    study.optimize(objective, n_trials=n_trials)

    print("\n" + "=" * 60)
    print("Best hyperparameters found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Best score: {study.best_value:.4f}")

    # Print as YAML snippet ready to paste into hybrid.yaml
    print("\nPaste into config/pose_optimizer/hybrid.yaml:")
    print(f"  w_repulsive:      {study.best_params['w_repulsive']:.6f}")
    print(f"  w_regularization: {study.best_params['w_regularization']:.6f}")
    print(f"  contact_th:       {study.best_params['contact_th']:.4f}")


if __name__ == '__main__':
    main()
