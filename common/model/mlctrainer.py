import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import open3d as o3d
import trimesh
import numpy as np
from multiprocessing import Pool
import wandb

from common.manopth.manopth.manolayer import ManoLayer
from common.model.handobject import recover_hand_verts_from_contact
from common.model.pose_optimizer import optimize_pose_wrt_local_grids
from common.model.handobject import HandObject, recover_hand_verts_from_contact
from common.model.hand_cse.hand_cse import HandCSE
from common.utils.vis import o3dmesh_from_trimesh, visualize_local_grid_with_hand, geom_to_img, extract_masked_mesh_components
from common.model.losses import masked_rec_loss, kl_div_normal_muvar
from common.msdf.utils.msdf import get_grid
from common.evaluation.eval_fns import calculate_metrics, calc_diversity

value_metrics = ["Contact Ratio", "Success Rate", "Pierce-Free Rate", "Cluster Size", "Entropy", "Canonical Entropy", "Canonical Cluster Size"]

class MLCTrainer(L.LightningModule):
    """
    The Lightning trainer interface to train Local-grid based contact autoencoder.
    """
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.debug = cfg.get('debug', False)
        self.loss_weights = cfg.train.loss_weights
        self.msdf_k = cfg.msdf.kernel_size
        self.lr = cfg.train.lr
        self.mano_layer = ManoLayer(mano_root=cfg.data.mano_root, side='right',
                                    use_pca=False, ncomps=45, flat_hand_mean=True)
        self.closed_mano_faces = np.load(osp.join())
        cse_ckpt = torch.load(cfg.data.hand_cse_path)

        handF = self.mano_layer.th_faces
        # Initialize model and load state
        self.cse_dim = cse_ckpt['emb_dim']
        self.handcse = HandCSE(n_verts=778, emb_dim=self.cse_dim, cano_faces=handF.cpu().numpy()).to(self.device)
        self.handcse.load_state_dict(cse_ckpt['state_dict'])
        self.handcse.eval()
        self.grid_coords = get_grid(self.cfg.msdf.kernel_size) * self.cfg.msdf.scale  # (K^3, 3)

    def training_step(self, batch, batch_idx):
        total_loss = self.train_val_step(batch, batch_idx, stage='train')
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        total_loss = self.train_val_step(batch, batch_idx, stage='val')
        return total_loss
    
    def train_val_step(self, batch, batch_idx, stage):
        self.grid_coords = self.grid_coords.view(-1, 3).to(self.device)
        obj_names = batch['objName']
        simp_obj_mesh = getattr(self.trainer.datamodule, f'{stage}_set').simp_obj_mesh
        obj_templates = [trimesh.Trimesh(simp_obj_mesh[name]['verts'], simp_obj_mesh[name]['faces'])
                         for name in obj_names]
        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer)
        handobject.load_from_batch(batch, obj_templates=obj_templates)
        lg_contact = handobject.ml_contact
        obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, self.msdf_k, self.msdf_k, self.msdf_k)

        obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:] # B x 3
        recon_lg_contact, mu, logvar = self.model(
            lg_contact.permute(0, 1, 5, 2, 3, 4), obj_msdf, obj_msdf_center)
        recon_lg_contact = recon_lg_contact.permute(0, 1, 3, 4, 5, 2)  # B x N x K x K x K x (1 + cse_dim)

        batch_size = handobject.batch_size
        contact_hat, cse_hat = recon_lg_contact[..., 0], recon_lg_contact[..., 1:]
        grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords[None, None, :, :]  # B x N x K^3 x 3
        pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
            self.handcse,
            contact_hat.reshape(batch_size, -1),
            cse_hat.reshape(batch_size, -1, self.cse_dim),
            grid_coords=grid_coords.view(batch_size, -1, 3),
            mask_th = 0.02
        )

        loss_dict = self.loss_net(recon_lg_contact, lg_contact, mu, logvar,
                                  pred_hand_verts, handobject.hand_verts,
                                  handobject.hand_vert_mask)
                                #   pred_verts_mask)
        print(loss_dict)
                                  
        # recon_loss = F.mse_loss(recon_grid_contact, gt_grid_contact.permute(0, 4, 1, 2, 3))
        # loss_dict = {f'{stage}/embedding_loss': loss, f'{stage}/recon_loss': recon_loss, f'{stage}/perplexity': perplexity}
        # total_loss = sum(loss_dict.values())
        # total_loss = loss + self.recon_weight * recon_loss
        if batch_idx % self.cfg[stage].vis_every_n_batches == 0:
            # Log image - works for both WandbLogger and TensorBoardLogger
            vis_idx = 0
            pred_img, pred_geoms = self.visualize_recon_hand_w_object(hand_verts=pred_hand_verts[vis_idx].detach().cpu().numpy(),
                                        hand_verts_mask=pred_verts_mask[vis_idx].detach().cpu().numpy(),
                                        hand_faces=self.mano_layer.th_faces.detach().cpu().numpy(),
                                        obj_mesh=handobject.obj_models[vis_idx],
                                        msdf_center=obj_msdf_center[vis_idx].detach().cpu().numpy(),
                                        grid_scale=self.cfg.msdf.scale,
                                        h=400, w=400)
            gt_img, gt_geoms = self.visualize_recon_hand_w_object(hand_verts=handobject.hand_verts[vis_idx].detach().cpu().numpy(),
                                        # hand_verts_mask=handobject.hand_vert_mask[vis_idx].detach().cpu().numpy(),
                                        hand_verts_mask=pred_verts_mask[vis_idx].detach().cpu().numpy(),
                                        hand_faces=self.mano_layer.th_faces.detach().cpu().numpy(),
                                        obj_mesh=handobject.obj_models[vis_idx],
                                        mesh_color=[0.2, 0.4, 0.8],
                                        msdf_center=obj_msdf_center[vis_idx].detach().cpu().numpy(),
                                        grid_scale=self.cfg.msdf.scale,
                                        h=400, w=400)
            vis_geoms = pred_geoms + [g.translate((0, 0.25, 0)) for g in gt_geoms]
            o3d.visualization.draw_geometries(vis_geoms, window_name='GT Hand-Object')
            img = np.concatenate([gt_img, pred_img], axis=0)
            if hasattr(self.logger, 'experiment'):
                if hasattr(self.logger.experiment, 'add_image'):
                    # TensorBoardLogger
                    global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.{stage}_dataloader()')) + batch_idx
                    self.logger.experiment.add_image(f'{stage}/compare_hand', img, global_step, dataformats='HWC')
                elif hasattr(self.logger.experiment, 'log'):
                    # WandbLogger
                    import wandb
                    self.logger.experiment.log({f'{stage}/compare_hand': wandb.Image(img)},
                                              step=self.global_step)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict['total_loss']
    
    def on_test_epoch_start(self):
        self.pool = Pool(min(self.cfg.test.batch_size, 16))
        self.all_results = []
        self.sample_joints = []

        ## Testing metrics
        self.runtime = 0
        if not self.debug:
            for metric in self.cfg.test.criteria:
                wandb.define_metric(metric, summary='mean')
    
    def test_step(self, batch, batch_idx):
        self.grid_coords = self.grid_coords.to(self.device)
        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer)
        obj_hulls = getattr(self.trainer.datamodule, 'test_set').obj_hulls
        obj_names = batch['objName']
        obj_hulls = [obj_hulls[name] for name in obj_names]
        obj_mesh_dict = getattr(self.trainer.datamodule, 'test_set').simp_obj_mesh
        obj_meshes = []
        for name in obj_names:
            obj_meshes.append(trimesh.Trimesh(obj_mesh_dict[name]['verts'], obj_mesh_dict[name]['faces']))

        if self.cfg.generator.model_type == 'gt':
            ## Test using gt contact grids.
            handobject.load_from_batch(batch, obj_templates=obj_meshes, obj_hulls=obj_hulls)
            recon_lg_contact = handobject.ml_contact
        else:
            handobject.load_from_batch_object_only(batch, obj_templates=obj_meshes, obj_hulls=obj_hulls)
            obj_msdf = handobject.obj_msdf[:, :self.msdf_k**3].view(-1, self.msdf_k, self.msdf_k, self.msdf_k)
            obj_msdf_center = handobject.obj_msdf[:, self.msdf_k**3:]
            recon_lg_contact = self.model.sample(obj_msdf, obj_msdf_center)
        pred_grid_contact = recon_lg_contact[..., 0]
        grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords[None, None, :, :]  # B x N x K^3 x 3
        pred_grid_cse = pred_grid_contact[..., 1:].permute(0, 2, 3, 4, 5, 1).view(batch_size, -1, self.cse_dim)
        pred_targetWverts = self.hand_cse.emb2Wverts(pred_grid_cse)
        batch_size = handobject.batch_size

        global_pose, mano_pose, mano_shape, mano_trans = optimize_pose_wrt_local_grids(
                    self.mano_layer, target_pts=grid_coords.view(batch_size, -1, 3),
                    target_W_verts=pred_targetWverts, mask=None,
                    n_iter=self.cfg.pose_optimizer.opt_iter, lr=self.cfg.pose_optimizer.opt_lr)
        
        handV, handJ, _ = self.mano_layer(torch.cat([global_pose, mano_pose], dim=1), th_betas=mano_shape, th_trans=mano_trans)

        handV, handJ = handV.detach().cpu().numpy(), handJ.detach().cpu().numpy()

        param_list = [{'dataset_name': 'grab', 'frame_name': f"{obj_names[i]}_{i}", 'hand_model': trimesh.Trimesh(handV[i], self.closed_mano_faces),
                       'obj_name': obj_names[i], 'hand_joints': handJ[i], 'obj_model': handobject.obj_models[i], 'obj_hulls': handobject.obj_hulls[i],
                       'idx': i} for i in range(handV.shape[0])]
            
        result = calculate_metrics(param_list, metrics=self.cfg.test.criteria, pool=self.pool, reduction='none')
        self.all_results.append(result)
        self.sample_joints.append(handJ)

    def on_test_epoch_end(self):
        final_metrics = {}
        for m in self.cfg.test.criteria:
            if "Entropy" not in m and "Cluster Size" not in m:
                all_metrics = np.concatenate([res[m] for res in self.all_results], axis=0)
                if m in value_metrics:
                    final_metrics[m] = {'value': np.mean(all_metrics).item()}
                else:
                    final_metrics[m] = {
                        'mean': np.mean(all_metrics).item(),
                        'std': np.std(all_metrics).item()
                    }

        ## Calculate diversity
        sample_joints = np.concatenate(self.sample_joints, axis=0)
        # run_time = np.concatenate(run_time, axis=0)
        entropy, cluster_size, entropy_2, cluster_size_2 = calc_diversity(sample_joints)
        final_metrics.update({
            "Entropy": {'value': entropy.item()},
            "Cluster Size": {'value': cluster_size.item()},
            "Canonical Entropy": {'value': entropy_2.item()},
            "Canonical Cluster Size": {'value': cluster_size_2.item()}
        })

        return final_metrics
    
        ## GT visualization
        # vis_idx = 0
        # gt_geoms = visualize_local_grid_with_hand(
        #         batch['localGrid'][vis_idx].cpu().numpy(), hand_verts=batch['nHandVerts'][vis_idx].cpu().numpy(),
        #         hand_faces=self.mano_layer.th_faces.cpu().numpy(), hand_cse=self.handcse.embedding_tensor.detach().cpu().numpy(),
        #         kernel_size=self.cfg.msdf.kernel_size, grid_scale=self.cfg.msdf.scale
        #     )
        # o3d.visualization.draw_geometries(gt_geoms, window_name='GT Local Grid Visualization')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def loss_net(self, pred_lgc, gt_lgc, mu, logvar, pred_hand_verts, gt_hand_verts, contact_verts_mask):
        """
        Compute the loss for training the GRIDAE
        1. Reconstruction loss between x and x_hat
        2. KL-divergence loss on z_e 
        """
        pred_contact, pred_cse = pred_lgc[..., 0], pred_lgc[..., 1:]
        gt_contact, gt_cse = gt_lgc[..., 0], gt_lgc[..., 1:]
        contact_loss = F.mse_loss(pred_contact, gt_contact)
        cse_loss = F.mse_loss(pred_cse, gt_cse).item()
        kl_loss = kl_div_normal_muvar(mu, logvar).item()
        hand_rec_loss = masked_rec_loss(pred_hand_verts, gt_hand_verts, contact_verts_mask).item()
        loss_dict = {
            'contact_loss': contact_loss,
            'cse_loss': cse_loss,
            'kl_loss': kl_loss,
            'hand_rec_loss': hand_rec_loss,
            'total_loss': self.loss_weights.w_contact * contact_loss + self.loss_weights.w_cse * cse_loss\
                + self.loss_weights.w_kl * kl_loss + self.loss_weights.w_rec * hand_rec_loss
        }

        return loss_dict
    
    @staticmethod
    def visualize_recon_hand_w_object(hand_verts, hand_verts_mask, hand_faces, obj_mesh, msdf_center, grid_scale, mesh_color=[0.8, 0.7, 0.6], h=500, w=500):
        masked_hand_geometries = extract_masked_mesh_components(
            hand_verts=hand_verts,
            hand_faces=hand_faces,
            vertex_mask=hand_verts_mask,
            create_geometries=True,
            mesh_color=mesh_color
        )
        obj_o3d_mesh = o3dmesh_from_trimesh(obj_mesh, color=[0.7, 0.7, 0.7])

        # Create bounding boxes for each MSDF center
        bbox_geometries = []
        for center in msdf_center:
            # Define 8 corners of the bounding box
            bbox_points = np.array([
                center + grid_scale * np.array([-1, -1, -1]),
                center + grid_scale * np.array([1, -1, -1]),
                center + grid_scale * np.array([1, 1, -1]),
                center + grid_scale * np.array([-1, 1, -1]),
                center + grid_scale * np.array([-1, -1, 1]),
                center + grid_scale * np.array([1, -1, 1]),
                center + grid_scale * np.array([1, 1, 1]),
                center + grid_scale * np.array([-1, 1, 1]),
            ])

            # Define edges of the bounding box
            bbox_lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
            ]

            # Create LineSet for bounding box
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(bbox_points)
            line_set.lines = o3d.utility.Vector2iVector(bbox_lines)
            line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in bbox_lines])  # Green
            bbox_geometries.append(line_set)

        vis_geoms = masked_hand_geometries + [obj_o3d_mesh] + bbox_geometries
        img = geom_to_img(vis_geoms, w=w, h=h, scale=0.7)
        return img, vis_geoms
    
    @staticmethod
    def vis_contact(obj_mesh, hand_mesh, obj_pts, obj_pt_mask):
        hand_mesh = o3dmesh_from_trimesh(hand_mesh, color=[0.8, 0.7, 0.6])
        obj_mesh = o3dmesh_from_trimesh(obj_mesh, color=[0.7, 0.7, 0.7])

        # Convert obj_pts to numpy if it's a tensor
        if isinstance(obj_pts, torch.Tensor):
            obj_pts_np = obj_pts.cpu().numpy()
        else:
            obj_pts_np = obj_pts

        # Convert obj_pt_mask to numpy if it's a tensor
        if isinstance(obj_pt_mask, torch.Tensor):
            obj_pt_mask_np = obj_pt_mask.cpu().numpy()
        else:
            obj_pt_mask_np = obj_pt_mask

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_pts_np)

        # Color points: red for masked (True), blue for rest (False)
        colors = np.zeros((len(obj_pts_np), 3))
        colors[obj_pt_mask_np] = [1.0, 0.0, 0.0]  # Red for selected points
        colors[~obj_pt_mask_np] = [0.0, 0.0, 1.0]  # Blue for rest
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize
        o3d.visualization.draw_geometries([hand_mesh, obj_mesh, pcd])


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        # Add a dummy parameter to satisfy optimizer requirements
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x, msdf, msdf_center):
        batch_size = x.shape[0]
        recon_x = x.clone()
        mean = torch.randn(batch_size, 16) * 0.1
        logvar = torch.randn(batch_size, 16) * 0.1

        return recon_x, mean, logvar