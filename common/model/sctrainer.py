import datetime
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import lightning as L
import trimesh
from collections import defaultdict
from copy import deepcopy, copy
import wandb
import open3d as o3d
from multiprocessing.pool import Pool
import time
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from common.manopth.manopth.manolayer import ManoLayer 
from common.model.handobject import HandObject
from common.model.hand_cse.hand_cse import HandCSE
from common.model.pose_optimizer import optimize_pose_contactopt
from common.evaluation.eval_fns import calculate_metrics, calc_diversity, value_metrics

thin_objects = ['wineglass', 'mug', 'fryingpan']

class SCTrainer(L.LightningModule):
    def __init__(self, model, cfg):
        super(SCTrainer, self).__init__()
        self.cfg = cfg
        self.corr_embedding_type = cfg.generator.embedding_type
        self.model_cfg = cfg.generator
        # self.label_cfg = cfg.label
        # self.n_neurons = cfg.generator.n_neurons
        # self.latentD = cfg.generator.latentD
        # self.hc = cfg.generator.pointnet_hc
        # self.object_feature = cfg.generator.obj_feature
        self.model = model

        ## Other utils
        self.closed_mano_faces = np.load(os.path.join('data', 'misc', 'closed_mano_r_faces.npy'))
        self.mano_layer = ManoLayer(mano_root=cfg.data.mano_root, use_pca=True, ncomps=cfg.pose_optimizer.ncomps,
                                    side='right', flat_hand_mean=False)
        self.hand_faces = self.mano_layer.th_faces
        self.part_ids = self.mano_layer.part_ids
        # self.testml = [testML(mano_root='data/mano_v1_2/models', use_pca=True, ncomps=26, side='right', flat_hand_mean=False)]
        self.lr = cfg.train.lr
        self.scheduler_step = cfg.train.scheduler_step
        self.lr_gamma = cfg.train.lr_gamma
        self.weight_rec = 1.0
        # self.contact_th = cfg.data.contact_th
        self.w_pene = cfg.pose_optimizer.w_pene

        # Convert cfg to dict to avoid OmegaConf struct mode issues
        self.save_hyperparameters({'cfg': OmegaConf.to_container(cfg, resolve=True)})
        self.validation_step_outputs = []
        if self.corr_embedding_type == 'cse':
            if hasattr(self.model, 'handcse'):
                self.handcse = self.model.handcse
                self.cse_dim = self.model.embed_dim
            else:
                handF = self.mano_layer.th_faces
                # Initialize model and load state
                cse_ckpt = torch.load(cfg.data.hand_cse_path)
                self.cse_dim = cse_ckpt['emb_dim']
                self.handcse = HandCSE(n_verts=778, emb_dim=cfg.data.hand_cse_dim, cano_faces=handF.cpu().numpy())
                self.handcse.load_state_dict(cse_ckpt['state_dict'])
                self.handcse.eval()
        else:
            self.embed_class = nn.Embedding(cfg.generator.part_dim, cfg.generator.pointnet_hc)

        self.debug = cfg.debug
        ## For simulation
        # self.object_templates = kwargs['object_templates']
        # self.object_hulls = kwargs['object_hulls']

    def training_step(self, batch, batch_idx):
        return self.train_val_process(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.train_val_process(batch, batch_idx, 'val')

    def train_val_process(self, batch, batch_idx, proc_name='train'):
        num_total_iter = self.trainer.num_training_batches if proc_name == 'train' else self.trainer.num_val_batches
        if type(num_total_iter) is list:
            num_total_iter = num_total_iter[0]

        self.weight_kl = self.kl_coeff(step=self.global_step,
                                       total_step=num_total_iter,
                                       constant_step=0,
                                       min_kl_coeff=1e-7,
                                       max_kl_coeff=self.cfg.generator.kl_coef)

        ho_gt = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer, normalize=False)
        # ho_gt = HandObject(self.device, self.hand_faces, self.part_ids, self.pressure_quantise_splits, contact_th=self.contact_th)
        ho_gt.load_from_batch(batch)

        verts_obj, obj_normals, contacts, nn_idx, parts = ho_gt.obj_verts, ho_gt.obj_normals,\
                ho_gt.contact_map, ho_gt.obj2hand_nn_idx, ho_gt.part_map
        # verts_obj, obj_normals, contacts, parts = ho_gt.get_phy_reps(
        #     self.model_cfg.obj_feature_keys, random_rotate=True if proc_name=='train' else False)
        gt = {'verts_object': verts_obj, 'normals_object':ho_gt.obj_normals, 'contacts_object': contacts}
        if self.corr_embedding_type == 'part':
            corr_feat = parts
            gt['partition_object'] = parts
        else:
            corr_feat = nn_idx
            gt['hand_cse'] = self.model.handcse.vert2emb(nn_idx.flatten()).reshape(nn_idx.shape[0], -1, self.cse_dim)
        results = self.model(verts_obj, obj_normals, contacts, corr_feat)
        # disps = torch.norm(batch['simuDisp'][:, :3], dim=-1)
        # disp_weight = 0.05 / (disps + 1e-6)
        # disp_weight[disp_weight > 1] = 1
        total_loss, loss_dict = self.loss_net(gt, results)
        log_loss_dict = {}
        for k, v in loss_dict.items():
            log_loss_dict[proc_name+'/'+k] = v
        self.log_dict(log_loss_dict, prog_bar=True, sync_dist=True)
        if proc_name == 'val':
            self.validation_step_outputs.append(loss_dict)
        if batch_idx % self.cfg[proc_name].vis_every_n_batch == 0:
            if proc_name == 'train':
                dataset = self.trainer.datamodule.train_set
            elif proc_name == 'val':
                dataset = self.trainer.datamodule.val_set
            obj_templates = []
            for obj_name in batch['objName']:
                obj_templates.append(
                    trimesh.Trimesh(dataset.obj_info[obj_name]['verts'], dataset.obj_info[obj_name]['faces']))
            # ho_gt.load_from_batch(batch, obj_templates=obj_templates)
            gt_img = ho_gt.vis_img(0, 250, 250, obj_templates=obj_templates, draw_maps=True)
            ho_pred = copy(ho_gt)
            ho_pred.contact_map = results['contacts_object'].squeeze(-1)
            if self.corr_embedding_type == 'cse':
                Wverts = self.handcse.emb2Wvert(results['partition_object']) # (B, N, 778)
                vertex_idx = torch.argmax(Wverts, dim=-1)  # (B, N)
                partition_object = self.mano_layer.part_ids[vertex_idx.cpu().numpy()]
                ho_pred.part_map = F.one_hot(torch.tensor(partition_object), num_classes=16).to(self.device).float()
                partition_object = results['partition_object']
            else:
                ho_pred.part_map = results['partition_object']
            pred_img = ho_pred.vis_img(0, 250, 250, obj_templates=obj_templates, draw_maps=True)
            img = wandb.Image(np.concatenate((gt_img, pred_img), axis=0), caption="Top: GT; Bottom: Pred")
            self.logger.experiment.log({f"{proc_name} Sample": img})
        return total_loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def on_validation_epoch_end(self):
        all_metrics = defaultdict(list)
        for out in self.validation_step_outputs:
            for k, v in out.items():
                all_metrics[k].append(v)

        mean_metrics = {}
        for k, v in all_metrics.items():
            mean_metrics[k] = sum(v) / len(v)

        self.logger.experiment.log(mean_metrics)
        return mean_metrics

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
        batch_size = self.cfg.test.batch_size
        obj_templates, obj_hull_templates = [], []
        dataset = self.trainer.datamodule.test_set
        for obj_name in batch['objName']:
            obj_templates.append(
                trimesh.Trimesh(dataset.obj_info[obj_name]['verts'], dataset.obj_info[obj_name]['faces']))
            obj_hull_templates.append(dataset.obj_hulls[obj_name])

        self.mano_layer.to(self.device)
        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer, normalize=True)

        obj_names = batch['objName']
        batch_start = time.time()
        if self.cfg.generator.model_type == 'gt':
            handobject.load_from_batch(batch, obj_templates=obj_templates)
            verts_obj, obj_normals, contacts, parts, nn_idx = handobject.obj_verts, handobject.obj_normals, handobject.contact_map, handobject.part_map, handobject.obj2hand_nn_idx
            contacts_object = contacts
            pred_parts = parts
            if self.corr_embedding_type == 'part':
                partition_object = pred_parts.argmax(dim=-1)
            elif self.corr_embedding_type == 'cse':
                cse = self.handcse.vert2emb(nn_idx.flatten()).reshape(nn_idx.shape[0], -1, self.cse_dim)
                partition_object = cse
        else:
            handobject.load_from_batch_obj_only(batch, obj_templates=obj_templates, obj_hulls=obj_hull_templates)
            # use_obj_normals = torch.cat([obj_normals[k] for k in self.model_cfg.obj_feature_keys], dim=-1)
            verts_obj, obj_normals = handobject.obj_verts, handobject.obj_normals
            batch_start = time.time()
            sample_result = self.model.sample(verts_obj, obj_normals)
            pred_contacts, pred_emb = sample_result
            contacts_object = pred_contacts.squeeze(1)
            if self.corr_embedding_type == 'part':
                pred_parts = pred_emb
                pred_parts = pred_parts.permute(0, 2, 1)
                partition_object = pred_parts.argmax(dim=-1)
            elif self.corr_embedding_type == 'cse':
                Wverts = self.handcse.emb2Wvert(pred_emb.permute(0, 2, 1)) # (B, N, 778)
                vertex_idx = torch.argmax(Wverts, dim=-1)  # (B, N)
                partition_object = self.mano_layer.part_ids[vertex_idx.cpu().numpy()]
                pred_parts = F.one_hot(torch.tensor(partition_object), num_classes=16).to(self.device).float()
                partition_object = pred_emb.permute(0, 2, 1)

        ## Ablation: avg predictor

        with torch.enable_grad():
            global_pose, mano_pose, mano_shape, mano_trans, init_pose = optimize_pose_contactopt(
                                self.mano_layer, verts_obj, obj_normals,
                                contacts_object, partition_object, n_iter=1000, save_history=False,
                                partition_type=self.corr_embedding_type, w_pen_cost=self.w_pene,
                                hand_cse=self.handcse if self.corr_embedding_type=='cse' else None,
                                is_thin=torch.LongTensor([obj_name in thin_objects for obj_name in obj_names]).to(self.device))

            self.runtime += time.time() - batch_start
            print(self.runtime / (batch_idx + 1))

        handV, handJ, _ = self.mano_layer(torch.cat([global_pose, mano_pose], dim=1), th_betas=mano_shape, th_trans=mano_trans)

        handV, handJ = handV.detach().cpu().numpy(), handJ.detach().cpu().numpy()

        param_list = [{'dataset_name': 'grab', 'frame_name': f"{obj_names[i]}_{i}", 'hand_model': trimesh.Trimesh(handV[i], self.closed_mano_faces),
                       'obj_name': obj_names[i], 'hand_joints': handJ[i], 'obj_model': obj_templates[i], 'obj_hulls': obj_hull_templates[i],
                       'idx': i} for i in range(handV.shape[0])]
            
        result = calculate_metrics(param_list, metrics=self.cfg.test.criteria, pool=self.pool, reduction='none')
        ## MPJPE:
        if self.cfg.generator.model_type == 'gt':
            mpjpe = np.linalg.norm(handobject.hand_joints.cpu().numpy() - handJ, axis=-1).mean(axis=1) * 1000  # B,
            mpvpe = np.linalg.norm(handobject.hand_verts.cpu().numpy() - handV, axis=-1).mean(axis=1) * 1000  # B,
            result.update({'MPJPE': mpjpe, 'MPVPE': mpvpe})

        # print(result)
        self.all_results.append(result)
        self.sample_joints.append(handJ)

        # Log raw per-sample metrics to wandb
        if not self.debug:
            # Option 1: Log each sample as individual rows (creates distributions in wandb)
            for i in range(len(next(iter(result.values())))):
                sample_metrics = {f"{metric_name}": float(metric_values[i])
                                 for metric_name, metric_values in result.items()}
                wandb.log(sample_metrics)
            # Visualize some samples

            vis_idx = 0
            pred_ho = copy(handobject)
            pred_ho.hand_verts = torch.tensor(handV, dtype=torch.float32)
            pred_ho.hand_joints = torch.tensor(handJ, dtype=torch.float32)
            pred_ho.contact_map = contacts_object
            pred_ho.part_map = pred_parts
            img = pred_ho.vis_img(vis_idx, 250, 250, obj_templates=obj_templates, draw_maps=True)
            if self.cfg.generator.model_type == 'gt':
                gt_img = handobject.vis_img(vis_idx, 250, 250, obj_templates=obj_templates, draw_maps=False)
                img = np.concatenate((gt_img, img), axis=0)

            wandb.log({f'test/sampled_grasp': wandb.Image(img)})
        
        else:
            print(result)
            for vis_idx in range(batch_size):
                # gt_geoms = handobject.get_vis_geoms(idx=vis_idx, obj_templates=obj_templates)
                handV0, handJ0, _ = self.mano_layer(torch.cat([init_pose[0], torch.zeros_like(mano_pose)], dim=1), th_trans=init_pose[1])
                pred_ho = copy(handobject)
                pred_ho.hand_verts = handV0
                pred_ho.hand_joints = handJ0
                pred_ho.contact_map = contacts_object
                pred_ho.part_map = pred_parts
                init_img = pred_ho.vis_img(vis_idx, 250, 250, obj_templates=obj_templates)

                pred_ho.hand_verts = torch.tensor(handV, dtype=torch.float32)
                pred_ho.hand_joints = torch.tensor(handJ, dtype=torch.float32)
                pred_img = pred_ho.vis_img(vis_idx, 250, 250, obj_templates=obj_templates, draw_maps=True)
                if self.cfg.generator.model_type == 'gt':
                    gt_img = handobject.vis_img(vis_idx, 250, 250, obj_templates=obj_templates)
                    vis_img = np.concatenate((gt_img, init_img, pred_img), axis=0)
                else:
                    vis_img = np.concatenate((init_img, pred_img), axis=0)
                plt.imsave(f'logs/tb_logs/debug_samples/test_sample_{batch_idx*batch_size+vis_idx}.png', vis_img)
                print(f"Saved debug image for sample {batch_idx*batch_size+vis_idx}")

            # o3d.visualization.draw(gt_img + [g['geometry'].translate((0, 0.25, 0)) if 'geometry' in g else g for g in pred_img])

    def on_test_epoch_end(self):
        final_metrics = {}

        # Compute statistics for all metrics from test results
        for m in self.cfg.test.criteria:
            if "Entropy" not in m and "Cluster Size" not in m:
                all_metrics = np.concatenate([res[m] for res in self.all_results], axis=0)
                # Log comprehensive statistics for each metric
                final_metrics[f"{m}/mean"] = np.mean(all_metrics).item()
                if m not in value_metrics:
                    final_metrics[f"{m}/std"] = np.std(all_metrics).item()
                    final_metrics[f"{m}/min"] = np.min(all_metrics).item()
                    final_metrics[f"{m}/max"] = np.max(all_metrics).item()
                    final_metrics[f"{m}/median"] = np.median(all_metrics).item()
                    # final_metrics[f"{m}/p25"] = np.percentile(all_metrics, 25).item()
                    # final_metrics[f"{m}/p75"] = np.percentile(all_metrics, 75).item()

        ## Calculate diversity
        sample_joints = np.concatenate(self.sample_joints, axis=0)
        # run_time = np.concatenate(run_time, axis=0)
        entropy, cluster_size, entropy_2, cluster_size_2 = calc_diversity(sample_joints)
        final_metrics.update({
            "Entropy": entropy.item(),
            "Cluster Size": cluster_size.item(),
            "Canonical Entropy": entropy_2.item(),
            "Canonical Cluster Size": cluster_size_2.item()
        })

        # Log final metrics to wandb
        if not self.debug:
            wandb.log(final_metrics)

        # with open('logs/grasp_results/grab_Obj_hulls.pkl', 'wb') as f:
        #     pickle.dump(self.obj_hulls, f)
        # summary_metrics = defaultdict(list)
        # for m in self.test_metrics:
        #     for k, v in m.items():
        #         summary_metrics[k].append(v)
        # for k, v in summary_metrics.items():
        #     summary_metrics[k] = sum(v) / len(v)
        # self.logger.experiment.log(summary_metrics)
        # for k, v in self.result_dict.items():
        #     self.result_dict[k] = np.concatenate(v, axis=0)
        # res_df = pd.DataFrame(self.result_dict)
        # res_df.to_csv(os.path.join('data', 'disp_raw_data', self.cfg.generator.name + '_' + self.cfg.data.dataset + '.csv'))
        self.pool.close()
        self.pool.join()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=self.lr_gamma)
        return [optimizer], [scheduler]

    def loss_net(self, dorig, drec):
        device = dorig['verts_object'].device
        dtype = dorig['verts_object'].dtype
        batch_size = dorig['verts_object'].shape[0]

        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([batch_size, self.model_cfg.latentD]), requires_grad=False).to(device).type(dtype),
            scale=torch.tensor(np.ones([batch_size, self.model_cfg.latentD]), requires_grad=False).to(device).type(dtype))

        q_z_contact = torch.distributions.normal.Normal(drec['mean_contact'], drec['std_contact'])
        loss_kl_contact = torch.sum(torch.sum(torch.distributions.kl.kl_divergence(q_z_contact, p_z), dim=[1])) / batch_size

        q_z_part = torch.distributions.normal.Normal(drec['mean_part'], drec['std_part'])
        loss_kl_part = torch.sum(torch.sum(torch.distributions.kl.kl_divergence(q_z_part, p_z), dim=[1])) / batch_size

        # q_z_pressure = torch.distributions.normal.Normal(drec['mean_pressure'], drec['std_pressure'])
        # loss_kl_pressure = torch.sum(torch.sum(torch.distributions.kl.kl_divergence(q_z_pressure, p_z), dim=[1])) / batch_size

        if self.model_cfg.robustkl:
            loss_kl_contact = torch.sqrt(1 + loss_kl_contact ** 2) - 1
            loss_kl_part = torch.sqrt(1 + loss_kl_part ** 2) - 1
            # loss_kl_pressure = torch.sqrt(1 + loss_kl_pressure ** 2) - 1

        loss_dict = {'loss_kl_contact': loss_kl_contact,
                     'loss_kl_part': loss_kl_part,
                    #  'loss_kl_pressure': loss_kl_pressure
                     }

        loss_kl_contact = loss_kl_contact * self.weight_kl
        loss_kl_part = loss_kl_part * self.weight_kl
        # loss_kl_pressure = loss_kl_pressure * self.weight_kl

        target_contact = dorig['contacts_object'].to(device).squeeze(dim=-1)
        weight = 1. + 5. * target_contact
        contact_obj_sub = target_contact - drec['contacts_object'].squeeze(dim=-1)
        contact_obj_weighted = contact_obj_sub * weight

        loss_contact_rec = F.l1_loss(contact_obj_weighted,
                                     torch.zeros_like(contact_obj_weighted, device=target_contact.device,
                                                      dtype=target_contact.dtype), reduction='none')
        loss_contact_rec = self.weight_rec * torch.sum(torch.mean(loss_contact_rec, dim=-1)) / batch_size

        if 'partition_object' in dorig:
            target_part = dorig['partition_object'].argmax(dim=-1).to(device)
            loss_part_rec = F.nll_loss(input=F.log_softmax(drec['partition_object'], dim=-1).float().permute(0, 2, 1),
                                    target=target_part.long(), reduction='none')
        elif 'hand_cse' in dorig:
            ## Use L2 loss
            target_part = dorig['hand_cse'].to(device)
            loss_part_rec = F.mse_loss(target_part, drec['partition_object'], reduction='none').sum(dim=-1)

        loss_part_rec = self.weight_rec * 0.5 * torch.sum(torch.mean(weight * loss_part_rec, dim=-1)) / batch_size

        # weight_pres = 1. + 5. * target_contact
        # target_pressure = torch.sum(dorig['pressure_object'].to(device)* self.mid_pts.to(self.device).view(1, 1, -1), dim=-1)
        ## use soft-argmax with beta=50 for such a regression problem.
        # remapped_pressure = torch.sum(F.softmax(drec['pressure_object'] * 50, dim=-1) * self.mid_pts.to(self.device).view(1, 1, -1), dim=-1)

        # loss_pressure_rec = F.nll_loss(input=F.log_softmax(drec['pressure_object'], dim=-1).float().permute(0, 2, 1),
        #                                target=target_pressure.long(), reduction='none')
        # pressure_obj_sub = target_pressure - drec['pressure_object'].squeeze(dim=-1)
        # loss_pressure_rec = F.l1_loss(remapped_pressure, target_pressure, reduction='none')
        # loss_pressure_rec = self.weight_rec * torch.sum(torch.sum(weight_pres * loss_pressure_rec, dim=-1) / weight_pres.sum(dim=-1) * disp_weight) / torch.sum(disp_weight)

        loss_dict.update({'loss_contact_rec': loss_contact_rec,
                          'loss_part_rec': loss_part_rec,
                        #   'loss_pressure_rec': loss_pressure_rec
                          })

        loss_total = loss_kl_contact + loss_kl_part + loss_contact_rec + loss_part_rec
        loss_dict['loss_total'] = loss_total

        return loss_total, loss_dict

    @staticmethod
    def kl_coeff(step, total_step, constant_step, min_kl_coeff, max_kl_coeff):
        return max(min(min_kl_coeff + (max_kl_coeff - min_kl_coeff) * (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)
