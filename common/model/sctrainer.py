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
from multiprocessing.pool import Pool
import time

from common.manopth.manopth.manolayer import ManoLayer 
from common.model.handobject import HandObject
from common.model.pose_optimizer import optimize_pose_contactopt
from common.evaluation.eval_fns import calculate_metrics, calc_diversity


class SCTrainer(L.LightningModule):
    def __init__(self, model, cfg):
        super(SCTrainer, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.generator
        # self.label_cfg = cfg.label
        self.n_neurons = cfg.generator.n_neurons
        self.latentD = cfg.generator.latentD
        self.hc = cfg.generator.pointnet_hc
        self.object_feature = cfg.generator.obj_feature
        self.model = model

        self.num_parts = 16
        self.embed_class = nn.Embedding(self.num_parts, self.hc)

        ## Other utils
        self.mano_layer = ManoLayer(mano_root=cfg.data.mano_root, use_pca=True, ncomps=26, side='right', flat_hand_mean=False)
        self.hand_faces = self.mano_layer.th_faces
        self.part_ids = self.mano_layer.part_ids
        # self.testml = [testML(mano_root='data/mano_v1_2/models', use_pca=True, ncomps=26, side='right', flat_hand_mean=False)]
        self.lr = cfg.train.lr
        self.scheduler_step = cfg.train.scheduler_step
        self.lr_gamma = cfg.train.lr_gamma
        self.max_kl_coef = cfg.generator.kl_coef
        self.weight_rec = 1.0
        # self.contact_th = cfg.data.contact_th

        self.save_hyperparameters(cfg)
        self.validation_step_outputs = []

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
                                       max_kl_coeff=self.max_kl_coef)

        ho_gt = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer, normalize=False)
        # ho_gt = HandObject(self.device, self.hand_faces, self.part_ids, self.pressure_quantise_splits, contact_th=self.contact_th)
        ho_gt.load_from_batch(batch)

        verts_obj, obj_normals, contacts, parts = ho_gt.obj_verts, ho_gt.obj_normals, ho_gt.contact_map, ho_gt.part_map
        # verts_obj, obj_features, contacts, parts = ho_gt.get_phy_reps(
        #     self.model_cfg.obj_feature_keys, random_rotate=True if proc_name=='train' else False)
        results = self.model(verts_obj, obj_normals, contacts, parts)
        gt = {'verts_object': verts_obj, 'normals_object':ho_gt.obj_normals, 'contacts_object': contacts, 'partition_object': parts}
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
            ho_pred.contact_map, ho_pred.part_map = results['contacts_object'].squeeze(-1), results['partition_object']
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
        ho_gt = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer)

        obj_names = batch['objName']
        ho_gt.load_from_batch_obj_only(batch, obj_templates=obj_templates, obj_hulls=obj_hull_templates)
        # use_obj_features = torch.cat([obj_features[k] for k in self.model_cfg.obj_feature_keys], dim=-1)
        verts_obj, obj_features, contacts, parts = ho_gt.obj_verts, ho_gt.obj_features, ho_gt.contact_map, ho_gt.part_map
        batch_start = time.time()
        sample_result = self.sample(verts_obj, obj_features)
        pred_contacts, pred_parts = sample_result

        contacts_object = pred_contacts.squeeze(-1)
        partition_object = pred_parts.argmax(dim=-1)
        ## Ablation: avg predictor
        # pressure_object[:] = 0.5295791

        with torch.enable_grad():
            global_pose, mano_pose, mano_shape, mano_trans = optimize_pose_contactopt(
                                self.mano_layer, verts_obj, ho_gt.obj_normals,
                                contacts_object, partition_object, n_iter=1000, ret_history=False)

            self.runtime += time.time() - batch_start
            print(self.runtime / (batch_idx + 1))
        res = {'pred_contacts': contacts_object.unsqueeze(-1), 'pred_parts': ho_gt.part_map, 'pred_pressure': ho_gt.pressure_map}
        hand_params = {'rot_aa': global_pose, 'pose': mano_pose, 'shape': mano_shape, 'trans': mano_trans}
        handV, handJ, handF = self.mano_layer.mesh_data_np(hand_params)
        # contact_mask = contacts_object > self.contact_th

        ho_pred = copy(ho_gt)
        res.update({'hand_verts': handV, 'hand_joints': handJ,
                    'obj_verts': np.stack([om.vertices for om in ho_gt.obj_models], axis=0),
                    'obj_faces': np.stack([om.faces for om in ho_gt.obj_models], axis=0)})
        self.sample_result.append(res)
        # ho_pred.hand_models = [trimesh.Trimesh(handV[i], handF) for i in range(handV.shape[0])]
        # if self.model_cfg.name == 'contactgen':
        #     handrot = torch.cat([global_pose, mano_pose], dim=-1)
        #     handV, handJ, hand_frames = self.mano_layer(handrot, th_betas=mano_shape, th_trans=mano_trans)
        #     handV = handV.detach().cpu().numpy()
        #     handJ = handJ.detach().cpu().numpy()
        # else:

        hand_models = [trimesh.Trimesh(handV[i], self.hand_faces) for i in range(handV.shape[0])]
        ho_pred.hand_models = hand_models
        ho_pred.hand_joints = handJ
        ho_pred.update_contact_from_hand_models()

        # ho_baseline = copy(ho_pred)
        # hand_params1 = {'rot_aa': global_pose1, 'pose': mano_pose1, 'shape': mano_shape1, 'trans': mano_trans1}
        # handV1, handJ1, handF1 = self.mano_layer.mesh_data_np(hand_params1)
        # ho_baseline.hand_models = [trimesh.Trimesh(handV1[i], self.hand_faces) for i in range(handV1.shape[0])]
        # ho_baseline.hand_joints = handJ1

        ## Calculate force errors: in a way similar to heatmap
        # pe = pressure_error(gt_pressure, pred_pressure, verts_obj)

        # for i in range(handV.shape[0]):
        param_list = [{'label_cfg': deepcopy(self.label_cfg), 'dataset_name': 'grab',
                       'frame_name': f"{obj_names[i]}_{i}", 'hand_model': hand_models[i],
                       'obj_name': obj_names[i], 'hand_joints': handJ[i],
                       'obj_model': ho_gt.obj_models[i], 'obj_hulls': ho_gt.obj_hulls[i],
                       'idx': batch_idx * batch_size + i, 'part_id': self.part_ids} for i in range(len(ho_pred.hand_models))]
        # self.obj_hulls += ho_gt.obj_hulls
        assert handJ.shape[1] == 21
        self.sample_joints.append(handJ)

        # result_metrics = self.pool.map(parallel_calculate_metrics, param_list)
        metric_names = ["PyBullet SimuDisp", "Pybullet Stable Rate", "Intersection Volume", "Contact Ratio"]
        results = calculate_metrics(param_list, pool=self.pool, metrics=metric_names)
        #
        metrics = {}
        for m in metric_names:
            metrics[m] = results[m]

        ## Calculate the GT pressure map using simulation.
        # qposes = np.stack(results['obj_disp'], axis=0)
        # label_disps = np.stack(results['label_obj_disp'], axis=0)
        # contacts = results['contacts']
        # ho_pred.calculate_pressure(contacts, torch.as_tensor(qposes).float(), torch.as_tensor(label_disps).float())
        ## Get part-level contact forces
        # gt_part_pres = torch.zeros_like(part_pres, device=self.device).float()
        # for b in range(batch_size):
        #     for c in contacts[b]:
        #         part_id = c['part_id']
        #         part_force = torch.as_tensor(c['frame'].reshape(3, 3).T @ c['force'].reshape(3, 1), device=self.device).squeeze()
        #         gt_part_pres[b, part_id] += part_force

        ## Force related predictions:
        # metrics['Force value error'] = torch.mean(torch.abs(torch.abs(torch.norm(gt_part_pres, dim=-1) - torch.norm(part_pres, dim=-1))))
        # metrics['Force angular error'] = torch.mean(torch.arccos(torch.sum(gt_part_pres * part_pres, dim=-1) / (torch.norm(gt_part_pres, dim=-1)*torch.norm(part_pres, dim=-1) + 1e-8)))
        # metrics['Clustered force value error'] = torch.mean(torch.abs(torch.abs(torch.norm(gt_part_pres, dim=-1) - torch.norm(new_part_pres, dim=-1))))
        # metrics['Clustered force angular error'] = torch.mean(torch.arccos(torch.sum(gt_part_pres * new_part_pres, dim=-1) / (torch.norm(gt_part_pres, dim=-1)*torch.norm(new_part_pres, dim=-1) + 1e-8)))

        # simu_pressure = ho_pred.quantised_pressure
        # pred_q_pressure = torch.argmax(pred_pressure, dim=-1)
        # if self.model_cfg.name != 'external' and contact_mask.any() and not self.test_gt:
            # pressure_acc = torch.sum(simu_pressure[contact_mask] == pred_q_pressure[contact_mask]) / torch.sum(contact_mask)
            # metrics.update({'Pressure Err': pressure_value_error(contact_mask, ho_pred.pressure_map.sum(dim=-1), pressure_object)})
            # gt_pressure = ho_pred.pressure_map.sum(dim=-1)[contact_mask]
            # metrics.update({'Pressure Err Avg Predictor': torch.abs(gt_pressure - 0.5295791).mean()})
            # metrics.update({'Quantised Pressure Err': pressure_value_error(contact_mask, self.mid_pts.to(self.device)[simu_pressure], pressure_object)})
        #
        #     ho_pred.onehot_pressure = pred_pressure
        #
        if not self.debug:
            self.logger.log_metrics(metrics)
        else:
            print(metrics)

        # if batch_idx % self.cfg.test.vis_every_n_batch == 0:
        print(metrics)
        idx = np.random.randint(0, len(ho_pred.hand_models))
        if not self.debug:
            pred_img = ho_pred.vis_img(idx, 300, 1200)
            # vis_img = np.concatenate((gt_img, pred_img), axis=1)
            vis_img = pred_img
            vis_img = wandb.Image(vis_img, caption='Sampled Result')
            self.logger.experiment.log({'Sample Results': vis_img}, step=batch_idx * batch_size + idx)

    def on_test_epoch_end(self):
        self.sample_joints = np.concatenate(self.sample_joints)
        entropy, cluster_size, entropy2, cluster_size2 = calc_diversity(self.sample_joints)
        self.logger.log_metrics({'Entropy': np.mean(entropy), 'Canonical Entropy': np.mean(entropy2),
                                'Cluster Size': np.mean(cluster_size), 'Canonical Cluster Size': np.mean(cluster_size2)})
        if not os.path.exists(self.tmp_dump_file):
            with open(self.tmp_dump_file, 'wb') as f:
                pickle.dump(self.sample_result, f)

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

        target_part = dorig['partition_object'].argmax(dim=-1).to(device)
        loss_part_rec = F.nll_loss(input=F.log_softmax(drec['partition_object'], dim=-1).float().permute(0, 2, 1),
                                   target=target_part.long(), reduction='none')
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
