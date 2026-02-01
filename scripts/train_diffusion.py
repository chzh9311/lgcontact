import torch
import lightning as L
import hydra
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
import datetime
from omegaconf import OmegaConf
from common.dataset_utils.datamodules import HOIDatasetModule
# from common.model.mlctrainer import MLCTrainer
from common.model.diff.dm.ddpm import DDPM
import common.model.diff.mdm.gaussian_diffusion as mdm_gd
from common.model.diff.unet import UNetModel
from common.model.gridae.gridae import GRIDAE
from common.model.lgcdifftrainer import LGCDiffTrainer
from common.utils.misc import set_seed, load_pl_ckpt
from lightning.pytorch.callbacks import ModelCheckpoint

OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)
@hydra.main(config_path="../config", config_name="mlcdiff")
def main(cfg):
    # Set seed for reproducibility FIRST
    if hasattr(cfg, 'seed'):
        set_seed(cfg.seed)
    else:
        set_seed(42)  # default seed

    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize the model, data module, and trainer
    # generator_module = importlib.import_module(f"common.model.{cfg.generator.model_type}.{cfg.generator.model_name}")
    # model = getattr(generator_module, cfg.generator.model_name.upper())(cfg)
    model = UNetModel(cfg.generator.unet)
    # MDM GaussianDiffusion
    mdm_cfg = cfg.generator.mdm
    diffusion = mdm_gd.GaussianDiffusion(
        timesteps=mdm_cfg.timesteps,
        schedule_cfg=mdm_cfg.schedule_cfg,
        model_mean_type=mdm_gd.ModelMeanType[mdm_cfg.model_mean_type.upper()],
        model_var_type=mdm_gd.ModelVarType[mdm_cfg.model_var_type.upper()],
        rand_t_type=mdm_cfg.rand_t_type,
        rescale_timesteps=mdm_cfg.rescale_timesteps,
        msdf_cfg=cfg.msdf)
    # DDPM (the base version of diffusion)
    # diffusion1 = DDPM(cfg.generator.ddpm)
    gridae = GRIDAE(cfg.ae, obj_1d_feat=True)
    
    t = datetime.datetime.now()
    if cfg.debug:
        logger = TensorBoardLogger(save_dir='logs/tb_logs')
    else:
        import wandb
        logger = WandbLogger(name=cfg.generator.model_name + '-' + cfg.run_phase + '-' + t.strftime('%Y%m%d-%H%M%S'),
                             project='LG3DContact',
                             log_model=True, save_dir='logs/wandb_logs')

    # Only disable inference_mode for testing (needed for pose optimization gradients)
    inference_mode = False if cfg.run_phase == 'test' else True
    checkpoint_callback = ModelCheckpoint(
        monitor='val/total_loss',
        dirpath=cfg.checkpoint.dirpath,
        filename=cfg.checkpoint.filename + '-{epoch:02d}-{val/total_loss:.4f}',
        save_top_k=cfg.checkpoint.save_top_k,
        mode='min',
        save_last=True,
    )

    # Use DDPStrategy with static_graph=True to handle frozen parameters without memory overhead
    # This is optimal when model has frozen weights (e.g., pretrained autoencoder)
    if cfg.trainer.get('devices', 1) > 1 and cfg.trainer.get('accelerator') == 'gpu':
        strategy = DDPStrategy(static_graph=True)
        trainer = L.Trainer(**cfg.trainer, strategy=strategy, inference_mode=inference_mode, logger=logger, callbacks=[checkpoint_callback])
    else:
        trainer = L.Trainer(**cfg.trainer, inference_mode=inference_mode, logger=logger, callbacks=[checkpoint_callback])

    data_module = HOIDatasetModule(cfg)

    # Start training
    if cfg.run_phase == 'train':
        pl_model = LGCDiffTrainer(gridae, model, diffusion, cfg)
        trainer.fit(pl_model, datamodule=data_module, ckpt_path=cfg.train.get('resume_ckpt', None))
    elif cfg.run_phase == 'val':
        # pl_model = LGCDiffTrainer(gridae, model, cfg)
        pl_model = LGCDiffTrainer.load_from_checkpoint(cfg.val.get('ckpt_path', None), gridae, model, diffusion, cfg)
        trainer.validate(pl_model, datamodule=data_module)
    else:
        sd = torch.load(cfg.ckpt_path, map_location='cpu')['state_dict']
        print('total keys in ckpt:', len(sd.keys()))
        load_pl_ckpt(gridae, sd, prefix='grid_ae.')
        load_pl_ckpt(model, sd, prefix='model.eps_model.')
        unused_keys = [k for k in sd.keys() if not (k.startswith('grid_ae.') or k.startswith('model.eps_model.'))]
        print(f'Unused keys in ckpt: {unused_keys}')

        pl_model = LGCDiffTrainer(gridae, model, diffusion, cfg)
        # pl_model = LGCDiffTrainer.load_from_checkpoint(cfg.ckpt_path, grid_ae=gridae, model=model, cfg=cfg)
        trainer.test(pl_model, datamodule=data_module)


if __name__ == '__main__':
    main()