import lightning as L
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import hydra
from omegaconf import OmegaConf
import torch
import datetime
from common.dataset_utils.datamodules import HOIDatasetModule
from common.dataset_utils.hoi4d_dataset import HOI4DHandDataModule
from common.model.vae.handvae import HandVAE, HandVAETrainer
from lightning.pytorch.callbacks import ModelCheckpoint
# from common.model.gridvqvae.gridvqvae import GRIDVQVAE

# Register OmegaConf resolvers for arithmetic operations
OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)
OmegaConf.register_new_resolver("sub", lambda x, y: x - y, replace=True)
OmegaConf.register_new_resolver("mul", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("div", lambda x, y: x / y, replace=True)

@hydra.main(config_path="../config", config_name="handvae")
def main(cfg):
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # wandb_key = ''
    # wandb.login(key=wandb_key)
    t = datetime.datetime.now()
    if cfg.debug:
        logger = TensorBoardLogger(save_dir='logs/tb_logs')
    else:
        import wandb
        logger = WandbLogger(name=cfg.ae.name+'-'+t.strftime('%Y%m%d-%H%M%S'),
                                    project='LG3DContact',
                                    log_model=True, save_dir='logs/wandb_logs')

    # Initialize the model, data module, and trainer
    model = HandVAE(cfg.handvae)
    checkpoint_callback = ModelCheckpoint(
        monitor='val/total_loss',
        dirpath=cfg.checkpoint.dirpath,
        filename=cfg.checkpoint.filename + '-{epoch:02d}-{val/total_loss:.4f}',
        save_top_k=cfg.checkpoint.save_top_k,
        mode='min',
        save_last=True,
    )

    data_module = HOIDatasetModule(cfg)
    # data_module = HOI4DHandDataModule(cfg)
    if cfg.run_phase == 'train':
        pl_trainer = HandVAETrainer(model, cfg)
        trainer = L.Trainer(**cfg.trainer, logger=logger, callbacks=[checkpoint_callback])
        # trainer.fit(pl_trainer, datamodule=data_module)
        trainer.fit(pl_trainer, datamodule=data_module, ckpt_path=cfg.train.get('resume_ckpt', None))
    elif cfg.run_phase == 'val':
        pl_trainer = HandVAETrainer.load_from_checkpoint(cfg.ckpt_path, model=model, cfg=cfg)
        trainer = L.Trainer(**cfg.trainer, logger=logger)
        trainer.validate(pl_trainer, datamodule=data_module)
    else:
        pl_trainer = HandVAETrainer.load_from_checkpoint(cfg.ckpt_path, model=model, cfg=cfg)
        trainer = L.Trainer(**cfg.trainer, logger=logger)
        trainer.test(pl_trainer, datamodule=data_module)
    
    # Start training
    # trainer.fit(pl_trainer, datamodule=data_module)


if __name__ == '__main__':
    main()