import lightning as L
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import hydra
from omegaconf import OmegaConf
import torch
import datetime
import importlib
from common.dataset_utils.datamodules import LocalGridDataModule
from common.model.lgtrainer import LGTrainer
# from common.model.gridvqvae.gridvqvae import GRIDVQVAE

# Register OmegaConf resolvers for arithmetic operations
OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)
OmegaConf.register_new_resolver("sub", lambda x, y: x - y, replace=True)
OmegaConf.register_new_resolver("mul", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("div", lambda x, y: x / y, replace=True)

@hydra.main(config_path="../config", config_name="gridae")
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
        logger = WandbLogger(name=cfg.model.name+'-'+t.strftime('%Y%m%d-%H%M%S'),
                                    project='LG3DContact',
                                    log_model=True, save_dir='logs/wandb_logs')
    
    # Initialize the model, data module, and trainer
    model_name = cfg.model.name.upper()
    model_module = importlib.import_module(f"common.model.{model_name.lower()}.{model_name.lower()}")
    model_class = getattr(model_module, model_name)
    model = model_class(cfg.model)
    # input = torch.randn(3, cfg.model.in_dim, 8, 8, 8)
    # embedding_loss, recon, perplexity = model(input, verbose=True)
    data_module = LocalGridDataModule(cfg)
    if cfg.run_phase == 'train':
        pl_trainer = LGTrainer(model, cfg)
        trainer = L.Trainer(**cfg.trainer, logger=logger)
        trainer.fit(pl_trainer, datamodule=data_module)
    else:
        pl_trainer = LGTrainer.load_from_checkpoint(cfg.ckpt_path, model=model, cfg=cfg)
        trainer = L.Trainer(**cfg.trainer, logger=logger)
        trainer.test(pl_trainer, datamodule=data_module)
    
    # Start training
    # trainer.fit(pl_trainer, datamodule=data_module)


if __name__ == '__main__':
    main()