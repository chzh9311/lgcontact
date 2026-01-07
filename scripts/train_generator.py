import lightning as L
import hydra
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
import datetime
from omegaconf import OmegaConf
from common.dataset_utils.datamodules import HOIDatasetModule
# from common.model.mlctrainer import MLCTrainer
from common.model.vae.dummy import DummyModel
from common.utils.misc import set_seed
import importlib

OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)
@hydra.main(config_path="../config", config_name="mlcontact_gen")
def main(cfg):
    # Set seed for reproducibility FIRST
    if hasattr(cfg, 'seed'):
        set_seed(cfg.seed)
    else:
        set_seed(42)  # default seed

    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize the model, data module, and trainer
    if cfg.generator.model_type == 'gt':
        model = DummyModel()
    else:
        generator_module = importlib.import_module(f"common.model.{cfg.generator.model_type}.{cfg.generator.model_name}")
        model = getattr(generator_module, cfg.generator.model_name.upper())(cfg)
    
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

    # Use DDPStrategy with static_graph=True to handle frozen parameters without memory overhead
    # This is optimal when model has frozen weights (e.g., pretrained autoencoder)
    if cfg.trainer.get('devices', 1) > 1 and cfg.trainer.get('accelerator') == 'gpu':
        strategy = DDPStrategy(static_graph=True)
        trainer = L.Trainer(**cfg.trainer, strategy=strategy, inference_mode=inference_mode, logger=logger)
    else:
        trainer = L.Trainer(**cfg.trainer, inference_mode=inference_mode, logger=logger)

    data_module = HOIDatasetModule(cfg)

    if cfg.contact_unit == 'grid':
        from common.model.mlctrainer import MLCTrainer
        trainer_module = MLCTrainer
    elif cfg.contact_unit == 'point':
        from common.model.sctrainer import SCTrainer
        trainer_module = SCTrainer

    # Start training
    if cfg.run_phase == 'train':
        pl_model = trainer_module(model, cfg)
        trainer.fit(pl_model, datamodule=data_module, ckpt_path=cfg.train.get('resume_ckpt', None))
    elif cfg.run_phase == 'val':
        pl_model = trainer_module(model, cfg)
        trainer.validate(pl_model, datamodule=data_module)
    elif cfg.generator.model_type == 'gt' and cfg.run_phase == 'test':
        pl_model = trainer_module(model, cfg)
        trainer.test(pl_model, datamodule=data_module)
    else:
        # Make config temporarily writable for checkpoint loading
        # pl_model = trainer_module.load_from_checkpoint(cfg.ckpt_path, model=model, cfg=cfg)
        pl_model = trainer_module(model, cfg)
        trainer.test(pl_model, datamodule=data_module)


if __name__ == '__main__':
    main()