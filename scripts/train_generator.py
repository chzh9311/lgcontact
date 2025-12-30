import lightning as L
import hydra
from omegaconf import OmegaConf
from common.dataset_utils.datamodules import HOIDatasetModule
from common.model.gridae_trainer import MLCTrainer, DummyModel
import importlib

@hydra.main(config_path="../config", config_name="mlcontact_gen")
def main(cfg):
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize the model, data module, and trainer
    if cfg.generator.model_type == 'gt':
        model = DummyModel()
    else:
        generator_module = importlib.import_module(f"common.model.{cfg.generator.model_type}.{cfg.generator.model_name}")
        model = getattr(generator_module, cfg.generator.model_name.upper())
    pl_trainer = MLCTrainer(model, cfg)
    trainer = L.Trainer(**cfg.trainer, inference_mode=False)
    data_module = HOIDatasetModule(cfg)
    
    # Start training
    trainer.test(pl_trainer, datamodule=data_module)


if __name__ == '__main__':
    main()