import os
from dotenv import load_dotenv

load_dotenv()

import argparse
import yaml
import pathlib
from datetime import datetime

import torch
import torch.utils.data
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from pl_modules import get_module
from datasets import get_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compositional Representation Learning Experiment Runner"
    )
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path of experiment config file (.yaml)",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Run in test mode",
    )

    args = parser.parse_args()
    with open(args.config_filepath, "r") as file_handle:
        config = yaml.safe_load(file_handle)

    data_root_dir = os.getenv("DATA_ROOT_DIR")
    experiment_root_dir = os.getenv("EXPERIMENT_ROOT_DIR")

    experiment_name = pathlib.Path(args.config_filepath).stem
    experiment_type = config["model"]["type"]
    experiment_id = experiment_name + "___" + datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    if args.test_mode:
        experiment_id = f"test_{experiment_id}"
    experiment_dir = (
        pathlib.Path(experiment_root_dir) / experiment_type / experiment_name / experiment_id
    )
    model_checkpoint_dir = experiment_dir / "checkpoints"
    model_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pl_module = get_module(config)
    train_dataset = get_dataset(config)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, **config["data"]["train"]["dataloader_config"]
    )

    wandb_logger = WandbLogger(project="compositional-representation-learning", name=experiment_id)
    wandb_logger.experiment.config.update(config)
    experiment = wandb_logger.experiment
    experiment.watch(pl_module, log='all', log_freq=10) 

    model_checkpoint = ModelCheckpoint(
        dirpath=model_checkpoint_dir,
        filename="model-{epoch:03d}",
        every_n_epochs=1,
        save_top_k=-1,
    )

    trainer = L.Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        default_root_dir=experiment_dir,
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[model_checkpoint],
        gradient_clip_val=1.0,
    )

    trainer.fit(pl_module, train_dataloaders=train_dataloader)
