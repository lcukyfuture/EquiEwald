"""
OC20 Model Training and Validation on val-id split
Only validates on val-id split where both adsorbate and catalyst compositions are in-distribution.
"""

import os
import argparse
import torch
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", 0))

import torch_geometric
import logging
from pathlib import Path
from tqdm import tqdm

from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.utils import setup_logging, load_config
from ocpmodels.datasets import LmdbDataset
from ocpmodels.common.registry import registry
from ocpmodels.trainers import ForcesTrainer


def main():
    """Main function for OC20 training and val-id validation"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/oc20/escn_ewald.yml')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('OC20 training requires an NVIDIA CUDA GPU. Use the CPU smoke tests to check the model installation.')
    torch.cuda.set_device(local_rank)

    # Initialize distributed training if not already initialized
    if not dist.is_initialized():
        if "RANK" not in os.environ:
            os.environ["RANK"] = str(local_rank)
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["RANK"])
        )

    # Setup logging
    setup_logging()

    # Load config
    config_path = args.config

    torch.cuda.empty_cache()
    conf = load_config(config_path)[0]
    task = conf["fixed"]["task"]
    model = conf["fixed"]["model"]
    optimizer = conf["fixed"]["optimizer"]
    name = conf["fixed"]["name"]
    logger_config = conf["fixed"]["logger"]

    # dataset_train contains the training set and combination of all four validation splits
    dataset = conf["fixed"]["dataset_train"]
    # val-id split where both adsorbate and catalyst compositions are in-distribution
    dataset_id = conf["fixed"]["dataset_id"]

    # Initialize trainer for training
    trainer = ForcesTrainer(
        task=task,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        identifier=name,
        run_dir="./",
        is_debug=False,
        print_every=5000,
        seed=0,
        logger=logger_config,
        local_rank=local_rank,
        amp=True,
    )

    # Train model
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # Load best checkpoint from training
    checkpoint_path = os.path.join(
        trainer.config["cmd"]["checkpoint_dir"], "best_checkpoint.pt"
    )
    print(f"Loading best checkpoint from: {checkpoint_path}")

    # Validate on val-id
    print("Initializing validator for val-id split...")
    validator = ForcesTrainer(
        task=task,
        model=model,
        dataset=dataset_id,
        optimizer=optimizer,
        identifier="validate_id",
        run_dir="./",
        is_debug=True,
        print_every=5000,
        seed=0,
        logger=logger_config,
        local_rank=local_rank,
        amp=True,
    )

    validator.load_checkpoint(checkpoint_path=checkpoint_path)
    print("Running validation on val-id split...")
    metrics = validator.validate()
    results_id = {key: val["metric"] for key, val in metrics.items()}
    print(f"val-id results for configuration {name}: {results_id}")

    return results_id


if __name__ == "__main__":
    results = main()
