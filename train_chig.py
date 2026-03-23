#!/usr/bin/env python
"""
AIMD-Chig Multi-GPU training script using eSCN-MD model
Combines eSCN-MD architecture with PyTorch Lightning for distributed training
"""

import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy

from datasets.chig_dataset import AIMDChigDataset
from fairchem.core.models.uma.escn_md_irreps import eSCNMDBackbone, MLP_EFS_Head


class ChigESCNLightningModule(pl.LightningModule):
    def __init__(self, config, task_mean, task_std):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Parse ewald_hyperparams
        ewald_hyperparams = None
        if config['model'].get('irreps', False) and 'ewald_hyperparams' in config:
            ewald_cfg = config['ewald_hyperparams']
            ewald_hyperparams = {
                'k_cutoff': ewald_cfg.get('k_cutoff'),
                'delta_k': ewald_cfg.get('delta_k'),
                'num_k_rbf': ewald_cfg.get('num_k_rbf'),
                'num_k_x': ewald_cfg.get('num_k_x'),
                'num_k_y': ewald_cfg.get('num_k_y'),
                'num_k_z': ewald_cfg.get('num_k_z'),
                'downprojection': ewald_cfg.get('downprojection', 128),
                'num_hidden': ewald_cfg.get('num_hidden', 2),
            }
            ewald_hyperparams = {k: v for k, v in ewald_hyperparams.items() if v is not None}
        
        # Initialize eSCN-MD backbone
        self.backbone = eSCNMDBackbone(
            max_num_elements=config['model'].get('num_elements', 90),
            sphere_channels=config['model'].get('sphere_channels', 128),
            lmax=config['model'].get('lmax', 2),
            mmax=config['model'].get('mmax', 2),
            cutoff=config['model'].get('cutoff', 5.0),
            max_neighbors=config['model'].get('max_neighbors', 300),
            num_layers=config['model'].get('num_layers', 4),
            hidden_channels=config['model'].get('hidden_channels', 128),
            edge_channels=config['model'].get('edge_channels', 128),
            regress_forces=True,
            direct_forces=False,  # Gradient-based forces
            use_pbc=False,
            always_use_pbc=False,
            otf_graph=False,
            dataset_list=["aimd_chig"],
            irreps=config['model'].get('irreps', False),
            ewald_hyperparams=ewald_hyperparams,
        )
        
        # Initialize EFS head
        self.head = MLP_EFS_Head(self.backbone, wrap_property=True)
        
        # Store normalization parameters
        self.register_buffer('task_mean', torch.tensor(task_mean, dtype=torch.float32))
        self.register_buffer('task_std', torch.tensor(task_std, dtype=torch.float32))
        
        # Loss functions (L1Loss/MAE for training, consistent with MD22)
        self.energy_criterion = nn.L1Loss()
        self.force_criterion = nn.L1Loss()
        
        # For collecting test results
        self.test_results = {'y_pred': [], 'y_true': [], 'dy_pred': [], 'dy_true': []}
    
    def convert_to_escn_format(self, data):
        """Convert AIMD-Chig data to eSCN-MD format."""
        if not hasattr(data, 'natoms'):
            data.natoms = torch.bincount(data.batch)
        
        if not hasattr(data, 'cell'):
            batch_size = int(data.batch.max()) + 1
            data.cell = torch.eye(3, device=data.pos.device, dtype=data.pos.dtype)
            data.cell = data.cell.unsqueeze(0).expand(batch_size, 3, 3).contiguous() * 100.0
        
        if not hasattr(data, 'charge'):
            batch_size = int(data.batch.max()) + 1
            data.charge = torch.zeros(batch_size, device=data.pos.device, dtype=data.pos.dtype)
        
        if not hasattr(data, 'spin'):
            batch_size = int(data.batch.max()) + 1
            data.spin = torch.zeros(batch_size, device=data.pos.device, dtype=data.pos.dtype)
        
        if not hasattr(data, 'dataset'):
            batch_size = int(data.batch.max()) + 1
            data.dataset = torch.zeros(batch_size, dtype=torch.long, device=data.pos.device)
        
        return data
    
    def normalize_data(self, data):
        """Normalize energy by subtracting task_mean."""
        data.y_original = data.y.clone()
        data.force_original = data.force.clone()
        # Normalize energy by subtracting mean
        data.y = data.y - self.task_mean
        return data
    
    def denormalize_predictions(self, pred_energy, pred_force):
        """Denormalize energy by adding back task_mean."""
        pred_energy = pred_energy + self.task_mean
        return pred_energy, pred_force
    
    def forward(self, data):
        """Forward pass through backbone and head."""
        data = self.convert_to_escn_format(data)
        # data = self.normalize_data(data)
        
        # Backbone (gradient will be enabled inside backbone/head as needed)
        emb = self.backbone(data)
        
        # Head (force computation happens here via autograd)
        outputs = self.head(data, emb)
        
        pred_energy = outputs['energy']['energy']
        pred_forces = outputs['forces']['forces']
        
        return pred_energy, pred_forces, data
    
    def training_step(self, batch, batch_idx):
        pred_energy, pred_forces, data = self(batch)
        
        # Losses (L1Loss/MAE)
        loss_e = self.energy_criterion(pred_energy, (data.y - self.task_mean))
        loss_f = self.force_criterion(pred_forces, data.force)
        loss = self.config['training']['energy_weight'] * loss_e + self.config['training']['force_weight'] * loss_f
        
        # MAE in kcal/mol
        e_mae = torch.mean(torch.abs(pred_energy.detach() + self.task_mean - data.y)).item()
        f_mae = torch.mean(torch.abs(pred_forces.detach() - data.force)).item()
        
        batch_size = pred_energy.size(0)
        
        # Log metrics
        self.log('train/loss_energy', loss_e, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log('train/loss_force', loss_f, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log('train/loss_total', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('train/e_mae', e_mae, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('train/f_mae', f_mae, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, prog_bar=False, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred_energy, pred_forces, data = self(batch)
        
        # Losses (L1Loss/MAE)
        loss_e = self.energy_criterion(pred_energy, (data.y - self.task_mean))
        loss_f = self.force_criterion(pred_forces, data.force)
        loss = self.config['training']['energy_weight'] * loss_e + self.config['training']['force_weight'] * loss_f
        
        # MAE in kcal/mol
        e_mae = torch.mean(torch.abs(pred_energy.detach() + self.task_mean - data.y)).item()
        f_mae = torch.mean(torch.abs(pred_forces.detach() - data.force)).item()

        batch_size = pred_energy.size(0)
        
        # Log metrics
        self.log('val/loss_energy', loss_e, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log('val/loss_force', loss_f, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log('val/loss_total', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('val/e_mae', e_mae, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('val/f_mae', f_mae, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        pred_energy, pred_forces, data = self(batch)
        
        loss_e = self.energy_criterion(pred_energy, (data.y - self.task_mean))
        loss_f = self.force_criterion(pred_forces, data.force)
        loss = self.config['training']['energy_weight'] * loss_e + self.config['training']['force_weight'] * loss_f
        
        # MAE in kcal/mol
        e_mae = torch.mean(torch.abs(pred_energy.detach() + self.task_mean - data.y)).item()
        f_mae = torch.mean(torch.abs(pred_forces.detach() - data.force)).item()
        # Collect predictions
        self.test_results['y_pred'].append(pred_energy.detach().cpu())
        self.test_results['y_true'].append(data.y.detach().cpu())
        self.test_results['dy_pred'].append(pred_forces.detach().cpu())
        self.test_results['dy_true'].append(data.force.detach().cpu())

        # Get batch size for logging
        batch_size = pred_energy.size(0)
        
        # Log metrics
        self.log('test/loss_energy', loss_e, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log('test/loss_force', loss_f, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log('test/loss_total', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('test/e_mae', e_mae, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('test/f_mae', f_mae, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        return loss
    
    def on_test_epoch_end(self):
        # Concatenate all test results
        if len(self.test_results['y_pred']) > 0:
            self.test_results['y_pred'] = torch.cat(self.test_results['y_pred'], dim=0)
            self.test_results['y_true'] = torch.cat(self.test_results['y_true'], dim=0)
            self.test_results['dy_pred'] = torch.cat(self.test_results['dy_pred'], dim=0)
            self.test_results['dy_true'] = torch.cat(self.test_results['dy_true'], dim=0)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Warmup scheduler
        warmup_steps = self.config['training'].get('warmup_steps', 1000)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


class ChigDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.task_mean = None  # Will be computed from training data
        self.task_std = 1.0    # No normalization for std
        
        print("=" * 80)
        print("ChigDataModule initialized (kcal/mol):")
        print(f"  Batch size: {config['training']['batch_size']}")
        print(f"  Eval batch size: {config['training']['eval_batch_size']}")
        print("=" * 80)
    
    def prepare_data(self):
        """Preprocess all splits to cache."""
        print("=" * 80)
        print("Preparing data (will cache processed datasets)...")
        print("=" * 80)
        
        for split_key in ["train_id", "vali_id", "test_id"]:
            try:
                _ = AIMDChigDataset(
                    self.config['data']['root'],
                    split_npz=self.config['data']['split_npz'],
                    split_key=split_key,
                    validate_data=True,
                    use_cache=True
                )
            except Exception as e:
                print(f"Warning: Failed to prepare {split_key}: {e}")
    
    def setup(self, stage=None):
        """Load cached datasets and compute task_mean from training set."""
        print("=" * 80)
        print("Loading datasets (kcal/mol units)...")
        print("=" * 80)
        
        # Load datasets
        self.train_set = AIMDChigDataset(
            self.config['data']['root'],
            split_npz=self.config['data']['split_npz'],
            split_key="train_id",
            validate_data=False,
            use_cache=True
        )
        self.val_set = AIMDChigDataset(
            self.config['data']['root'],
            split_npz=self.config['data']['split_npz'],
            split_key="vali_id",
            validate_data=False,
            use_cache=True
        )
        self.test_set = AIMDChigDataset(
            self.config['data']['root'],
            split_npz=self.config['data']['split_npz'],
            split_key="test_id",
            validate_data=False,
            use_cache=True
        )
        
        # Calculate task_mean from training set
        print("\nCalculating task_mean from training set...")
        y_list = []
        for data in self.train_set:
            if data.y is not None:
                y_list.append(data.y)
        
        if len(y_list) == 0:
            raise ValueError("No valid energy values found in training set!")
        
        y_tensor = torch.cat(y_list, dim=0)
        self.task_mean = float(y_tensor.mean())
        energy_std = float(y_tensor.std())
        
        print("=" * 80)
        print("Dataset Statistics:")
        print(f"  Train: {len(self.train_set)} samples")
        print(f"  Val:   {len(self.val_set)} samples")
        print(f"  Test:  {len(self.test_set)} samples")
        print(f"  Energy mean (task_mean): {self.task_mean:.2f} kcal/mol")
        print(f"  Energy std:              {energy_std:.2f} kcal/mol")
        print(f"  Units: kcal/mol")
        print(f"  Note: task_mean will be used for normalization during training")
        print("=" * 80)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            persistent_workers=True if self.config['training']['num_workers'] > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.config['training']['eval_batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            persistent_workers=True if self.config['training']['num_workers'] > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.config['training']['eval_batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            persistent_workers=True if self.config['training']['num_workers'] > 0 else False
        )


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_args():
    parser = argparse.ArgumentParser(description="AIMD-Chig Multi-GPU Training with eSCN-MD")
    parser.add_argument("--config", type=str, default="configs/chig/chig_escn.yaml", help="Path to config YAML file")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU indices (e.g., '0,1,2')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--irreps", action="store_true", help="Enable Ewald blocks")
    parser.add_argument("--no-irreps", action="store_true", help="Disable Ewald blocks")
    return parser.parse_args()


def main():
    args = get_args()
    
    # Set GPU devices FIRST, before any CUDA operations
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print(f"Setting CUDA_VISIBLE_DEVICES to: {args.gpus}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.seed is not None:
        config['data']['seed'] = args.seed
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.irreps:
        config['model']['irreps'] = True
    if args.no_irreps:
        config['model']['irreps'] = False
    
    # Set random seed
    pl.seed_everything(config['data']['seed'], workers=True)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        accelerator = 'cpu'
        device_count = 1
    elif torch.cuda.device_count() == 0:
        print("No CUDA devices found, switching to CPU")
        accelerator = 'cpu'
        device_count = 1
    else:
        accelerator = 'gpu'
        default = ",".join(str(i) for i in range(torch.cuda.device_count()))
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
        device_count = len(cuda_visible_devices)
        print(f"Available GPUs: {device_count}")
    
    # Create output directory
    irreps_flag = "ewald" if config['model'].get('irreps', False) else "no_ewald"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"chig_escn_{irreps_flag}_ngpus_{device_count}_bs_{config['training']['batch_size']}_seed_{config['data']['seed']}_{timestamp}"
    
    log_dir = os.path.join(config['output']['base_dir'], dir_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_save_path = os.path.join(log_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to: {config_save_path}")
    
    # Initialize data module
    data_module = ChigDataModule(config)
    data_module.prepare_data()
    data_module.setup()
    
    # Initialize model
    model = ChigESCNLightningModule(config, data_module.task_mean, data_module.task_std)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 80)
    print(f"Model Parameters (eSCN-MD):")
    print(f"  Total:     {total_params/1e6:.3f}M")
    print(f"  Trainable: {trainable_params/1e6:.3f}M")
    print("=" * 80)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val/loss_total",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
        filename="epoch{epoch:03d}-val_loss{val/loss_total:.4f}",
        auto_insert_metric_name=False,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor="val/loss_total",
        patience=50,
        mode="min",
        verbose=True
    )
    
    # Loggers
    tb_logger = TensorBoardLogger(save_dir=log_dir, name="", version="", default_hp_metric=False)
    csv_logger = CSVLogger(save_dir=log_dir, name="", version="")
    
    # Trainer configuration
    trainer_kwargs = {
        "max_epochs": config['training']['epochs'],
        "accelerator": accelerator,
        "devices": device_count,
        "precision": "32-true",
        "callbacks": [checkpoint_callback, early_stopping],
        "logger": [tb_logger, csv_logger],
        "enable_progress_bar": True,
        "gradient_clip_val": config['training']['clip_grad'] if config['training']['clip_grad'] > 0 else None,
        "log_every_n_steps": 10,
        "inference_mode": False,  # Required: force computation uses torch.autograd.grad
    }
    
    # Only add DDP strategy for multi-GPU
    if accelerator == 'gpu' and device_count > 1:
        trainer_kwargs["strategy"] = DDPStrategy(
            find_unused_parameters=True,
            process_group_backend='nccl'
        )
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Print training info
    print("=" * 80)
    print("Training Configuration:")
    print(f"  Model: eSCN-MD + AIMNet2")
    print(f"  Ewald blocks: {config['model'].get('irreps', False)}")
    print(f"  Log directory: {log_dir}")
    print(f"  Accelerator: {accelerator}")
    print(f"  Devices: {device_count}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['lr']}")
    print(f"  Seed: {config['data']['seed']}")
    print("=" * 80)
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, datamodule=data_module)
    
    # Test
    print("\n" + "=" * 80)
    print("Testing with best checkpoint...")
    print("=" * 80)
    trainer.test(model, datamodule=data_module, ckpt_path="best")
    
    # Print final results
    if len(model.test_results['y_pred']) > 0:
        e_mae = torch.mean(torch.abs(model.test_results['y_pred'] - model.test_results['y_true'])).item()
        f_mae = torch.mean(torch.abs(model.test_results['dy_pred'] - model.test_results['dy_true'])).item()
        print("\n" + "=" * 80)
        print("[Final Test Results - eSCN-MD]")
        print(f"  Energy MAE: {e_mae:.4f} kcal/mol")
        print(f"  Force MAE:  {f_mae:.4f} kcal/(mol·Å)")
        print("=" * 80)
        
        # Save results
        results_path = os.path.join(log_dir, "test_results.txt")
        with open(results_path, 'w') as f:
            f.write(f"Energy MAE: {e_mae:.4f} kcal/mol\n")
            f.write(f"Force MAE:  {f_mae:.4f} kcal/(mol·Å)\n")
        print(f"Results saved to: {results_path}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
