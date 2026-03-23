#!/usr/bin/env python3
"""
MD22 training script for eSCN-MD Irreps model with Multi-GPU support
"""
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Iterable, Dict
import yaml
import shutil
import logging
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datasets.md22_dataset import MD22_ESCN
from fairchem.core.models.uma.escn_md_irreps import eSCNMDBackbone, MLP_EFS_Head


def setup_logging(log_dir, run_name, rank=0):
    """Setup logging to both file and console."""
    log_file = os.path.join(log_dir, f"{run_name}_training_rank{rank}.log")
    
    logger = logging.getLogger(f"rank{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    if rank == 0:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger, log_file


def setup_distributed(rank, world_size, backend='nccl'):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum, self.cnt = 0.0, 0
    @property
    def avg(self):
        return self.sum / max(1, self.cnt)
    def update(self, val, n=1):
        self.sum += float(val) * int(n)
        self.cnt += int(n)


def get_args():
    parser = argparse.ArgumentParser("MD22 + eSCN-MD Irreps Multi-GPU (Fairchem UMA)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Mode: train or test")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for testing")
    parser.add_argument("--config", type=str, default="configs/md22/escn_md_irreps.yaml", help="Path to config YAML file")
    parser.add_argument("--root", type=str, default=None, help="Override dataset root directory")
    parser.add_argument("--molecule", type=str, default=None, help="Override molecule name")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override evaluation batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--use-ewald", action="store_true", help="Enable Ewald blocks")
    parser.add_argument("--no-ewald", action="store_true", help="Disable Ewald blocks")
    parser.add_argument("--gpus", type=str, default=None, help="GPU IDs to use (comma-separated)")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_args_with_config(args, config):
    """Merge command line arguments with config."""
    if args.root is not None:
        config['data']['root'] = args.root
    if args.molecule is not None:
        config['data']['molecule'] = args.molecule
    if args.seed is not None:
        config['data']['seed'] = args.seed
    
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.eval_batch_size is not None:
        config['training']['eval_batch_size'] = args.eval_batch_size
    if args.lr is not None:
        config['training']['lr'] = args.lr
    
    if args.use_ewald:
        config['model']['ewald_hyperparams'] = config['model'].get('ewald_hyperparams', {
            "num_k_x": 2,
            "num_k_y": 2,
            "num_k_z": 2,
            "downprojection_size": 64,
            "num_hidden": 2,
        })
    if args.no_ewald:
        config['model']['ewald_hyperparams'] = None
    
    if args.gpus is not None:
        gpu_list = [int(x.strip()) for x in args.gpus.split(',')]
        config['system']['gpus'] = gpu_list
    
    return config


def count_params(model: torch.nn.Module, print_top_k: int = 15, logger=None):
    """Count model parameters."""
    if isinstance(model, DDP):
        model = model.module
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    log_fn = logger.info if logger else print
    log_fn(f"[Params] total={total/1e6:.3f}M, trainable={trainable/1e6:.3f}M")

    bucket = []
    for name, m in model.named_children():
        n = sum(p.numel() for p in m.parameters())
        bucket.append((name, n))
    bucket.sort(key=lambda x: x[1], reverse=True)
    for name, n in bucket[:print_top_k]:
        log_fn(f"  - {name:<24} {n/1e6:>7.3f}M")


def _log_step_scalars(writer: SummaryWriter, split: str, scalars: Dict[str, float], step: int):
    """Write scalars to TensorBoard."""
    if writer is None:
        return
    for k, v in scalars.items():
        writer.add_scalar(f"{split}/{k}", float(v), step)


def convert_to_escn_format(data):
    """Ensure data has all required fields for eSCN-MD."""
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
    
    if not hasattr(data, 'pbc'):
        batch_size = int(data.batch.max()) + 1
        data.pbc = torch.zeros(batch_size, 3, dtype=torch.bool, device=data.pos.device)
    
    return data


def train_one_epoch(
    config,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    energy_criterion: nn.Module,
    force_criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter = None,
    global_step: int = 0,
    print_freq: int = 100,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    rank: int = 0,
    world_size: int = 1,
):
    backbone.train()
    head.train()
    energy_criterion.train()
    force_criterion.train()

    loss_meters = {"energy": AverageMeter(), "force": AverageMeter(), "total": AverageMeter()}
    mae_meters  = {"energy": AverageMeter(), "force": AverageMeter()}
    
    train_cfg = config['training']
    t0 = time.perf_counter()

    for step, data in enumerate(data_loader):
        task_mean = config['normalization'][0]
        data = data.to(device)
        data = convert_to_escn_format(data)

        emb = backbone(data)
        outputs = head(data, emb)
        
        pred_energy = outputs['energy']['energy']
        pred_forces = outputs['forces']['forces']

        if torch.isnan(pred_energy).any() or torch.isnan(pred_forces).any():
            if rank == 0:
                print(f"WARNING: NaN detected! Skipping batch.")
            continue
        
        loss_e = energy_criterion(pred_energy, (data.y - task_mean))
        loss_f = force_criterion(pred_forces, data.force)
        loss = train_cfg['energy_weight'] * loss_e + train_cfg['force_weight'] * loss_f

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg['clip_grad'] and train_cfg['clip_grad'] > 0:
            if world_size > 1:
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.module.parameters()) + list(head.module.parameters()), 
                    train_cfg['clip_grad']
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()), 
                    train_cfg['clip_grad']
                )
        optimizer.step()

        e_mae = torch.mean(torch.abs(pred_energy.detach() + task_mean - data.y)).item()
        f_mae = torch.mean(torch.abs(pred_forces.detach() - data.force)).item()

        bs = pred_energy.shape[0]
        loss_meters["energy"].update(loss_e.item(), n=bs)
        loss_meters["force"].update(loss_f.item(), n=pred_forces.shape[0])
        loss_meters["total"].update(loss.item(), n=bs)
        mae_meters["energy"].update(e_mae, n=bs)
        mae_meters["force"].update(f_mae, n=pred_forces.shape[0])

        if rank == 0:
            _log_step_scalars(writer, "train", {
                "loss_energy": loss_e.item(),
                "loss_force": loss_f.item(),
                "loss_total": loss.item(),
                "e_mae": e_mae,
                "f_mae": f_mae,
                "lr": optimizer.param_groups[0]["lr"],
            }, global_step)

        if rank == 0 and ((step % print_freq == 0) or (step == len(data_loader) - 1)):
            elapsed = time.perf_counter() - t0
            speed_ms = 1e3 * elapsed / max(1, (step + 1))
            print(
                f"Epoch[{epoch}] Step[{step}/{len(data_loader)}] "
                f"loss_e={loss_meters['energy'].avg:.5f} "
                f"loss_f={loss_meters['force'].avg:.5f} "
                f"loss_total={loss_meters['total'].avg:.5f} "
                f"e_MAE={mae_meters['energy'].avg:.5f} "
                f"f_MAE={mae_meters['force'].avg:.5f} "
                f"{speed_ms:.1f}ms/step"
            )

        global_step += 1

    if rank == 0:
        _log_step_scalars(writer, "train", {
            "epoch_loss_energy": loss_meters["energy"].avg,
            "epoch_loss_force": loss_meters["force"].avg,
            "epoch_loss_total": loss_meters["total"].avg,
            "epoch_e_mae": mae_meters["energy"].avg,
            "epoch_f_mae": mae_meters["force"].avg,
        }, epoch)

    return mae_meters, loss_meters, global_step


@torch.no_grad()
def evaluate(
    config,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    energy_criterion: nn.Module,
    force_criterion: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    writer: SummaryWriter = None,
    split_name: str = "val",
    global_step: int = 0,
    print_freq: int = 100,
    rank: int = 0,
):
    backbone.eval()
    head.eval()
    energy_criterion.eval()
    force_criterion.eval()

    loss_meters = {"energy": AverageMeter(), "force": AverageMeter(), "total": AverageMeter()}
    mae_meters  = {"energy": AverageMeter(), "force": AverageMeter()}
    
    train_cfg = config['training']
    t0 = time.perf_counter()

    for step, data in enumerate(data_loader):
        data = data.to(device)
        data = convert_to_escn_format(data)
        
        task_mean = config['normalization'][0]

        emb = backbone(data)
        outputs = head(data, emb)
        
        pred_energy = outputs['energy']['energy']
        pred_forces = outputs['forces']['forces']

        loss_e = energy_criterion(pred_energy, (data.y - task_mean))
        loss_f = force_criterion(pred_forces, data.force)
        loss = train_cfg['energy_weight'] * loss_e + train_cfg['force_weight'] * loss_f

        e_mae = torch.mean(torch.abs(pred_energy.detach() + task_mean - data.y)).item()
        f_mae = torch.mean(torch.abs(pred_forces.detach() - data.force)).item()

        bs = pred_energy.shape[0]
        loss_meters["energy"].update(loss_e.item(), n=bs)
        loss_meters["force"].update(loss_f.item(), n=pred_forces.shape[0])
        loss_meters["total"].update(loss.item(), n=bs)
        mae_meters["energy"].update(e_mae, n=bs)
        mae_meters["force"].update(f_mae, n=pred_forces.shape[0])

        if rank == 0:
            _log_step_scalars(writer, split_name, {
                "loss_energy": loss_e.item(),
                "loss_force": loss_f.item(),
                "loss_total": loss.item(),
                "e_mae": e_mae,
                "f_mae": f_mae,
            }, global_step)
        global_step += 1

        if rank == 0 and ((step % print_freq == 0) or (step == len(data_loader) - 1)):
            elapsed = time.perf_counter() - t0
            speed_ms = 1e3 * elapsed / max(1, (step + 1))
            print(
                f"[{split_name}] Step[{step}/{len(data_loader)}] "
                f"loss_e={loss_meters['energy'].avg:.5f} "
                f"loss_f={loss_meters['force'].avg:.5f} "
                f"loss_total={loss_meters['total'].avg:.5f} "
                f"e_MAE={mae_meters['energy'].avg:.5f} "
                f"f_MAE={mae_meters['force'].avg:.5f} "
                f"{speed_ms:.1f}ms/step"
            )

    if rank == 0:
        _log_step_scalars(writer, split_name, {
            "epoch_loss_energy": loss_meters["energy"].avg,
            "epoch_loss_force": loss_meters["force"].avg,
            "epoch_loss_total": loss_meters["total"].avg,
            "epoch_e_mae": mae_meters["energy"].avg,
            "epoch_f_mae": mae_meters["force"].avg,
        }, 0)

    return mae_meters, loss_meters, global_step


def train_worker(rank, world_size, config):
    """Worker function for each GPU process."""
    if world_size > 1:
        torch.cuda.set_device(rank)
        setup_distributed(rank, world_size, backend=config['system'].get('dist_backend', 'nccl'))
    else:
        torch.cuda.set_device(0)
    
    gpu_ids = config['system'].get('gpus', list(range(world_size)))
    
    data_cfg = config['data']
    train_cfg = config['training']
    model_cfg = config['model']
    output_cfg = config['output']
    system_cfg = config['system']
    
    ewald_flag = "ewald" if model_cfg.get('ewald_hyperparams') is not None else "no_ewald"
    
    timestamp = config.get('timestamp', time.strftime("%Y%m%d-%H%M%S"))
    run_name = f"{data_cfg['molecule']}_escn_md_irreps_{ewald_flag}_{timestamp}"
    run_dir = os.path.join(output_cfg['base_dir'], run_name)

    ckpt_dir = os.path.join(run_dir, output_cfg['ckpt_dir'])
    log_dir = os.path.join(run_dir, output_cfg['log_dir'])

    if rank == 0:
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    logger, log_file = setup_logging(log_dir, run_name, rank=rank)
    
    if rank == 0:
        logger.info("="*80)
        logger.info(f"MD22 Multi-GPU Training with eSCN-MD Irreps (world_size={world_size})")
        logger.info("="*80)
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Ewald blocks: {ewald_flag}")
        logger.info(f"GPUs: {system_cfg.get('gpus', list(range(world_size)))}")
    
    writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None

    np.random.seed(data_cfg['seed'] + rank)
    torch.manual_seed(data_cfg['seed'] + rank)
    torch.cuda.manual_seed_all(data_cfg['seed'] + rank)
    
    if system_cfg['deterministic']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        logger.info(f"Using device: {device}")

    if rank == 0:
        logger.info("Loading datasets...")
    
    train_set = MD22_ESCN(data_cfg['root'], data_cfg['molecule'], split="train", seed=data_cfg['seed'])
    val_set   = MD22_ESCN(data_cfg['root'], data_cfg['molecule'], split="val",   seed=data_cfg['seed'])
    test_set  = MD22_ESCN(data_cfg['root'], data_cfg['molecule'], split="test",  seed=data_cfg['seed'])

    if rank == 0:
        logger.info(f"Train set: {len(train_set)} samples")
        logger.info(f"Val set:   {len(val_set)} samples")
        logger.info(f"Test set:  {len(test_set)} samples")
    
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_set, 
        batch_size=train_cfg['batch_size'], 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=system_cfg['num_workers'],
        persistent_workers=True if system_cfg['num_workers'] > 0 else False,
        pin_memory=True
    )
    val_loader   = DataLoader(
        val_set,   
        batch_size=train_cfg['eval_batch_size'], 
        shuffle=False,
        sampler=val_sampler, 
        num_workers=system_cfg['num_workers'],
        persistent_workers=True if system_cfg['num_workers'] > 0 else False,
        pin_memory=True
    )
    test_loader  = DataLoader(
        test_set,  
        batch_size=train_cfg['eval_batch_size'], 
        shuffle=False,
        sampler=test_sampler, 
        num_workers=system_cfg['num_workers'],
        persistent_workers=True if system_cfg['num_workers'] > 0 else False,
        pin_memory=True
    )

    if rank == 0:
        logger.info("Creating eSCN-MD Irreps model...")
    
    backbone = eSCNMDBackbone(
        max_num_elements=model_cfg.get('num_elements', 90),
        sphere_channels=model_cfg.get('sphere_channels', 128),
        lmax=model_cfg.get('lmax', 3),
        mmax=model_cfg.get('mmax', 2),
        cutoff=model_cfg.get('cutoff', 5.0),
        max_neighbors=model_cfg.get('max_neighbors', 300),
        num_layers=model_cfg.get('num_layers', 4),
        hidden_channels=model_cfg.get('hidden_channels', 128),
        edge_channels=model_cfg.get('edge_channels', 128),
        regress_forces=True,
        direct_forces=False,
        use_pbc=False,
        always_use_pbc=False,
        otf_graph=False,
        dataset_list=[data_cfg['molecule']],
        ewald_hyperparams=model_cfg.get('ewald_hyperparams'),
    ).to(device)
    
    head = MLP_EFS_Head(backbone, wrap_property=True).to(device)
    
    if world_size > 1:
        backbone = DDP(
            backbone, 
            device_ids=[rank], 
            find_unused_parameters=system_cfg.get('find_unused_parameters', False)
        )
        head = DDP(
            head, 
            device_ids=[rank], 
            find_unused_parameters=system_cfg.get('find_unused_parameters', False)
        )

    if rank == 0:
        logger.info("Model parameters:")
        logger.info("Backbone:")
        count_params(backbone, logger=logger)
        logger.info("Head:")
        count_params(head, logger=logger)
        
        total_params = sum(p.numel() for p in (backbone.module if world_size > 1 else backbone).parameters())
        total_params += sum(p.numel() for p in (head.module if world_size > 1 else head).parameters())
        logger.info(f"Total parameters: {total_params/1e6:.3f}M")

    energy_criterion = nn.L1Loss()
    force_criterion = nn.L1Loss()
    
    all_params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=train_cfg.get('lr_factor', 0.8),
        patience=train_cfg.get('lr_patience', 30),
        min_lr=train_cfg.get('lr_min', 1e-7),
    )

    global_steps = {"train": 0, "val": 0, "test": 0}

    if rank == 0:
        logger.info("="*80)
        logger.info("Starting training...")
        logger.info("="*80)
    
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_mae = {"energy": 0.0, "force": 0.0}
    
    for epoch in range(1, train_cfg['epochs'] + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        tr_mae, tr_loss, global_steps["train"] = train_one_epoch(
            config, backbone, head, energy_criterion, force_criterion,
            train_loader, optimizer, device, epoch,
            writer=writer, global_step=global_steps["train"], 
            print_freq=train_cfg['print_freq'], scheduler=scheduler,
            rank=rank, world_size=world_size
        )
        
        va_mae, va_loss, global_steps["val"] = evaluate(
            config, backbone, head, energy_criterion, force_criterion,
            val_loader, device, writer=writer, split_name="val",
            global_step=global_steps["val"], print_freq=train_cfg['print_freq'],
            rank=rank
        )
        
        val_loss_total = va_loss["total"].avg
        scheduler.step(val_loss_total)
        
        if rank == 0:
            writer.add_scalars("epoch_summary/eMAE", {
                "train_energy": tr_mae["energy"].avg,
                "train_force": tr_mae["force"].avg,
                "val_energy": va_mae["energy"].avg,
                "val_force": va_mae["force"].avg,
            }, epoch)
            writer.add_scalars("epoch_summary/loss", {
                "train_energy": tr_loss["energy"].avg,
                "train_force": tr_loss["force"].avg,
                "train_total": tr_loss["total"].avg,
                "val_energy": va_loss["energy"].avg,
                "val_force": va_loss["force"].avg,
                "val_total": va_loss["total"].avg,
            }, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("train/learning_rate_epoch", current_lr, epoch)

            logger.info(
                f"Epoch {epoch:03d} | "
                f"train: loss={tr_loss['total'].avg:.6f} (E:{tr_loss['energy'].avg:.6f} F:{tr_loss['force'].avg:.6f}) "
                f"eMAE={tr_mae['energy'].avg:.4f} fMAE={tr_mae['force'].avg:.4f} | "
                f"val: loss={va_loss['total'].avg:.6f} (E:{va_loss['energy'].avg:.6f} F:{va_loss['force'].avg:.6f}) "
                f"eMAE={va_mae['energy'].avg:.4f} fMAE={va_mae['force'].avg:.4f} | "
                f"lr={current_lr:.2e}"
            )

            if val_loss_total < best_val_loss:
                best_val_loss = val_loss_total
                best_epoch = epoch
                best_val_mae = {
                    "energy": va_mae["energy"].avg,
                    "force": va_mae["force"].avg,
                }
                
                backbone_state = backbone.module.state_dict() if world_size > 1 else backbone.state_dict()
                head_state = head.module.state_dict() if world_size > 1 else head.state_dict()
                
                ckpt = {
                    "backbone_state_dict": backbone_state,
                    "head_state_dict": head_state,
                    "config": config,
                    "epoch": epoch,
                    "val_loss": val_loss_total,
                    "val_mae_energy": va_mae["energy"].avg,
                    "val_mae_force": va_mae["force"].avg,
                }
                torch.save(ckpt, os.path.join(ckpt_dir, f"best_{data_cfg['molecule']}.pt"))
                logger.info(f"  → Saved best model (val_loss: {val_loss_total:.6f})")
            
            if output_cfg['save_last']:
                backbone_state = backbone.module.state_dict() if world_size > 1 else backbone.state_dict()
                head_state = head.module.state_dict() if world_size > 1 else head.state_dict()
                
                ckpt_last = {
                    "backbone_state_dict": backbone_state,
                    "head_state_dict": head_state,
                    "config": config,
                    "epoch": epoch,
                }
                torch.save(ckpt_last, os.path.join(ckpt_dir, f"last_{data_cfg['molecule']}.pt"))
    
    if rank == 0:
        logger.info("")
        logger.info("="*80)
        logger.info("TRAINING COMPLETED - Loading best model for test evaluation")
        logger.info("="*80)
    
    best_ckpt_path = os.path.join(ckpt_dir, f"best_{data_cfg['molecule']}.pt")
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    
    if world_size > 1:
        backbone.module.load_state_dict(best_ckpt["backbone_state_dict"])
        head.module.load_state_dict(best_ckpt["head_state_dict"])
    else:
        backbone.load_state_dict(best_ckpt["backbone_state_dict"])
        head.load_state_dict(best_ckpt["head_state_dict"])
    
    if rank == 0:
        logger.info("Evaluating on test set with best model...")
    
    te_mae, te_loss, _ = evaluate(
        config, backbone, head, energy_criterion, force_criterion,
        test_loader, device, writer=writer, split_name="test",
        global_step=0, print_freq=train_cfg['print_freq'],
        rank=rank
    )
    
    if rank == 0:
        logger.info("")
        logger.info("="*80)
        logger.info("FINAL TEST RESULTS")
        logger.info("="*80)
        logger.info(f"Test Loss:   {te_loss['total'].avg:.6f}")
        logger.info(f"Test eMAE:   {te_mae['energy'].avg:.4f} kcal/mol")
        logger.info(f"Test fMAE:   {te_mae['force'].avg:.4f} kcal/mol/Å")
        logger.info("="*80)
        
        writer.add_scalar("test/final_loss_energy", te_loss["energy"].avg, 0)
        writer.add_scalar("test/final_loss_force", te_loss["force"].avg, 0)
        writer.add_scalar("test/final_loss_total", te_loss["total"].avg, 0)
        writer.add_scalar("test/final_e_mae", te_mae["energy"].avg, 0)
        writer.add_scalar("test/final_f_mae", te_mae["force"].avg, 0)
        
        summary_path = os.path.join(run_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING SUMMARY (eSCN-MD Irreps)\n")
            f.write("="*80 + "\n")
            f.write(f"Molecule: {data_cfg['molecule']}\n")
            f.write(f"Ewald blocks: {ewald_flag}\n")
            f.write(f"Total epochs: {train_cfg['epochs']}\n")
            f.write(f"Best Model (Epoch {best_epoch}):\n")
            f.write(f"  Val eMAE:   {best_val_mae['energy']:.4f} kcal/mol\n")
            f.write(f"  Val fMAE:   {best_val_mae['force']:.4f} kcal/mol/Å\n")
            f.write(f"Test Results:\n")
            f.write(f"  Test eMAE:  {te_mae['energy'].avg:.4f} kcal/mol\n")
            f.write(f"  Test fMAE:  {te_mae['force'].avg:.4f} kcal/mol/Å\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Results saved to: {run_dir}")
        
        writer.flush()
        writer.close()
    
    if world_size > 1:
        cleanup_distributed()


def main():
    args = get_args()
    
    if args.mode == "test" and args.checkpoint is None:
        print("ERROR: --checkpoint is required in test mode")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"ERROR: Config file {args.config} not found")
        sys.exit(1)
    config = load_config(args.config)
    
    config = merge_args_with_config(args, config)
    
    system_cfg = config['system']
    gpu_ids = system_cfg.get('gpus', [])
    if len(gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        print(f"Setting CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    config['timestamp'] = time.strftime("%Y%m%d-%H%M%S")
    
    data_cfg = config['data']
    print(f"Loading training set for normalization statistics...")
    
    try:
        train_set = MD22_ESCN(data_cfg['root'], data_cfg['molecule'], split="train", seed=data_cfg['seed'])
    except Exception as e:
        print(f"ERROR: Failed to load training dataset: {e}")
        sys.exit(1)
    
    energy_list = []
    for data in train_set:
        energy_list.append(data.y.item())
    
    mean = np.mean(energy_list)
    std = np.std(energy_list)

    print(f"Energy statistics - Mean: {mean:.2f} kcal/mol, Std: {std:.2f} kcal/mol")
    config['normalization'] = [mean, std]

    if args.mode == "test":
        print(f"\n{'='*80}")
        print(f"Running in TEST mode - Not implemented yet")
        print(f"{'='*80}\n")
        sys.exit(1)
    else:
        print(f"\n{'='*80}")
        print(f"Running in TRAIN mode")
        print(f"  Config file: {args.config}")
        print(f"  Molecule: {data_cfg['molecule']}")
        print(f"  Epochs: {config['training']['epochs']}")
        print(f"  GPUs: {system_cfg.get('gpus', [0])}")
        print(f"{'='*80}\n")
        
        if system_cfg.get('distributed', False) and len(system_cfg.get('gpus', [])) > 1:
            world_size = len(system_cfg['gpus'])
            print(f"Starting distributed training on {world_size} GPUs: {system_cfg['gpus']}")
            mp.spawn(
                train_worker,
                args=(world_size, config),
                nprocs=world_size,
                join=True
            )
        else:
            if system_cfg.get('cuda', True) and torch.cuda.is_available():
                if len(system_cfg.get('gpus', [])) > 0:
                    device_id = system_cfg['gpus'][0]
                    print(f"Starting single GPU training on GPU {device_id}")
                    torch.cuda.set_device(device_id)
            train_worker(0, 1, config)


if __name__ == "__main__":
    main()
