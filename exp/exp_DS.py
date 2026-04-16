"""Training and evaluation orchestration for CocycleNet experiments.

This module provides a PyTorch Lightning `LightningModule` wrapper and a
`trainer` helper to run experiments. The code here was refactored to be more
robust for public release: logging is used instead of prints, model lookup is
case-insensitive, and seeding is performed at runtime (not import time).
"""

import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch
from torch import nn
from dataset.Meta import DATA_META
from models import CocycleNet, DeepKoopman, LRAN
from utils import RMSELoss, HsLoss, HsLoss1D, compute_vpt
from pytorch_lightning.loggers import TensorBoardLogger
import os
from dataset.data_loader import DataModule
import json
from torch.optim.lr_scheduler import _LRScheduler
import math

logger = logging.getLogger(__name__)

model_dict = {
    'CocycleNet': CocycleNet,
    'DeepKoopman': DeepKoopman,
    'LRAN': LRAN,
}

class DualOneCycleScheduler(_LRScheduler):
    def __init__(
        self, optimizer, max_lrs, total_steps, pct_start=0.3, 
        anneal_strategy='cos', div_factors=None, final_div_factors=None, last_epoch=-1
    ):
        assert len(optimizer.param_groups) == 3, "Optimizer must have 3 param_groups: [Main, Expert, Gate]"

        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy

        # Unpack LR configurations
        self.lrs = []
        for i in range(3):
            max_lr = max_lrs[i]
            init_lr = max_lr / div_factors[i]
            final_lr = max_lr / final_div_factors[i]
            self.lrs.append({'init': init_lr, 'max': max_lr, 'final': final_lr})
            
        super().__init__(optimizer, last_epoch)

    def _get_annealed_lr(self, start, end, pct):
        if self.anneal_strategy == 'cos':
            cos_out = math.cos(math.pi * pct) + 1
            return end + (start - end) / 2.0 * cos_out
        return (end - start) * pct + start

    def _compute_step_lr(self, step, conf):
        step = min(step, self.total_steps)
        warmup_steps = self.pct_start * self.total_steps
        if step <= warmup_steps:
            return self._get_annealed_lr(conf['init'], conf['max'], step / warmup_steps)
        
        decay_pct = (step - warmup_steps) / (self.total_steps - warmup_steps)
        return self._get_annealed_lr(conf['max'], conf['final'], decay_pct)

    def get_lr(self):
        return [self._compute_step_lr(self.last_epoch, self.lrs[i]) for i in range(3)]

    def step(self, epoch=None):
        super().step(epoch)
    
class NeuralModel(pl.LightningModule):

    def __init__(self, args, steps_per_epoch):
        super(NeuralModel, self).__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.steps_per_epoch = steps_per_epoch
        self.meta = DATA_META[args.dataset]
        # initialize model
        model_module = None
        for key, mod in model_dict.items():
            if key.lower() == args.model.lower():
                model_module = mod
                break
        if model_module is None:
            raise ValueError(f"Unknown model '{args.model}'. Available: {list(model_dict.keys())}")
        self.model = model_module.model(self.args, self.meta).float()
        self.lyap_unit_per_step = self.meta.deltat * args.downsample * self.meta.mle

        # Loss functions
        self.train_loss = self._setup_loss(args)

        # Path management
        self.paths = {
            "train": f"./figures/train/{args.setting}",
            "val": f"./figures/Val/{args.setting}",
            "test": f"./figures/test/{args.setting}"
        }
        for p in self.paths.values(): os.makedirs(p, exist_ok=True)

        self.test_step_outputs = []
        
    def _setup_loss(self, args):
        if args.loss == 'mse': return nn.MSELoss()
        if args.loss == 'hs1d': return HsLoss1D(rel=args.rel, k=args.k)
        return HsLoss(rel=args.rel, k=args.k)
    
    def forward(self, x):
        x_hat, z, z_l = self.model(x)
        return self.model.normalizer.decode(x_hat), z, z_l
        
    def on_train_epoch_start(self):
        # Ensure training horizon matches sequence length
        self.model.update_steps(self.args.seq_len)

    def training_step(self, batch, batch_idx):
        x = batch
        x_target = x[:, :self.args.seq_len]
        # Model forward
        x_hat, _, _ = self(x_target)

        # Calculate losses in normalized space
        with torch.no_grad():
            target_norm = self.model.normalizer.encode(x_target)
        x_hat_norm = self.model.normalizer.encode(x_hat)
        lin_loss = self.model.lin_loss
        rec_loss = self.train_loss(x_hat_norm, target_norm)
        total_loss = lin_loss +  rec_loss

        # Logging
        self.log_dict({
            'Train/recon_loss': rec_loss,
            'Train/linear_loss': lin_loss,
            'Train/total_loss': total_loss
        }, on_step=True, prog_bar=True)

        return total_loss

    def on_validation_epoch_start(self):
        self.model.update_steps(self.args.pred_len)

    def validation_step(self, batch, batch_idx):
        x_target = batch[:, :self.args.pred_len]
        x_hat, _, _ = self(x_target)
        with torch.no_grad():
            x_hat_norm = self.model.normalizer.encode(x_hat)
            target_norm = self.model.normalizer.encode(x_target)

        rec_loss = nn.functional.mse_loss(x_hat_norm, target_norm)
        vpt = compute_vpt(x_hat, x_target, 0.3*self.meta.std, dt=self.lyap_unit_per_step)

        #### log metric
        self.log('Val/val_recon_loss', rec_loss, prog_bar=True)
        self.log('hp_metric', vpt, prog_bar=True)
        return vpt


    def on_test_start(self):
    # Initialize storage for test step outputs at the start of each test epoch
        super().on_test_epoch_start()
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):
        if hasattr(self.model, 'update_steps'):
            self.model.update_steps(batch.shape[1])
        # Forecast from the first step
        x_hat, _, _ = self(batch[:,:1,:])  
        # Forecast evaluation
        x_pred = x_hat[:, :self.args.pred_len]
        target = batch[:, :self.args.pred_len] 
        
        vpt1 = compute_vpt(x_pred, target, 0.1 * self.meta.std, 
                          dt=self.lyap_unit_per_step, mode=None)
        vpt2 = compute_vpt(x_pred, target, 0.2 * self.meta.std, 
                          dt=self.lyap_unit_per_step, mode=None)
        result = {
            "rmse_loss": RMSELoss()(x_pred, target) ,
            'vpt1_dist': vpt1,
            'vpt2_dist': vpt2
        }
        self.test_step_outputs.append(result)
        return result

    def on_test_end(self, outputs=None):
        if not self.test_step_outputs:
            return
        avg_rmse = torch.stack([x["rmse_loss"] for x in self.test_step_outputs]).mean(dim=0)
        std_rmse = torch.stack([x["rmse_loss"] for x in self.test_step_outputs]).std(dim=0)
        vpt1_distribution = torch.cat([x["vpt1_dist"] for x in self.test_step_outputs])
        vpt2_distribution = torch.cat([x["vpt2_dist"] for x in self.test_step_outputs])
        for t, val in enumerate(avg_rmse):
            self.logger.experiment.add_scalar("test/rmse_vs_t", val, global_step= t*self.lyap_unit_per_step)

        results = {
            "summary": {
                "mean_vpt1": vpt1_distribution.mean().item(),
                "median_vpt1": vpt1_distribution.median().item(),
                "std_vpt1": vpt1_distribution.std().item(),
                "mean_vpt2": vpt2_distribution.mean().item(),
                "median_vpt2": vpt2_distribution.median().item(),
                "std_vpt2": vpt2_distribution.std().item(),
            },
            "curves": {
                "rmse_mean": avg_rmse.cpu().numpy().tolist(),
                "rmse_std": std_rmse.cpu().numpy().tolist(),
            },
            "distributions": {
                "vpt1_all_samples": vpt1_distribution.cpu().numpy().tolist(),
                "vpt2_all_samples": vpt2_distribution.cpu().numpy().tolist()
            }
        }

        # Set save directory for test results
        save_dir = os.path.join(self.logger.log_dir if self.logger else "./", "test_results")
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, f"test_result.json")

        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
        # Clear temporary test states to free memory
        self._cleanup_test_states()

    def _cleanup_test_states(self):
        if hasattr(self.model, 'Cocycle'):
            self.model.Cocycle.gate_prob_recording = False
            if hasattr(self.model.Cocycle, 'gate_probs_all_steps'):
                self.model.Cocycle.gate_probs_all_steps.clear()

    def configure_optimizers(self, epoch=None):
        try:
            # Group parameters
            groups = [
                {"params": list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()), 
                 "lr": self.args.lr_main, "weight_decay": self.args.decay_weight},
                {"params": list(self.model.Cocycle.opt.generators.parameters()), 
                 "lr": self.args.lr_expert, "weight_decay": 0.0},
                {"params": list(self.model.Cocycle.opt.gate.parameters()), 
                 "lr": self.args.lr_gate, "weight_decay": 0.0}
            ]

            optimizer = torch.optim.AdamW(groups)

            total_steps = self.steps_per_epoch * self.trainer.max_epochs
            scheduler = DualOneCycleScheduler(
                    optimizer,
                    max_lrs=[self.args.lr_main, self.args.lr_expert, self.args.lr_gate],
                    total_steps=total_steps,
                    div_factors=[25., 25., 100.],
                    final_div_factors=[self.args.div_main, self.args.div_expert, self.args.div_gate],
                    pct_start=self.args.warm_pct
                    )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
            }

        except AttributeError:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr_main,
                weight_decay=self.args.decay_weight,
            )
            total_steps = self.steps_per_epoch * self.args.train_epochs

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.args.lr_main,
                total_steps=total_steps,
                pct_start=self.args.warm_pct,
                anneal_strategy='cos',
                cycle_momentum=False
            )
        
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
            }

# Define a trainer
def trainer(args, devices=1):
    # Set global seed for reproducibility
    try:
        pl.seed_everything(int(getattr(args, 'seed', 2025)), workers=True)
    except Exception:
        pl.seed_everything(int(getattr(args, 'seed', 2025)))
    # Create data loaders
    datamodule = DataModule(
                root_path=args.root_path, data_path=args.data_path,
                batch_size=args.batch_size, seq_len=args.seq_len,
                pred_len=args.pred_len, downsample=args.downsample
                )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    steps_per_epoch = len(train_loader)
    ## callbacks
    early_stop = EarlyStopping(
        monitor='hp_metric',    
        patience=args.patience,  
        mode='max',            
        min_delta=1e-6,    
        verbose=False
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='hp_metric',
        dirpath=f'./checkpoints',
        filename=f'{args.setting}',
        mode='max'
    )
    # Initialize trainer    
    trainer = pl.Trainer(
        max_epochs=args.train_epochs, devices=devices,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='step'),
            early_stop],
        check_val_every_n_epoch=10,
        deterministic=True,
        logger=TensorBoardLogger("./logs", name=args.setting, default_hp_metric=False),
        gradient_clip_val=args.clip_grad,
        )
    
    ckpt_path = f'./checkpoints/{args.setting}.ckpt'
    if os.path.exists(ckpt_path):
        # Fix legacy keys in state_dict if necessary
        best_model = NeuralModel.load_from_checkpoint(ckpt_path,args=args,steps_per_epoch=1)
        trainer.test(best_model, datamodule=datamodule)
    else:
        # Initialize model
        model = NeuralModel(args, steps_per_epoch=steps_per_epoch)
        trainer.fit(model, datamodule)
        
        logger.info('------ Testing phase ------------')
        best_model = NeuralModel.load_from_checkpoint(checkpoint_callback.best_model_path, args=args, steps_per_epoch=steps_per_epoch)
        trainer.test(best_model, datamodule=datamodule)