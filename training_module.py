'''
PyTorch Lightning module for training AlphaNet
'''
from typing import Dict, List, Optional, Tuple
import pdb
from pathlib import Path
import torch
from torch import nn

from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, CosineAnnealingLR
from pytorch_lightning import LightningModule
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, CosineSimilarity
import torch_scatter
from torch_scatter import scatter_mean

from utils import average_over_batch_metrics, pretty_print
import utils as diff_utils

import yaml
from torch_geometric.data import Batch
import random
random.seed(42)

LR_SCHEDULER = {
    "cos": CosineAnnealingWarmRestarts,
    "step": StepLR,
}
GLOBAL_ATOM_NUMBERS = torch.tensor([1, 6, 7, 8])


def custom_collate(data_list):
    for i, data in enumerate(data_list):
        if not hasattr(data, "pbc"):
            data.pbc = torch.zeros((3,), dtype=torch.bool)
        if not hasattr(data, "spin"):
            data.spin = torch.ones(1, dtype=torch.long)
        if not hasattr(data, "charge"):
            data.charge = torch.zeros(1, dtype=torch.long)
        print(f"collate[{i}] pbc: {getattr(data, 'pbc', None)}")
    batch = Batch.from_data_list(data_list)
    print(f"Batch keys: {batch.keys}")
    return batch
def compute_extra_props_aimnet(batch, pos_require_grad=True):
    # device = batch.energy.device
    # indices = batch.one_hot.long().argmax(dim=1)
    # batch.z = GLOBAL_ATOM_NUMBERS.to(device)[indices.to(device)]
    batch.pos = remove_mean_batch(batch.pos, batch.batch)
    # batch.atomic_numbers = batch.charges
    
    # num_molecules = int(batch.batch.max().item()) + 1
    # batch.spin = torch.ones(num_molecules, dtype=torch.long, device=device)
    # batch.charge = torch.zeros(num_molecules, dtype=torch.long, device=device)
    # batch.pbc = torch.zeros((num_molecules, 3), dtype=torch.bool, device=device)

    if pos_require_grad:
        batch.pos.requires_grad_(True)
    return batch

def compute_extra_props(batch, pos_require_grad=True):
    device = batch.energy.device
    indices = batch.one_hot.long().argmax(dim=1)
    batch.z = GLOBAL_ATOM_NUMBERS.to(device)[indices.to(device)]
    batch.pos = remove_mean_batch(batch.pos, batch.batch)
    batch.atomic_numbers = batch.charges
    
    num_molecules = int(batch.batch.max().item()) + 1
    batch.spin = torch.ones(num_molecules, dtype=torch.long, device=device)
    batch.charge = torch.zeros(num_molecules, dtype=torch.long, device=device)
    batch.pbc = torch.zeros((num_molecules, 3), dtype=torch.bool, device=device)

    if pos_require_grad:
        batch.pos.requires_grad_(True)
    return batch

def compute_extra_props_new(batch, pos_require_grad=True):
    device = batch.energy.device

    batch.pos = remove_mean_batch(batch.pos, batch.batch)
    batch.atomic_numbers = batch.z
    
    num_molecules = int(batch.batch.max().item()) + 1
    batch.spin = torch.ones(num_molecules, dtype=torch.long, device=device)
    batch.charge = torch.zeros(num_molecules, dtype=torch.long, device=device)
    batch.pbc = torch.zeros((num_molecules, 3), dtype=torch.bool, device=device)
    batch.natoms = len(batch.z) 
    if pos_require_grad:
        batch.pos.requires_grad_(True)
    return batch


def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x





class PotentialModule(LightningModule):
    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        training_config: Dict,
    ) -> None:
        super().__init__()


        self.model_config = model_config


        from PaiNN import PainnModel
        self.potential = PainnModel()
          
        self.optimizer_config = optimizer_config
        self.training_config = training_config
        self.pos_require_grad = True

        self.clip_grad = training_config["clip_grad"]
        if self.clip_grad:
            self.gradnorm_queue = diff_utils.Queue()
            self.gradnorm_queue.add(3000)
        self.save_hyperparameters()

        self.loss_fn = nn.L1Loss()
        self.MAEEval = MeanAbsoluteError()
        self.MAPEEval = MeanAbsolutePercentageError()
        self.cosineEval = CosineSimilarity(reduction="mean")
        self.val_step_outputs = []
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.potential.parameters(),
            **self.optimizer_config
        )

        if not self.training_config["lr_schedule_type"] is None:
            scheduler_func = LR_SCHEDULER[self.training_config["lr_schedule_type"]]
            scheduler = scheduler_func(
                optimizer=optimizer,
                **self.training_config["lr_schedule_config"]
            )
            return [optimizer], [scheduler]
        return optimizer

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.dataset = torch.load(self.training_config['datadir'])
            random.shuffle(self.dataset)
            self.train_dataset = self.dataset[:int(len(self.dataset)*0.85)]
            self.val_dataset = self.dataset[int(len(self.dataset)*0.85):]
            print("# of training data: ", len(self.train_dataset))
            print("# of validation data: ", len(self.val_dataset))
 
    def sample_with_mask(self, n, num_samples, mask):
        if mask.shape[0] != n:
            raise ValueError("Mask length must be equal to the number of rows in the grid (n)")
        
        # Calculate total available columns after applying the mask
        # Only rows where mask is True are considered
        valid_rows = torch.where(mask)[0]  # Get indices of rows that are True
        if valid_rows.numel() == 0:
            raise ValueError("No valid rows available according to the mask")

        # Each valid row contributes 3 indices
        valid_indices = valid_rows.repeat_interleave(3) * 3 + torch.tensor([0, 1, 2]).repeat(valid_rows.size(0)).to(mask.device)

        # Sample unique indices from the valid indices
        chosen_indices = valid_indices[torch.randperm(valid_indices.size(0))[:num_samples]]

        # Convert flat indices back to row and column indices
        row_indices = chosen_indices // 3
        col_indices = chosen_indices % 3

        # Combine into 2-tuples
        samples = torch.stack((row_indices, col_indices), dim=1)
        
        return samples
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_config["bz"],
            shuffle=True,
            num_workers=self.training_config["num_workers"],
            collate_fn=custom_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.training_config["bz"],
            shuffle=False,
            num_workers=self.training_config["num_workers"],
            collate_fn=custom_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.training_config["bz"],
            shuffle=False,
            num_workers=self.training_config["num_workers"],
            collate_fn=custom_collate,
        )

    @torch.enable_grad()
    def compute_loss(self, batch):
        batch.pos.requires_grad_()


        hat_ae, hat_forces = self.potential.forward(
            batch.to(self.device),
        )

        hat_ae = hat_ae.squeeze().to(self.device)
        hat_forces = hat_forces.to(self.device)

        ae = batch.energy.to(self.device)
        forces = batch.force.to(self.device)

        if self.training_config['pretrain']:

            floss = self.loss_fn(forces, hat_forces)
            info = {
                "MAE_F": floss.detach().item(),

            }
            
            loss = floss * 100 
            return loss, info
        eloss = self.loss_fn(ae, hat_ae)
        floss = self.loss_fn(forces, hat_forces)
        info = {
            "MAE_E": eloss.detach().item(),
            "MAE_F": floss.detach().item(),

        }
        self.MAEEval.reset()
        self.MAPEEval.reset()
        self.cosineEval.reset()
        
        loss = floss * 100 + eloss * 4 
        return loss, info

    def training_step(self, batch, batch_idx):
        loss, info = self.compute_loss(batch)

        self.log("train-totloss", loss, rank_zero_only=True)

        for k, v in info.items():
            self.log(f"train-{k}", v, rank_zero_only=True)
        del info
        return loss

    def __shared_eval(self, batch, batch_idx, prefix, *args):
      with torch.enable_grad():
        loss, info = self.compute_loss(batch)
        info["totloss"] = loss.item()

        info_prefix = {}
        for k, v in info.items():
            key = f"{prefix}-{k}"
            if isinstance(v, torch.Tensor):
                v = v.detach()
                if v.is_cuda:
                    v = v.cpu()
                if v.numel() == 1:
                    info_prefix[key] = v.item()
                else:
                    info_prefix[key] = v.numpy()
            else:
                info_prefix[key] = v
            self.log(key, v, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        del info
      return info_prefix
  
    def _shared_eval(self, batch, batch_idx, prefix, *args):
        loss, info = self.compute_loss(batch)
        detached_loss = loss.detach()
        info["totloss"] = detached_loss.item()
       # info["totloss"] = loss.item()
        
        info_prefix = {}
        for k, v in info.items():
            info_prefix[f"{prefix}-{k}"] = v
        del info

        if torch.cuda.is_available():
           torch.cuda.empty_cache()
        return info_prefix

    def validation_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "val", *args)

    def test_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "test", *args)
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):

        self.val_step_outputs.append(outputs)

    def on_validation_epoch_end(self):
        
        val_epoch_metrics = average_over_batch_metrics(self.val_step_outputs)
        if self.trainer.is_global_zero:
            pretty_print(self.current_epoch, val_epoch_metrics, prefix="val")

        val_epoch_metrics.update({"epoch": self.current_epoch})
        for k, v in val_epoch_metrics.items():
            self.log(k, v, sync_dist=True)
    
        self.val_step_outputs.clear()
        

    def _configure_gradient_clipping(
        self,
        optimizer,
        # optimizer_idx,
        gradient_clip_val,
        gradient_clip_algorithm
    ):

        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 1.5 * stdev of the recent history.
        max_grad_norm = 2 * self.gradnorm_queue.mean() + \
            3 * self.gradnorm_queue.std()

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g['params']]
        grad_norm = diff_utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=max_grad_norm,
                            gradient_clip_algorithm='norm')

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')