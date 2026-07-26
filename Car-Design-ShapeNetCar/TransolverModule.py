import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as pl


class TransolverModule(pl.LightningModule):
    def __init__(self, model, lr, reg=1):
        super().__init__()
        self.model = model
        self.lr = lr
        self.reg = reg
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, batch):
        cfd_data, geom = batch
        return self.model((cfd_data, geom))

    def prepare_step(self, batch):
        cfd_data, geom = batch
        out = self.model((cfd_data, geom)).float()
        targets = cfd_data.y

        loss_press = self.criterion(out[cfd_data.surf, -1], targets[cfd_data.surf, -1]).mean(dim=0)
        return cfd_data, out, targets, loss_press

    def training_step(self, batch, batch_idx):
        cfd_data, out, targets, loss_press = self.prepare_step(batch)
        loss_velo = F.smooth_l1_loss(out[:, :-1], targets[:, :-1], reduction='none', beta=0.01).mean()
        total_loss = loss_velo + self.reg * loss_press

        batch_size = getattr(cfd_data, 'num_graphs', 1)
        self.log_dict({'train_loss': total_loss, 'train_loss_velo': loss_velo, 'train_loss_press': loss_press},
                      on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return total_loss

    def validation_step(self, batch, batch_idx):
        cfd_data, out, targets, loss_press = self.prepare_step(batch)
        loss_velo = self.criterion(out[:, :-1], targets[:, :-1]).mean()
        val_loss = loss_velo + self.reg * loss_press

        batch_size = getattr(cfd_data, 'num_graphs', 1)
        self.log_dict({'val_loss': val_loss, 'val_loss_velo': loss_velo, 'val_loss_press': loss_press},
                      on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

    def configure_optimizers(self):
        # ==========================================================
        # 1. PARAMETER GROUPING (The Anti-Overfitting Fix)
        # ==========================================================
        velo_params = []
        press_params = []

        for name, param in self.model.named_parameters():
            if 'velo' in name:
                velo_params.append(param)
            else:
                press_params.append(param)

        # ==========================================================
        # 2. ADAM-W OPTIMIZER WITH TARGETED DECAY
        # ==========================================================
        # Velocity gets a gentle pull to stop memorizing the wake.
        # Pressure stays at 0.0 to learn the sharp boundary conditions perfectly.
        adam_eps = 1e-8
        optimizer = torch.optim.AdamW([
            {'params': velo_params, 'weight_decay': 0.0},
            {'params': press_params, 'weight_decay': 0.0}
        ], lr=self.lr, eps=adam_eps)

        # ==========================================================
        # 3. LEARNING RATE SCHEDULER
        # ==========================================================
        # estimated_stepping_batches already accounts for the number of
        # devices, so the schedule is correct for any world size.
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            final_div_factor=1000.,
        )
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'}}
