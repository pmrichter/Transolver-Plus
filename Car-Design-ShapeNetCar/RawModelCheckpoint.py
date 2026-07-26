import os
import torch

from lightning.pytorch.callbacks import Callback


class RawModelCheckpoint(Callback):

    def __init__(self, path, seed):
        self.path = path
        self.seed = seed

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        if trainer.current_epoch > 120:
            torch.save(pl_module.model,self.path + os.sep + f'model_{trainer.current_epoch}_hyperspherical_400_l1_{self.seed}.pth')
