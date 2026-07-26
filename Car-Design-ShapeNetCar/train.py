import numpy as np
import time, json, os
import torch
import torch.nn as nn

import lightning as pl
from lightning.pytorch.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from TransolverModule import TransolverModule

from RawModelCheckpoint import RawModelCheckpoint

# Allow high precision calculation on Ampere architecture. Does not affect MPS on Mac.
torch.set_float32_matmul_precision('high')

seed = 1


def get_nb_trainable_params(model):
    '''
    Return the number of trainable parameters
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(train_dataset, val_dataset, tmodel, hparams, path, reg=1, val_iter=1, coef_norm=[]):
    pl.seed_everything(seed, workers=True)

    use_ampere = tmodel.attn_type == 'dot_product_flash' and torch.cuda.is_available()
    if use_ampere:
        print("Using bf16 autocast for flash attention")
    else:
        print("Using full precision")

    lit_model = TransolverModule(tmodel, lr=hparams['lr'], reg=reg)

    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    trainer = pl.Trainer(
        max_epochs=hparams['nb_epochs'],
        accelerator='auto',
        devices='auto',
        strategy='auto',
        precision='bf16-mixed' if use_ampere else '32-true',
        check_val_every_n_epoch=val_iter if val_iter is not None else 1,
        limit_val_batches=0.0 if val_iter is None else 1.0,
        num_sanity_val_steps=0,
        logger=CSVLogger(save_dir=path),
        callbacks=[RawModelCheckpoint(path, seed)],
        enable_checkpointing=False,
    )

    start = time.time()
    trainer.fit(lit_model, train_loader, val_loader)
    time_elapsed = time.time() - start

    if trainer.is_global_zero:
        params_model = get_nb_trainable_params(tmodel).astype('float')
        print('Number of parameters:', params_model)
        print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
        torch.save(tmodel, path + os.sep + f'model_{hparams["nb_epochs"]}_hyperspherical_400_l1_{seed}.pth')

        if val_iter is not None:
            train_loss = trainer.callback_metrics.get('train_loss')
            val_loss = trainer.callback_metrics.get('val_loss')
            with open(path + os.sep + f'log_{hparams["nb_epochs"]}_hyperspherical_400_l1_{seed}.json', 'a') as f:
                json.dump(
                    {
                        'nb_parameters': params_model,
                        'time_elapsed': time_elapsed,
                        'hparams': hparams,
                        'train_loss': train_loss.item() if train_loss is not None else None,
                        'val_loss': val_loss.item() if val_loss is not None else None,
                        'coef_norm': list(coef_norm),
                    }, f, indent=12, cls=NumpyEncoder
                )

    return tmodel
