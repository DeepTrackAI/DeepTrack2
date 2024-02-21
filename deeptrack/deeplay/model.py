import deeplay as dl
from ..features import Feature
from ..pytorch import Dataset

import numpy as np
import torch
import lightning as L
from typing import Union, Optional, Literal
import tqdm

class LogHistory(L.Callback):
    def __init__(self):
        self.history = {}
        self.step_history = {}

    def on_train_epoch_end(self, trainer: dl.Trainer, pl_module: L.LightningModule) -> None:
        for key, value in trainer.callback_metrics.items():
            self.history.setdefault(key, []).append(value.item())
    
    def on_validation_epoch_end(self, trainer: dl.Trainer, pl_module: L.LightningModule) -> None:
        for key, value in trainer.callback_metrics.items():
            self.history.setdefault(key, []).append(value.item())


class Model(dl.Application):

    def __init__(self, 
                 model, 
                 train_data=None,
                 val_data=None,
                 test_data=None,
                 loss=None,
                 metrics=None,
                 train_metrics=None,
                 val_metrics=None,
                 test_metrics=None,
                 optimizer=None,
                 **kwargs):
        
        super().__init__(
                         loss=loss,
                         metrics=metrics,
                         train_metrics=train_metrics,
                         val_metrics=val_metrics,
                         test_metrics=test_metrics,
                         optimizer=optimizer,
                         **kwargs)
        
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def forward(self, x):
        return self.model(x)
    
    
    def create_data(self, data, batch_size=32, steps_per_epoch=100, replace=False, **kwargs):
        if isinstance(data, Feature):
            return Dataset(data, length=batch_size*steps_per_epoch, replace=replace)
        elif isinstance(data, (Dataset, torch.utils.data.Dataset)):
            return data
        elif isinstance(data, (tuple, list)):
            datas = [self.create_data(d) for d in data]
            return torch.utils.data.TensorDataset(*datas)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            return data
        else:
            raise ValueError(f"Data type {type(data)} not supported")

    def fit(self, 
            max_epochs=None, 
            batch_size=32, 
            steps_per_epoch=100,
            replace=False, 
            val_batch_size=None, 
            callbacks=[], 
            **kwargs) -> LogHistory:
        val_batch_size = val_batch_size or batch_size
        train_data = self.create_data(self.train_data, batch_size, steps_per_epoch, replace)
        val_data = self.create_data(self.val_data, val_batch_size, steps_per_epoch, replace) if self.val_data else None
        
        history = LogHistory()
        callbacks = callbacks + [history]
        trainer = dl.Trainer(max_epochs=max_epochs, callbacks=callbacks, **kwargs)

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        if not self._has_built:
            self.build()

        if val_data:
            val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=True) if val_data else None
            trainer.fit(self, train_dataloader, val_dataloader)
        else:
            trainer.fit(self, train_dataloader)

        return history
    
    def test(self, data, metrics, batch_size=32, **kwargs):
        device = self.trainer.strategy.root_device
        self.to(device)
        test_data = self.create_data(data)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

        for (x, y) in tqdm.tqdm(test_dataloader):
            y_hat = self(x.to(device))
            for metric in metrics.values():
                metric.to(device)
                metric.update(y_hat.to(device), y.to(device))
        
        return {name: metrics[name].compute() for name in metrics}

    def compute_loss(self, y_hat, y) -> torch.Tensor | dl.Dict[str, torch.Tensor]:
        if isinstance(self.loss, (torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss)):
            y = y.float()
        return super().compute_loss(y_hat, y)
