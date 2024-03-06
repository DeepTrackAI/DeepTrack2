import deeplay as dl
from ..features import Feature
from ..pytorch import Dataset
from lightning.pytorch.callbacks import RichProgressBar

import numpy as np
import torch
import lightning as L
from typing import Union, Optional, Literal, Tuple, Callable, Sequence, Dict
import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import torchmetrics as tm

Optimizer = dl.Optimizer

import logging
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

class LogHistory(L.Callback):
    """A keras-like history callback for lightning. Keeps track of metrics and losses during training and validation.
    
    Example:
    >>> history = LogHistory()
    >>> trainer = dl.Trainer(callbacks=[history])
    >>> trainer.fit(model, train_dataloader, val_dataloader)
    >>> history.history {"train_loss_epoch": {"value": [0.1, 0.2, 0.3], "epoch": [0, 1, 2], "step": [0, 100, 200]}}
    """

    @property
    def history(self):
        return {key: 
                {
                    "value": [item["value"] for item in value],
                    "epoch": [item["epoch"] for item in value],
                    "step": [item["step"] for item in value]
                }
                for key, value in self._history.items()}

    @property
    def step_history(self):
        return {key: 
                {
                    "value": [item["value"] for item in value],
                    "epoch": [item["epoch"] for item in value],
                    "step": [item["step"] for item in value]
                }
                for key, value in self._step_history.items()}
    
    def __init__(self):
        self._history = {}
        self._step_history = {}

    def on_train_batch_end(self, trainer: dl.Trainer, *args, **kwargs) -> None:
        for key, value in trainer.callback_metrics.items():
            if key.endswith("_step"):
                self._step_history.setdefault(key, []).append(self._logitem(trainer, value))

    def on_train_epoch_end(self, trainer: dl.Trainer, *args, **kwargs) -> None:
        for key, value in trainer.callback_metrics.items():
            if key.startswith("train") and key.endswith("_epoch"):
                self._history.setdefault(key, []).append(self._logitem(trainer, value))
    
    def on_validation_epoch_end(self, trainer: dl.Trainer, *args, **kwargs) -> None:
        for key, value in trainer.callback_metrics.items():
            if key.startswith("val") and key.endswith("_epoch"):
                self._history.setdefault(key, []).append(self._logitem(trainer, value))

    def _logitem(self, trainer, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        return {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "value": value
        }

    def plot(self, *args, yscale="log", **kwargs):
        """Plot the history of the metrics and losses.
        """

        history = self.history
        step_history = self.step_history
        
        keys = list(history.keys())
        keys = [key.replace("val", "").replace("train", "") for key in keys]
        unique_keys = list(set(keys))
        # sort unique keys same as keys
        unique_keys.sort(key=lambda x: keys.index(x))
        keys = unique_keys

        max_width = 3
        rows = len(keys) // max_width + 1
        width = min(len(keys), max_width)

        fig, axes = plt.subplots(rows, width, figsize=(15, 5 * rows))

        if len(keys) == 1:
            axes = np.array([[axes]])

        for ax, key in zip(axes.ravel(), keys):
            train_key = "train" + key
            val_key = "val" + key
            step_key = ("train" + key).replace("epoch", "step")

            
            if step_key in step_history:
                ax.plot(step_history[step_key]["step"], step_history[step_key]["value"], label=step_key, color="C1", alpha=0.25)
            if train_key in history:
                step = np.array(history[train_key]["step"])
                step[1:] = step[1:] - (step[1:] - step[:-1]) / 2
                step[0] /= 2
                marker_kwargs = dict(marker="o", markerfacecolor="white", markeredgewidth=1.5) if len(step) < 20 else {}
                ax.plot(step, history[train_key]["value"], label=train_key, color="C1", **marker_kwargs)

            if val_key in history:
                marker_kwargs = dict(marker="d", markerfacecolor="white", markeredgewidth=1.5) if len(step) < 20 else {}
                ax.plot(history[val_key]["step"], history[val_key]["value"], label=val_key, color="C3", linestyle="--", **marker_kwargs)
            
            ax.set_title(key.replace("_", " ").replace("epoch", "").strip().capitalize())
            ax.set_xlabel("Step")

            ax.legend()
            ax.set_yscale(yscale)
        
        return fig, axes
    
class Model(dl.Application):
    """A wrapper for a pytorch model for easy integration with deeptrack pipelines.
    
    A lightning model with convenience methods for training, validation and testing.
    Intended for use with deeptrack pipelines. For more advanced use cases, consider 
    using deeplay or lightning directly. 
    """

    def __init__(self, 
                 model: torch.nn.Module, 
                 train_data: Union[Feature, torch.utils.data.Dataset, Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...], torch.Tensor, np.ndarray, None] = None,
                 val_data: Union[Feature, torch.utils.data.Dataset, Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...], torch.Tensor, np.ndarray, None] = None,
                 test_data: Union[Feature, torch.utils.data.Dataset, Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...], torch.Tensor, np.ndarray, None] = None,
                 data_channel_position: Literal["first", "last"] = "last",
                 loss: Union[torch.nn.Module, Callable[..., torch.Tensor], None] = None,
                 optimizer: Optional[dl.Optimizer] = None,
                 metrics: Optional[Sequence[tm.Metric]] = None,
                 train_metrics: Optional[Sequence[tm.Metric]] = None,
                 val_metrics: Optional[Sequence[tm.Metric]] = None,
                 test_metrics: Optional[Sequence[tm.Metric]] = None,
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
        self.data_channel_position = data_channel_position

    def forward(self, x):
        return self.model(self._maybe_to_channel_first(x))
    
    def _maybe_to_channel_first(self, x, other=None):
        if (self.data_channel_position == "last" and 
            x.ndim > 2 and
            x.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64]):
            return x.permute(0, -1, *range(1, x.ndim-1))
        return x
    
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
            val_steps_per_epoch=10,
            callbacks=[], 
            permute_target_channels: Union[bool, Literal["auto"]] = "auto",
            **kwargs) -> LogHistory:
        """Train the model on the training data.

        Train the model on the training data, with optional validation data.

        Parameters
        ----------
        max_epochs : int
            The maximum number of epochs to train the model.
        batch_size : int
            The batch size to use for training.
        steps_per_epoch : int
            The number of steps per epoch (used if train_data is a Feature).
        replace : bool or float
            Whether to replace the data after each epoch (used if train_data is a Feature).
            If a float, the data is replaced with the given probability.
        val_batch_size : int
            The batch size to use for validation. If None, the training batch size is used.
        val_steps_per_epoch : int
            The number of steps per epoch for validation.
        callbacks : list
            A list of callbacks to use during training.
        permute_target_channels : bool or "auto"
            Whether to permute the target channels to channel first. If "auto", the model will
            attempt to infer the correct permutation based on the input and target shapes.
        **kwargs
            Additional keyword arguments to pass to the trainer.
        """        
        
        self.permute_target_channels = permute_target_channels

        val_batch_size = val_batch_size or batch_size
        val_steps_per_epoch = val_steps_per_epoch or 10
        train_data = self.create_data(self.train_data, batch_size, steps_per_epoch, replace)
        val_data = self.create_data(self.val_data, val_batch_size, val_steps_per_epoch, False) if self.val_data else None
        
        history = LogHistory()
        progressbar = RichProgressBar()
        
        callbacks = callbacks + [history, progressbar]
        trainer = dl.Trainer(max_epochs=max_epochs, callbacks=callbacks, **kwargs)

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        if not self._has_built:
            self.build()

        if val_data:
            val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=False) if val_data else None
            trainer.fit(self, train_dataloader, val_dataloader)
        else:
            trainer.fit(self, train_dataloader)

        return history
    
    def test(self, 
             data: Union[Feature, torch.utils.data.Dataset, Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...], torch.Tensor, np.ndarray], 
             metrics: Union[tm.Metric, Tuple[str, tm.Metric], Sequence[Union[tm.Metric, Tuple[str, tm.Metric]]], Dict[str, tm.Metric]],
             batch_size: int = 32):
        """Test the model on the given data.

        Test the model on the given data, using the given metrics. Metrics can be
        given as a single metric, a tuple of name and metric, a sequence of metrics
        (or tuples of name and metric) or a dictionary of metrics. In the case of
        tuples, the name is used as the key in the returned dictionary. In the case
        of metrics, the name of the metric is used as the key in the returned dictionary.

        Parameters
        ----------
        data : data-like
            The data to test the model on. Can be a Feature, a torch.utils.data.Dataset, a tuple of tensors, a tensor or a numpy array.
        metrics : metric-like
            The metrics to use for testing. Can be a single metric, a tuple of name and metric, a sequence of metrics (or tuples of name and metric) or a dictionary of metrics.
        batch_size : int
            The batch size to use for testing.
        """

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

    def train_preprocess(self, batch):
        x, y = super().train_preprocess(batch)
        if self.permute_target_channels == "auto":
            y = self._maybe_to_channel_first(y)
        elif self.permute_target_channels is True:
            y = y.permute(0, -1, *range(1, x.ndim-1))
        return x, y
    
    val_preprocess = train_preprocess
    test_preprocess = train_preprocess