from time import time
import matplotlib.pyplot as plt
import numpy as np

from deeptrack.visualization import colors


class TrainingPlotter:
    def __init__(self, *args, **kwargs):
        self.height = 4
        self.width = 10


class TrainingLossPlotter(TrainingPlotter):
    """Plots the training loss during training.

    Parameters
    ----------
    loss_names : list of str
        The names of the losses to plot.
    quantiles : list of float
        The lower and upper quantiles to fill the area between.
    xscale, yscale : str
        The scale of the x and y axes. One of "linear", "log", "symlog", "logit",
    """

    def __init__(
        self, loss_names=["loss"], quantiles=[0, 1], xscale="log", yscale="log"
    ):
        super().__init__()
        self.loss_names = loss_names
        self.xscale = xscale
        self.yscale = yscale
        self.quantiles = quantiles

        self.epoch_data = dict(**{k: [[], [], []] for k in loss_names})

    def plot(self, ax: plt.Axes, data):
        """Plot the training loss.
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on.
        data : dict
            The data to plot.
        """
        for i, loss_name in enumerate(self.loss_names):

            color = colors[i % len(colors)]

            datavec = self.epoch_data[loss_name]
            batchdata = data[loss_name]

            vmin, vmax = np.quantile(batchdata, self.quantiles)
            vmean = np.mean(batchdata)

            datavec[0].append(vmin)
            datavec[1].append(vmean)
            datavec[2].append(vmax)

            x = np.arange(len(datavec[0])) + 1

            ax.fill_between(x, datavec[0], datavec[2], color=color, ec=color, alpha=0.4)
            ax.plot(x, datavec[1], color=color, label=loss_name)

        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend()


class TrainingSpeedPlotter(TrainingPlotter):
    """Plots the time it takes per batch on average."""

    def __init__(self):
        super().__init__()
        self.height = 4
        self.time_data = []

    def plot(self, ax: plt.Axes, data):
        """Plot the training speed.
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on.
        data : dict
            The data to plot.
        """

        # Add the current time to time_data
        self.time_data.append(time())

        # Calculate the difference between each timepoint in seconds

        dataitem = list(data.values())[0]
        time_diffs = np.diff(self.time_data) / len(dataitem)

        x = np.arange(len(time_diffs)) + 1

        # Plot the time difference
        ax.plot(x, time_diffs * 1e3, color=colors[0])

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Time per batch (ms)")
