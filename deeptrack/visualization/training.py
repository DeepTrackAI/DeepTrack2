import matplotlib.pyplot as plt
import numpy as np

from deeptrack.visualization import colors


class TrainingPlotter:
    def __init__(self, *args, **kwargs):
        self.height = 5
        self.width = 9


class TrainingLossPlotter(TrainingPlotter):
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

        handles = []
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

            ax.fill_between(x, datavec[0], datavec[2], color=color, alpha=0.4)

            ax.plot(x, datavec[0], color=color, label=loss_name)

            handles.append(ax.plot(x, datavec[1], color=color))

            ax.plot(x, datavec[2], color=color)

        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend()
