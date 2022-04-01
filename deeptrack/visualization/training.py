from time import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from deeptrack.visualization import colors


class TrainingPlotter:
    """Base class for plotting dara during training."""

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

    def plot(self, ax: plt.Axes = None, data=None):
        """Plot the training loss.
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on.
        data : dict
            The data to plot.
        """

        if ax is None:
            ax = plt.gca()

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

    def plot(self, ax: plt.Axes = None, data=None):
        """Plot the training speed.
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on.
        data : dict
            The data to plot.
        """

        if ax is None:
            ax = plt.gca()

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


class LossPropertyPlotter(TrainingPlotter):
    """Plots the model loss metrics as a function of the value of some property of the data.

    Parameters
    ----------
    images : list of Image
       List of images to evaluate on
    target : function or numpy array or list of numpy arrays
        Function to calculate the target, or a numpy array of the same shape as the images
    property : str
       Name of the property to compare with.
    model : Keras model
         Model to evaluate the images on.
    metrics : list of str or function or tensorflow.keras.metrics.Metric or None
        List of metrics to plot. Can also be `loss` or the name of a custom metric. If None, use the model's metrics_names.
    number_of_bins : int
        The number of bins to use for the smoothed plot. Default is 20
    """

    def __init__(
        self, images, target, model, property, metrics=None, number_of_bins=20
    ):
        super().__init__()
        self.model = model
        self.property = property
        self.metrics = metrics
        self.number_of_bins = number_of_bins

        # Get the property of the images
        self.properties = [img.get_property(self.property) for img in images]

        # Sort the images, properties and targets by the property and make them numpy arrays
        self.properties, self.images, self.target = zip(
            *sorted(zip(self.properties, images, target), key=lambda x: x[0])
        )
        # Make the properties numpy arrays
        self.properties = np.array(self.properties)
        # Make the images numpy arrays
        self.images = np.array(self.images)
        # Make the targets numpy arrays
        self.target = np.array(self.target)

        self.epoch_data = []

    def plot(self, ax: plt.Axes = None, data=None):
        """Create the plot.
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on. Default is plt.gca()
        data : dict
            The data to plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes that was plotted on.

        """

        if ax is None:
            ax = plt.gca()

        # Get metric functions and names of the model that matches `metrics`.
        metrics = self.metrics

        if metrics is None:
            metrics = self.model.metrics_names

        for i, metric in enumerate(metrics):
            if metric == "loss":
                metrics[i] = self.model.loss

        metric_functions = [tf.metrics.get(metric) for metric in metrics]

        # Get the metric names. If the metric is a function, use the name of the function.
        # If the metric is a metric, use the name of the metric.
        # If the metric is a string, use the string.
        metric_names = [
            metric.name
            if hasattr(metric, "name")
            else metric.__name__
            if hasattr(metric, "__name__")
            else metric
            for metric in metrics
        ]

        preds = self.model.predict(x=self.images, verbose=0)

        # Calculate each metric in model for each image in self.images
        # Call metric.reset_states between each metric calculation
        metrics = {}
        for (metric_function, metric_names) in zip(metric_functions, metric_names):

            values = []
            for P, T in zip(preds, self.target):
                values.append(metric_function(P, T))

            metrics[metric_names] = np.array(values)

        # Scatterplot each metric in loss
        for i, (metric_name, metric_data) in enumerate(metrics.items()):

            color = colors[i % len(colors)]

            # Scatter with facecolor and edgecolor as color
            ax.scatter(
                self.properties,
                metric_data,
                color=color,
                edgecolor=color,
                label=metric_name,
                alpha=0.5,
            )

            # Bin the loss by the property
            bins = np.linspace(
                np.min(self.properties), np.max(self.properties), self.number_of_bins
            )

            # round properties to nearest bin
            binned_properties = np.digitize(self.properties, bins)

            # Caluclate the mean loss within each bin
            binned_loss = np.array(
                [
                    np.mean(metric_data[binned_properties == i])
                    for i in range(1, len(bins))
                ]
            )

            # Plot the binned loss
            ax.plot(bins[1:], binned_loss, color=color, label=metric_name)

        ax.set_xlabel(self.property)
        ax.set_ylabel("Loss")
        ax.legend()

        return ax