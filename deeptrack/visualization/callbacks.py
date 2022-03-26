from operator import sub
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt


class VisualizationCallback(Callback):
    def __init__(self, plotters=[]):

        self.plotters = plotters
        self.batch_data = {}

        super().__init__()

    def on_train_begin(self, logs=None):
        self.axes = self.prepare_figure()
        return super().on_train_begin(logs)

    def on_batch_end(self, batch, logs=None):

        for key, val in logs.items():
            if key not in self.batch_data:
                self.batch_data[key] = []

            self.batch_data[key].append(val)

        return super().on_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):

        axes = self.prepare_figure()

        [ax.cla() for ax in axes]

        for plotter, axis in zip(self.plotters, axes):
            plotter.plot(axis, self.batch_data)

        self.clear_display()
        plt.show()

        self.batch_data = {}

        return super().on_epoch_end(epoch, logs)

    def prepare_figure(self):

        plt.ion()

        height = (
            sum([plotter.height for plotter in self.plotters])
            + len(self.plotters) * 0.5
            + 0.5
        )

        width = max([plotter.width for plotter in self.plotters]) + 1

        fig = plt.figure(figsize=(width, height))

        axes = []

        y = height - 0.5
        for plotter in self.plotters:

            sub_height = plotter.height
            sub_width = plotter.width

            ax = fig.add_axes(
                (
                    0.5 / width,
                    (y - sub_height) / height,
                    sub_width / width,
                    sub_height / height,
                )
            )
            axes.append(ax)

            y -= sub_height + 0.5

        return axes

    def clear_display(self):

        try:
            import IPython.display

            IPython.display.clear_output(True)
        except Exception:
            # Not in notebook
            pass
