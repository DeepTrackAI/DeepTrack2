import io
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

        [ax.cla() for ax in self.axes]

        for plotter, axis in zip(self.plotters, self.axes):
            plotter.plot(axis, self.batch_data)

        self.display_figure()

        self.batch_data = {}

        return super().on_epoch_end(epoch, logs)

    def display_figure(self):
        # Shows the figure. If inside a notebook, the figure is displayed using display.display(fig).
        # If outside a notebook, the figure is displayed using plt.show().

        try:
            import IPython.display

            # Make figure background white
            fig = plt.gcf()
            fig.set_facecolor("white")

            # Convert figure to JPEG data URL
            buf = io.BytesIO()
            plt.savefig(buf, format="jpeg")
            buf.seek(0)
            jpeg_data = buf.getvalue()

            # clear display
            IPython.display.clear_output(wait=True)
            # display PNG data URL
            IPython.display.display(
                IPython.display.Image(data=jpeg_data, format="jpeg")
            )
        except ImportError:
            # Not in notebook
            plt.show()

        return

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