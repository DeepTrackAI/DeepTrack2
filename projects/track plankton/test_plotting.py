import unittest
from plotting import *
from loader import *
from utils import *
from models import *
import sys


class TestPlotting(unittest.TestCase):
    def test_plot_image(self):
        particle = stationary_spherical_plankton()
        optics = plankton_brightfield()
        output_image = create_image(1, particle, optics, 0, 1)
        plot_image(output_image)

    def test_plot_label(self):
        particle = stationary_spherical_plankton()
        optics = plankton_brightfield()
        output_image = create_image(1, particle, optics, 0, 1)
        plot_label(get_target_image, output_image)

    def test_plot_image_stack(self):
        folder_path = sys.path[0] + "\\S arenicola and R baltica"
        im_stack = get_image_stack(
            outputs=[0],
            folder_path=folder_path,
            frame_im0=16,
            im_size_width=1024,
            im_size_height=1024,
            im_resize_width=1024,
            im_resize_height=1024,
            function_img=[],
            function_diff=[normalize_image],
        )
        plot_image_stack(im_stack)

    def test_plot_batch(self):
        particle = moving_spherical_plankton()
        optics = plankton_brightfield()
        sequential_particle = Sequential(
            particle, position=get_position_moving_plankton
        )
        sequence = create_sequence(1, sequential_particle, optics, 0, 1)
        sequence_length = 3
        imaged_particle_sequence = Sequence(sequence, sequence_length=sequence_length)
        batch_function = create_custom_batch_function(imaged_particle_sequence)
        plot_batch(imaged_particle_sequence, batch_function)

    def test_plot_prediction(self):
        model = generate_unet()
        folder_path = sys.path[0] + "\\S arenicola and R baltica"
        im_stack = get_image_stack(
            outputs=[0],
            folder_path=folder_path,
            frame_im0=16,
            im_size_width=1024,
            im_size_height=1024,
            im_resize_width=1024,
            im_resize_height=1024,
            function_img=[],
            function_diff=[normalize_image],
        )
        plot_prediction(model, im_stack)

    def test_plot_and_save_track(self):
        positions = [
            np.random.rand(10, 2) * 1000,
            np.random.rand(10, 2) * 1000,
            np.random.rand(10, 2) * 1000,
        ]
        list_of_plankton = assign_positions_to_planktons(
            positions, max_dist=15, time_threshold=5, extrapolate=True
        )
        folder_path = path = sys.path[0] + "\\S arenicola and R baltica"

        plot_and_save_track(
            no_of_frames=1,
            plankton_track=list_of_plankton,
            plankton_dont_track=None,
            folder_path=folder_path,
            frame_im0=0,
            save_images=0,
            show_plankton_track=True,
            show_plankton_dont_track=0,
            show_specific_plankton=None,
            show_numbers_track=True,
            show_numbers_dont_track=0,
            show_numbers_specific_plankton=0,
            specific_plankton=None,
        )
        print(
            "Blue circles are not meant to overlap plankton, if the plot is shown it works."
        )

    def test_plot_found_positions(self):
        positions = [
            np.random.rand(10, 2) * 1000,
            np.random.rand(10, 2) * 1000,
            np.random.rand(10, 2) * 1000,
        ]
        plot_found_positions(positions)


if __name__ == "__main__":
    unittest.main()
