import unittest
from loader import *


class TestLoader(unittest.TestCase):
    def test_stationary_spherical_plankton(self):
        particle = stationary_spherical_plankton()
        optics = plankton_brightfield()
        output_image = create_image(1, particle, optics, 0, 1).resolve()
        self.assertEqual(str(type(output_image)), "<class 'deeptrack.image.Image'>")
        self.assertEqual(output_image.shape, (256, 256, 1))

    def test_stationary_ellipsoid_plankton(self):
        particle = stationary_ellipsoid_plankton()
        optics = plankton_brightfield()
        output_image = create_image(1, particle, optics, 0, 1).resolve()
        self.assertEqual(str(type(output_image)), "<class 'deeptrack.image.Image'>")
        self.assertEqual(output_image.shape, (256, 256, 1))

    def test_moving_spherical_plankton(self):
        particle = moving_spherical_plankton()
        optics = plankton_brightfield()
        output_image = create_image(1, particle, optics, 0, 1).resolve()
        self.assertEqual(str(type(output_image)), "<class 'deeptrack.image.Image'>")
        self.assertEqual(output_image.shape, (256, 256, 1))

    def test_moving_ellispoid_plankton(self):
        particle = moving_spherical_plankton()
        optics = plankton_brightfield()
        output_image = create_image(1, particle, optics, 0, 1).resolve()
        self.assertEqual(str(type(output_image)), "<class 'deeptrack.image.Image'>")
        self.assertEqual(output_image.shape, (256, 256, 1))

    def test_create_sequence(self):
        particle = moving_spherical_plankton()
        optics = plankton_brightfield()
        sequential_particle = Sequential(
            particle, position=get_position_moving_plankton
        )
        sequence = create_sequence(1, sequential_particle, optics, 0, 1)
        sequence_length = 3
        imaged_particle_sequence = Sequence(
            sequence, sequence_length=sequence_length
        ).resolve()
        self.assertIsInstance(imaged_particle_sequence, list)
        self.assertEqual(np.asarray(imaged_particle_sequence).shape, (3, 256, 256, 1))

    def test_get_target_image(self):
        particle = stationary_spherical_plankton()
        optics = plankton_brightfield()
        output_image = create_image(1, particle, optics, 0, 1).resolve()
        self.assertEqual(get_target_image(output_image).shape, (256, 256, 2))
        self.assertIsInstance(get_target_image(output_image), np.ndarray)

    def test_get_target_sequence(self):
        particle = moving_spherical_plankton()
        optics = plankton_brightfield()
        sequential_particle = Sequential(
            particle, position=get_position_moving_plankton
        )
        sequence = create_sequence(1, sequential_particle, optics, 0, 1)
        sequence_length = 3
        imaged_particle_sequence = Sequence(
            sequence, sequence_length=sequence_length
        ).resolve()
        target = get_target_sequence(imaged_particle_sequence)
        self.assertEqual(target.shape, (256, 256, 4))
        self.assertIsInstance(target, np.ndarray)

    def test_create_custom_batch_function(self):
        particle = moving_spherical_plankton()
        optics = plankton_brightfield()
        sequential_particle = Sequential(
            particle, position=get_position_moving_plankton
        )
        sequence = create_sequence(1, sequential_particle, optics, 0, 1)
        sequence_length = 3
        imaged_particle_sequence = Sequence(
            sequence, sequence_length=sequence_length
        ).resolve()
        batch_function = create_custom_batch_function(imaged_particle_sequence)
        self.assertEqual(str(type(batch_function)), "<class 'function'>")
        train_stack = batch_function(imaged_particle_sequence)
        self.assertEqual(train_stack.shape, (256, 256, 5))


if __name__ == "__main__":
    unittest.main()
