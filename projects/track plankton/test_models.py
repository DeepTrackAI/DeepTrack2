import unittest
from models import *
from loader import *
from deeptrack.generators import ContinuousGenerator


class TestModels(unittest.TestCase):
    def test_softmax_categorical(self):
        t = np.ones((1, 2, 2))
        p = np.ones((1, 2, 2))
        t[0, :, :] = np.array([[1, 2], [2, 2]])
        p[0, :, :] = np.array([[1, 2], [2, 1]])
        T = K.constant(t)
        P = K.constant(p)
        error = np.array(softmax_categorical(T, P)).tolist()
        self.assertEqual(error, 0.5198612809181213)

    def test_generate_unet(self):
        model = generate_unet()
        self.assertEqual(str(type(model)), "<class 'deeptrack.models.KerasModel'>")

    def test_train_model_early_stopping(self):
        particle = stationary_spherical_plankton()
        optics = plankton_brightfield()
        output_image = create_image(1, particle, optics, 0, 1)
        batch_function = create_custom_batch_function(
            output_image.resolve(), outputs=[0]
        )

        generator = ContinuousGenerator(
            output_image,
            get_target_image,
            batch_function,
            batch_size=1,
            min_data_size=1,
            max_data_size=4,
        )

        model = generate_unet(None, None, 1, 2)
        im_stack = np.ones((1, 32, 32, 1))
        pred1 = model.predict(im_stack)
        model = train_model_early_stopping(
            model, generator, patience=10, epochs=3, steps_per_epoch=1
        )
        pred2 = model.predict(im_stack)
        self.assertNotEqual(pred1[0][0][0][0], pred2[0][0][0][0])


if __name__ == "__main__":
    unittest.main()
