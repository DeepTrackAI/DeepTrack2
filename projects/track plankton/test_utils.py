import unittest
from utils import *
import sys
from models import *


class TestUtils(unittest.TestCase):
    def test_normalize_image(self):
        image = np.array([1, 2, 3])
        normalized_image = normalize_image(image)
        self.assertEqual(sum(normalized_image), sum(np.array([0.0, 0.5, 1.0])))

    def test_remove_running_mean(self):
        path = sys.path[0] + "\\S arenicola and R baltica"
        image = sys.path[0] + "\\S arenicola and R baltica\\frame00.jpg"
        image = cv2.imread(sys.path[0] + "\\S arenicola and R baltica\\frame00.jpg", 0)
        path = sys.path[0] + "\\S arenicola and R baltica"
        im = remove_running_mean(
            image, path_folder=path, tot_no_of_frames=10, center_frame=4
        )
        self.assertEqual(np.mean(im), 0.734881878711456)
        self.assertEqual(im.shape, (1024, 1280))

    def test_get_mean_image(self):
        path = sys.path[0] + "\\S arenicola and R baltica"
        im = get_mean_image(path)
        self.assertEqual(np.mean(im), 2.3189709472656252)
        self.assertEqual(im.shape, (1024, 1280))

    def test_get_image_stack(self):
        path = sys.path[0] + "\\S arenicola and R baltica"
        im_stack = get_image_stack(folder_path=path)
        self.assertEqual(im_stack.shape, (1, 1024, 1280, 1))

    def test_get_blob_center(self):
        a = np.zeros((5, 5))
        a[2, 2] = 1
        center = get_blob_center(1, a)
        self.assertEqual(center, (2.0, 2.0))
        self.assertIsInstance(center, tuple)

    def test_get_blob_centers(self):
        a = np.zeros((6, 6))
        a[1, 1] = 1
        a[4, 4] = 1
        a[0, 4] = 1
        centers = get_blob_centers(a)
        self.assertEqual(
            sum(sum(centers)), sum(sum(np.array([[0.0, 4.0], [1.0, 1.0], [4.0, 4.0]])))
        )

    def test_extract_positions_from_prediction(self):
        path = sys.path[0] + "\\S arenicola and R baltica"
        im_stack = get_image_stack(folder_path=path)
        model = generate_unet()
        prediction_shape = model.predict(im_stack).shape
        positions = extract_positions_from_prediction(im_stack, model, layer=1)
        self.assertEqual(prediction_shape, (1, 1024, 1280, 2))
        self.assertIsInstance(positions, np.ndarray)

    def test_extract_positions(self):
        path = sys.path[0] + "\\S arenicola and R baltica"
        model = generate_unet()
        positions = extract_positions(3, 0, model=model, layer=1, folder_path=path)
        self.assertEqual(len(positions), 3)

    def test_initialize_plankton(self):
        positions = np.array([[1, 2], [1, 2], [1, 2]])
        dict_of_plankton = initialize_plankton(positions, 3)
        self.assertIsInstance(dict_of_plankton, dict)
        self.assertEqual(len(dict_of_plankton), 3)

    def test_update_list_of_plankton(self):
        positions1 = np.array([[1, 2], [1, 2], [10, 2]])
        dict_of_plankton = initialize_plankton(positions1, 2)
        positions2 = np.array([[1, 200], [13, 2], [1, 4]])
        update_list_of_plankton(
            list_of_plankton=dict_of_plankton, positions=positions2, timestep=1
        )
        self.assertEqual(sum(dict_of_plankton["plankton3"].positions[1]), 201)

    def test_assign_positions_to_planktons(self):
        positions = [np.nan] * 2
        positions[0] = np.array([[1, 2], [1, 2], [10, 2]])
        positions[1] = np.array([[1, 201], [13, 2], [1, 4]])
        dict_of_plankton = assign_positions_to_planktons(positions)
        self.assertEqual(sum(dict_of_plankton["plankton3"].positions[1]), 202)

    def test_interpolate_gaps_in_plankton_positions(self):
        positions = [np.nan] * 3
        positions[0] = np.array([[1, 20], [1, 2]])
        positions[1] = np.array([[1, 40], [1, 2]])
        positions[2] = np.array([[1, 24], [1, 2]])
        dict_of_plankton = assign_positions_to_planktons(positions)
        interpolate_gaps_in_plankton_positions(dict_of_plankton)
        self.assertEqual(sum(dict_of_plankton["plankton0"].positions[1]), 23)

    def test_extrapolate_positions(self):
        positions = [np.nan] * 4
        positions[0] = np.array([[1, 20], [1, 2]])
        positions[1] = np.array([[1, 22], [1, 2]])
        positions[2] = np.array([[1, 24], [1, 2]])
        dict_of_plankton = assign_positions_to_planktons(positions)
        self.assertEqual(extrapolate_positions(dict_of_plankton, timestep=3)[0, 1], 26)

    def test_trim_list_from_stationary_plankton(self):
        positions = [np.nan] * 3
        positions[0] = np.array([[1, 20], [1, 2]])
        positions[1] = np.array([[1, 22], [1, 2]])
        positions[2] = np.array([[1, 24], [1, 2]])
        dict_of_plankton = assign_positions_to_planktons(positions)
        dict_of_plankton = trim_list_from_stationary_planktons(
            dict_of_plankton, min_distance=2
        )
        self.assertEqual(len(dict_of_plankton), 1)

    def test_split_plankton(self):
        positions = [np.nan] * 3
        positions[0] = np.array([[1, 20], [1, 2]])
        positions[1] = np.array([[1, 22], [1, 200]])
        positions[2] = np.array([[1, 24], [1, 300]])
        dict_of_plankton = assign_positions_to_planktons(positions)
        list1, list2 = split_plankton(dict_of_plankton, percentage_threshold=0.67)
        self.assertEqual(len(list1), 1)
        self.assertEqual(len(list2), 3)

    def test_get_mean_net_and_gross_distance(self):
        positions = [np.nan] * 3
        positions[0] = np.array([[1, 20], [1, 2]])
        positions[1] = np.array([[1, 22], [1, 2]])
        positions[2] = np.array([[1, 24], [1, 3]])
        dict_of_plankton = assign_positions_to_planktons(positions)
        net, gross = get_mean_net_and_gross_distance(dict_of_plankton)
        self.assertEqual(net[1, 0], 2.5)

    def test_crop_and_append_image(self):
        image = cv2.imread(sys.path[0] + "\\S arenicola and R baltica\\frame00.jpg", 0)
        cropped_image = crop_and_append_image(
            image=image,
            col_delete_list=[0, 200, 400, 900],
            row_delete_list=[0, 400, 500, 700],
            mult_of=16,
        )
        self.assertEqual(cropped_image.shape, (416, 576))
        self.assertEqual(sum(sum(cropped_image)), 75422)

    def test_fix_positions_from_cropping(self):
        positions = [np.nan] * 3
        positions[0] = np.array([[298.0, 20.0], [1.0, 2.0]])
        positions[1] = np.array([[110.0, 22.0], [1.0, 2.0]])
        positions[2] = np.array([[100.0, 24.0], [1.0, 3.0]])
        new_positions = fix_positions_from_cropping(
            positions,
            col_delete_list=[0, 200, 400, 900],
            row_delete_list=[0, 400, 500, 700],
        )
        self.assertEqual(new_positions[0][0, 0], 898.0)

    def test_get_track_durations(self):
        positions = [np.nan] * 3
        positions[0] = np.array([[298.0, 20.0], [1.0, 2.0]])
        positions[1] = np.array([[110.0, 22.0], [1.0, 2.0]])
        positions[2] = np.array([[100.0, 24.0], [1.0, 3.0]])
        dict_of_plankton = assign_positions_to_planktons(positions)
        self.assertEqual(sum(get_track_durations(dict_of_plankton)), 4.0)

    def test_get_found_plankton_at_timestep(self):
        positions = [np.nan] * 3
        positions[0] = np.array([[298.0, 20.0], [1.0, 2.0]])
        positions[1] = np.array([[110.0, 22.0], [1.0, 2.0]])
        positions[2] = np.array([[100.0, 24.0], [1.0, 3.0]])
        dict_of_plankton = assign_positions_to_planktons(positions)
        self.assertTrue(
            (
                get_found_plankton_at_timestep(dict_of_plankton)
                == np.array([2.0, 2.0, 2.0])
            ).all()
        )

    def test_extract_positions_from_list(self):
        positions = [np.nan] * 3
        positions[0] = np.array([[298.0, 20.0], [1.0, 2.0]])
        positions[1] = np.array([[110.0, 22.0], [1.0, 2.0]])
        positions[2] = np.array([[100.0, 24.0], [1.0, 3.0]])
        dict_of_plankton = assign_positions_to_planktons(positions)
        self.assertEqual(extract_positions_from_list(dict_of_plankton)[0, 0], 298.0)


if __name__ == "__main__":
    unittest.main()
