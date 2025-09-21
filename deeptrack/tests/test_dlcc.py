# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name
 
# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import shutil
import tempfile
from pathlib import Path

import numpy as np
from numpy.random import Generator, PCG64
from PIL import Image

import deeptrack as dt
from deeptrack import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


class TestDLCC(unittest.TestCase):

    def test_3_B(self):
        ## PART 1
        # Image pipeline with reproducible randomness.
        rng = Generator(PCG64(42))

        image_size = 3

        particle = dt.scatterers.MieSphere(
            position=lambda: rng.uniform(image_size / 2 - 5,
                                         image_size / 2 + 5, 2),
            z=lambda: rng.uniform(-1, 1),
            radius=lambda: rng.uniform(500, 600) * 1e-9,
            refractive_index=lambda: rng.uniform(1.37, 1.42),
            position_unit="pixel",
        )

        brightfield_microscope = dt.optics.Brightfield(
            wavelength=630e-9,
            NA=0.8,
            resolution=1e-6,
            magnification=15,
            refractive_index_medium=1.33,
            output_region=(0, 0, image_size, image_size),
        )

        imaged_particle = brightfield_microscope(particle)

        # First resolve
        expected_first = np.array(
            [[[0.84823867], [0.84193078], [0.85179172]],
            [[0.80131109], [0.79478238], [0.80499449]],
            [[0.76094944], [0.75431347], [0.76469805]]]
        )
        actual_first = imaged_particle()
        np.testing.assert_allclose(actual_first, expected_first,
                                   rtol=1e-7, atol=1e-7)

        # No change when resolving again
        np.testing.assert_allclose(imaged_particle(), expected_first,
                                   rtol=1e-7, atol=1e-7)

        # Change after update
        expected_second = np.array(
            [[[1.02220767], [0.98545219], [0.95156275]],
            [[0.98707199], [0.94484236], [0.90718955]],
            [[0.94221739], [0.89565235], [0.85521306]]]
        )
        actual_second = imaged_particle.update()()
        np.testing.assert_allclose(actual_second, expected_second,
                                   rtol=1e-7, atol=1e-7)

        ## PART 2
        # Poisson uses np.random so no randomness reproducibility,
        # so images are not reproducible.
        noise = dt.Poisson(
            min_snr=5,
            max_snr=20,
            background=1,
            snr=lambda min_snr, max_snr: rng.uniform(min_snr, max_snr),
        )
        noisy_imaged_particle = imaged_particle >> noise

        normalization = dt.NormalizeMinMax(
            lambda: rng.uniform(0.0, 0.2),
            lambda: rng.uniform(0.8, 1.0),
        )
        image_pipeline = noisy_imaged_particle >> normalization

        # First resolve
        actual_first = image_pipeline()

        # No change when resolving again
        np.testing.assert_allclose(image_pipeline(), actual_first,
                                rtol=1e-7, atol=1e-7)

        # Change after update
        actual_second = image_pipeline.update()()
        assert float(np.max(np.abs(actual_first - actual_second))) > 1e-6

        ## PART 3
        # Images are not reproducible becaus eof Poisson, but positions are.
        rng = Generator(PCG64(42))

        pipeline = image_pipeline & particle.position

        # First resolve
        _, actual_position_first = pipeline.update()()

        expected_position_first = np.array([4.23956049, 0.8887844 ])
        np.testing.assert_allclose(actual_position_first,
                                   expected_position_first,
                                   rtol=1e-7, atol=1e-7)

        # No change when resolving again
        _, actual_position_first_2 = pipeline()
        np.testing.assert_allclose(actual_position_first_2,
                                   expected_position_first,
                                   rtol=1e-7, atol=1e-7)

        # Change after update
        expected_position_second = np.array([-2.21886367,  1.00385938])
        _, actual_position_second = pipeline.update()()
        np.testing.assert_allclose(actual_position_second,
                                   expected_position_second,
                                   rtol=1e-7, atol=1e-7)

    def test_4_1(self):
        ## PART 1
        # Deterministic pipeline.

        particle = dt.Sphere(
            position=np.array([0.5, 0.5]) * 4,
            position_unit="pixel",
            radius=500 * dt.units.nm,
            refractive_index=1.45 + 0.02j,
        )

        brightfield_microscope = dt.Brightfield(
            wavelength=500 * dt.units.nm,
            NA=1.0,
            resolution=1 * dt.units.um,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 4, 4),
        )

        illuminated_sample = brightfield_microscope(particle)

        # First resolve
        expected_first = np.array(
            [[[0.55382582], [0.55944586], [0.54341977], [0.55944587]],
            [[0.55944586], [0.48773998], [0.43907532], [0.48773999]],
            [[0.54341977], [0.43907532], [0.37978135], [0.43907532]],
            [[0.55944587], [0.48773999], [0.43907532], [0.48773999]]]
        )
        actual_first = illuminated_sample()
        np.testing.assert_allclose(actual_first, expected_first,
                                   rtol=1e-7, atol=1e-7)

        # No change when resolving again
        np.testing.assert_allclose(illuminated_sample(), expected_first,
                                   rtol=1e-7, atol=1e-7)

        # No change also after update (deterministic pipeline)
        np.testing.assert_allclose(illuminated_sample.update()(),
                                   expected_first,
                                   rtol=1e-7, atol=1e-7)

        ## PART 2
        # Non-reproducible randomness for noisy_particle.
        if TORCH_AVAILABLE:
            clean_particle = (
                illuminated_sample
                >> dt.NormalizeMinMax()
                >> dt.MoveAxis(2, 0)
                >> dt.pytorch.ToTensor(dtype=torch.float)
            )

            noise = dt.Poisson(snr=lambda: 2.0 + np.random.rand())

            noisy_particle = (
                illuminated_sample >> noise
                >> dt.NormalizeMinMax()
                >> dt.MoveAxis(2, 0)
                >> dt.pytorch.ToTensor(dtype=torch.float)
            )

            pip = noisy_particle & clean_particle

            # First resolve
            actual_noisy_first, actual_clean_first = pip()
            expected_clean_first = torch.tensor(
                [[[0.9687, 1.0000, 0.9108, 1.0000],
                [1.0000, 0.6009, 0.3300, 0.6009],
                [0.9108, 0.3300, 0.0000, 0.3300],
                [1.0000, 0.6009, 0.3300, 0.6009]]]
            ).to(device=actual_clean_first.device,
                 dtype=actual_clean_first.dtype)
            torch.testing.assert_close(actual_clean_first,
                                       expected_clean_first,
                                       rtol=1e-7, atol=1e-4)

            # No change after resolving again
            actual_noisy_first_2, actual_clean_first_2 = pip()
            torch.testing.assert_close(actual_clean_first_2,
                                       expected_clean_first,
                                       rtol=1e-7, atol=1e-4)
            torch.testing.assert_close(actual_noisy_first_2,
                                       actual_noisy_first,
                                       rtol=1e-7, atol=1e-4)

            # No change for clean also after update (deterministic pipeline),
            # but change for noisy
            actual_noisy_second, actual_clean_second = pip.update().resolve()
            torch.testing.assert_close(actual_clean_second,
                                       expected_clean_first,
                                       rtol=1e-7, atol=1e-4)
            self.assertFalse(torch.allclose(
                actual_noisy_first, actual_noisy_second, rtol=1e-7, atol=1e-4
            ))

        ## PART 3
        # Verify generation of blank image.
        if TORCH_AVAILABLE:
            blank = brightfield_microscope(particle ^ 0)
            blank_pip = (
                blank # >> noise >> dt.NormalizeMinMax()
                >> dt.MoveAxis(2, 0)
                >> dt.pytorch.ToTensor(dtype=torch.float)
            )

            expected = torch.tensor(
                [[[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]]]
            )
            torch.testing.assert_close(blank_pip(), expected,
                                       rtol=1e-7, atol=1e-4)

        ## PART 4
        # Check diverse particle pipeline.
        if TORCH_AVAILABLE:
            diverse_particle = dt.Sphere(
                position=lambda: np.array([.2, .2]
                                          + np.random.rand(2) * .6) * 4,
                radius=lambda: 500 * dt.units.nm * (1 + np.random.rand()),
                position_unit="pixel",
                refractive_index=1.45 + 0.02j,
            )
            diverse_illuminated_sample = \
                brightfield_microscope(diverse_particle)
            diverse_clean_particle = (
                diverse_illuminated_sample
                >> dt.NormalizeMinMax()
                >> dt.MoveAxis(2, 0)
                >> dt.pytorch.ToTensor(dtype=torch.float)
            )
            diverse_noisy_particle = (
                diverse_illuminated_sample
                >> noise
                >> dt.NormalizeMinMax()
                >> dt.MoveAxis(2, 0)
                >> dt.pytorch.ToTensor(dtype=torch.float)
            )
            diverse_pip = diverse_noisy_particle & diverse_clean_particle

            # First resolve
            diverse_noisy_first, diverse_clean_first = diverse_pip()

            # Idempotent without update()
            diverse_noisy_first_2, diverse_clean_first_2 = diverse_pip()
            torch.testing.assert_close(diverse_clean_first_2,
                                       diverse_clean_first,
                                       rtol=1e-7, atol=1e-4)
            torch.testing.assert_close(diverse_noisy_first_2,
                                       diverse_noisy_first,
                                       rtol=1e-7, atol=1e-4)

            # After update(), BOTH should change (geometry + noise)
            diverse_noisy_second, diverse_clean_second = \
                diverse_pip.update().resolve()
            self.assertFalse(torch.allclose(
                diverse_clean_second, diverse_clean_first, rtol=1e-7, atol=1e-4
            ))
            self.assertFalse(torch.allclose(
                diverse_noisy_second, diverse_noisy_first, rtol=1e-7, atol=1e-4
            ))

    def test_4_A(self):
        if TORCH_AVAILABLE:
            # Temporary root (deleted finally)
            tmp_root = tempfile.mkdtemp(prefix="mnist_like_")
            data_root = Path(tmp_root) / "mnist"
            train_dir = data_root / "train"
            test_dir  = data_root / "test"

            try:
                train_dir.mkdir(parents=True, exist_ok=True)
                test_dir.mkdir(parents=True, exist_ok=True)

                H, W = 8, 8

                # Non-uniform patterns
                grad = np.linspace(0, 255, H*W, dtype=np.uint8).reshape(H, W)
                checker = (
                    (np.indices((H, W)).sum(axis=0) % 2) * 255
                ).astype(np.uint8)
                grad_inv = (255 - grad).astype(np.uint8)
                stripes = np.tile(
                    ((np.arange(W) % 2) * 255).astype(np.uint8), (H, 1)
                )

                # Save 2 grayscale images in train/
                Image.fromarray(grad, mode="L") \
                    .save(train_dir / "0_train.png")
                Image.fromarray(checker, mode="L") \
                    .save(train_dir / "1_train.png")

                # Save 2 grayscale images in test/
                Image.fromarray(grad_inv, mode="L") \
                    .save(test_dir / "0_test.png")
                Image.fromarray(stripes, mode="L") \
                    .save(test_dir / "1_test.png")

                # print("Data root:", data_root)
                # print("Train files:", sorted(os.listdir(train_dir)))
                # print("Test files:",  sorted(os.listdir(test_dir)))

                ## PART 1
                # Loading image files into a pipeline.

                train_files = dt.sources.ImageFolder(root=str(train_dir))
                test_files  = dt.sources.ImageFolder(root=str(test_dir))
                files = dt.sources.Join(train_files, test_files)

                assert len(train_files) == 2
                assert len(test_files) == 2

                image_pip = (
                    dt.LoadImage(files.path)
                    >> dt.NormalizeMinMax()
                    >> dt.MoveAxis(2, 0)
                    >> dt.pytorch.ToTensor(dtype=torch.float)
                )

                train_dataset = dt.pytorch.Dataset(
                    image_pip & image_pip,
                    inputs=train_files,
                )

                # Get images
                x_a, x_b = train_dataset[0]  # Tensors, identical content
                assert isinstance(x_a, torch.Tensor)
                assert isinstance(x_b, torch.Tensor)
                assert x_a.shape == x_b.shape
                assert torch.equal(x_a, x_b)
                assert len(train_dataset) == len(train_files) == 2
                assert x_a.ndim == 3 and x_a.shape[1:] == (H, W)
                assert x_a.dtype == torch.float32
                assert 0.0 <= float(x_a.min()) <= float(x_a.max()) <= 1.0

                # With DataLoader
                loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=2, shuffle=False,
                )
                for xa, xb in loader:
                    assert xa.shape == xb.shape
                    assert xa.ndim == 4 \
                        and xa.shape[1:] == x_a.shape  # (B,C,H,W)

                ## PART 2
                # Test dataset with label pipelines.

                label_pip = dt.Value(files.label_name[0]) >> int
                test_dataset = dt.pytorch.Dataset(
                    image_pip & label_pip, inputs=test_files
                )

                assert len(test_dataset) == len(test_files) == 2

                x0, y0 = test_dataset[0]
                assert isinstance(x0, torch.Tensor)
                assert x0.ndim == 3 and x0.shape[1:] == (H, W)
                assert 0.0 <= float(x0.min()) <= float(x0.max()) <= 1.0    
                assert isinstance(y0, torch.Tensor)
                assert y0.ndim == 1 and y0.shape == (1,)

                # Same index is idempotent
                x0b, y0b = test_dataset[0]
                torch.testing.assert_close(x0b, x0, rtol=0.0, atol=0.0)
                assert y0b == y0

                # Check we see both labels {0,1} across the dataset
                labels = [test_dataset[i][1] for i in range(len(test_dataset))]
                assert labels == [torch.tensor([0]), torch.tensor([1])]

                # DataLoader sanity
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=2, shuffle=False
                )
                xb, yb = next(iter(test_loader))
                assert xb.ndim == 4 and xb.shape[1:] == x0.shape  # (B,C,H,W)
                assert yb.ndim == 2 and yb.shape[0] == xb.shape[0]

            except Exception:
                raise
            finally:
                # Clean up the temporary dataset tree
                shutil.rmtree(tmp_root, ignore_errors=True)

    def test_4_B(self):
        pass  # Essentially same code as test_4_A

    def test_4_C(self):
        if TORCH_AVAILABLE:
            ## PART 1
            # Load dataframe.
            from io import StringIO
            import pandas as pd

            csv_data = (
                "-0.1,-2.8,-3.7,-4.3,-4.3,-3.4,-2.1,-1.8,-1.2,-0.2,-0.3,1.0\n"
                "-1.1,-3.9,-4.2,-4.5,-4.0,-3.2,-1.5,-0.9,0.04,0.26,0.64,0.0\n"
                "-0.5,-2.5,-3.8,-4.5,-4.1,-3.1,-1.7,-1.1,-0.3,-0.0,-0.0,1.0\n"
                "0.49,-1.9,-3.6,-4.3,-4.2,-3.8,-1.6,-1.3,-0.9,-0.6,-0.4,0.0\n"
                "0.80,-0.8,-2.3,-3.9,-4.3,-2.5,-1.7,-1.5,-0.7,-0.5,-0.3,1.0\n"
                "0.80,-0.8,-2.3,-3.9,-3.8,-2.5,-1.7,-1.5,-0.7,-0.5,-0.3,0.0\n"
                "-0.1,-2.8,-3.7,-4.3,-3.4,-2.1,-1.8,-1.2,-0.4,-0.2,-0.3,1.0\n"
                "-1.1,-3.9,-4.5,-4.0,-3.2,-1.5,-0.9,-0.7,0.04,0.26,0.64,0.0\n"
                "-0.5,-3.8,-4.5,-4.1,-3.1,-1.7,-1.4,-1.1,-0.3,-0.0,-0.0,1.0\n"
                "-1.9,-3.6,-4.3,-4.2,-3.8,-2.9,-1.6,-1.3,-0.9,-0.6,-0.4,0.0"
            )

            dataframe = pd.read_csv(StringIO(csv_data), header=None)
            raw_data = dataframe.values
            ecgs = raw_data[:, 1:-2]
            labels = raw_data[:, -1].astype(bool)

            sources = dt.sources.Source(ecg=ecgs, is_normal=labels)
            train_sources, test_sources = \
                dt.sources.random_split(sources, [0.8, 0.2])
            normal_sources = \
                train_sources.filter(lambda ecg, is_normal: is_normal)

            assert len(sources) == 10
            assert len(train_sources) == 8
            assert len(test_sources) == 2

            ## PART 2
            # Instantiate and use pipeline.
            min_normal = np.min([source["ecg"] for source in normal_sources])
            max_normal = np.max([source["ecg"] for source in normal_sources])

            ecg_pip = (
                dt.Value(sources.ecg - min_normal) / (max_normal - min_normal)
                >> dt.Unsqueeze(axis=0)
                >> dt.pytorch.ToTensor(dtype=torch.float)
            )
            label_pip = dt.Value(sources.is_normal)

            # All normalized values should be between 0 and 1
            for i in range(len(sources)):
                ecg = ecg_pip(sources[i])
                assert isinstance(ecg, torch.Tensor)
                assert 0 <= ecg.min() <= 1

            # All labels should be bool
            for i in range(len(sources)):
                label = label_pip(sources[i])
                assert not isinstance(label, torch.Tensor)
                assert isinstance(label, (bool, np.bool_))

            for source in sources:
                ecg, label = (ecg_pip & label_pip)(source)
                assert 0 <= ecg.min() <= 1
                assert isinstance(label, (bool, np.bool_))

            # PART 3
            train_dataset = dt.pytorch.Dataset(ecg_pip & ecg_pip,
                                               inputs=normal_sources)
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=2)

            for ecg_in, ecg_out in loader:
                assert torch.equal(ecg_in, ecg_out)

    def test_5_1(self):
        pass


if __name__ == "__main__":
    unittest.main()
