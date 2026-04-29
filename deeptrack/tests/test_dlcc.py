import unittest

import warnings
from contextlib import contextmanager

import glob
import platform
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

    _EXPECTED_OPTICS_WARNING_PATTERNS = (
        r"Brightfield imaging from ScatteredVolume assumes a weak-phase / projection approximation.*",
        r"Darkfield imaging from ScatteredVolume is a very rough approximation.*",
        r"Approximating darkfield contrast from refractive index.*",
        r"Fluorescence scatterer has no 'intensity'.*",
    )

    @contextmanager
    def _suppress_expected_optics_warnings(self):
        with warnings.catch_warnings():
            for pattern in self._EXPECTED_OPTICS_WARNING_PATTERNS:
                warnings.filterwarnings(
                    "ignore",
                    message=pattern,
                    category=UserWarning,
                )
            yield

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

        with self._suppress_expected_optics_warnings():
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
        
        with self._suppress_expected_optics_warnings():
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
        with self._suppress_expected_optics_warnings():
            actual_first = image_pipeline()

        # No change when resolving again
        np.testing.assert_allclose(image_pipeline(), actual_first,
                                rtol=1e-7, atol=1e-7)

        # Change after update
        with self._suppress_expected_optics_warnings():
            actual_second = image_pipeline.update()()
        assert float(np.max(np.abs(actual_first - actual_second))) > 1e-6

        ## PART 3
        # Images are not reproducible becaus eof Poisson, but positions are.
        rng = Generator(PCG64(42))

        pipeline = image_pipeline & particle.position

        # First resolve
        with self._suppress_expected_optics_warnings():
            _, actual_position_first = pipeline.update()()

        expected_position_first = np.array([4.23956049, 0.8887844 ])
        np.testing.assert_allclose(actual_position_first,
                                   expected_position_first,
                                   rtol=1e-7, atol=1e-7)

        # No change when resolving again
        with self._suppress_expected_optics_warnings():
            _, actual_position_first_2 = pipeline()
        np.testing.assert_allclose(actual_position_first_2,
                                   expected_position_first,
                                   rtol=1e-7, atol=1e-7)

        # Change after update
        expected_position_second = np.array([-2.21886367,  1.00385938])
        
        with self._suppress_expected_optics_warnings():
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

        with self._suppress_expected_optics_warnings():
            actual_first = illuminated_sample()
        np.testing.assert_allclose(actual_first, expected_first,
                                   rtol=1e-7, atol=1e-7)

        # No change when resolving again
        with self._suppress_expected_optics_warnings():
            np.testing.assert_allclose(illuminated_sample(), expected_first,
                                        rtol=1e-7, atol=1e-7)

        # No change also after update (deterministic pipeline)
        with self._suppress_expected_optics_warnings():
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
            with self._suppress_expected_optics_warnings():
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
            with self._suppress_expected_optics_warnings():
                actual_noisy_first_2, actual_clean_first_2 = pip()
            torch.testing.assert_close(actual_clean_first_2,
                                       expected_clean_first,
                                       rtol=1e-7, atol=1e-4)
            torch.testing.assert_close(actual_noisy_first_2,
                                       actual_noisy_first,
                                       rtol=1e-7, atol=1e-4)

            # No change for clean also after update (deterministic pipeline),
            # but change for noisy
            with self._suppress_expected_optics_warnings():
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
            with self._suppress_expected_optics_warnings():
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
            with self._suppress_expected_optics_warnings():
                diverse_noisy_first, diverse_clean_first = diverse_pip()

            # Idempotent without update()
            with self._suppress_expected_optics_warnings():
                diverse_noisy_first_2, diverse_clean_first_2 = diverse_pip()
            torch.testing.assert_close(diverse_clean_first_2,
                                       diverse_clean_first,
                                       rtol=1e-7, atol=1e-4)
            torch.testing.assert_close(diverse_noisy_first_2,
                                       diverse_noisy_first,
                                       rtol=1e-7, atol=1e-4)

            # After update(), BOTH should change (geometry + noise)
            with self._suppress_expected_optics_warnings():
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
            # Temporary root (deleted in finally)
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

            # Regression check: activating items from split/filter subsets
            # must update the original source used by the pipeline.
            assert len(normal_sources) >= 2

            normal_sources[0]()
            np.testing.assert_array_equal(
                sources.ecg(),
                normal_sources[0]["ecg"],
            )
            assert sources.is_normal() == normal_sources[0]["is_normal"]

            normal_sources[1]()
            np.testing.assert_array_equal(
                sources.ecg(),
                normal_sources[1]["ecg"],
            )
            assert sources.is_normal() == normal_sources[1]["is_normal"]

            # Ensure activation actually changes the parent source.
            assert not np.array_equal(
                normal_sources[0]["ecg"],
                normal_sources[1]["ecg"],
            )

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
            for i in range(len(normal_sources)):
                ecg = ecg_pip(normal_sources[i])
                expected = (
                    normal_sources[i]["ecg"] - min_normal
                ) / (max_normal - min_normal)
                assert isinstance(ecg, torch.Tensor)
                assert 0 <= ecg.min() <= 1
                assert ecg.max() <= 1

                np.testing.assert_allclose(
                    ecg.squeeze().detach().cpu().numpy(),
                    expected,
                    rtol=1e-6,
                    atol=1e-6,
                )

            # All labels should be bool
            for i in range(len(sources)):
                label = label_pip(sources[i])
                assert not isinstance(label, torch.Tensor)
                assert isinstance(label, (bool, np.bool_))

            for source in sources:
                ecg, label = (ecg_pip & label_pip)(source)
                assert isinstance(label, (bool, np.bool_))

            # PART 3
            train_dataset = dt.pytorch.Dataset(ecg_pip & ecg_pip,
                                               inputs=normal_sources)
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=2)

            for ecg_in, ecg_out in loader:
                assert torch.equal(ecg_in, ecg_out)

    def test_5_1(self):
        def select_labels(class_labels):
            """Create a function to filter and remap labels in ..."""
            def inner(segmentation):
                seg = segmentation.copy()
                mask = seg * np.isin(seg, class_labels).astype(np.uint8)
                new_seg = (np.select([mask == c for c in class_labels],
                                        np.arange(len(class_labels)) + 1)
                            .astype(np.uint8).squeeze())
                one_hot_encoded_seg = np.eye(len(class_labels) + 1)[new_seg]
                return one_hot_encoded_seg        
            return inner

        if TORCH_AVAILABLE:
            # Temporary root (deleted in finally)
            tmp_root = tempfile.mkdtemp(prefix="tissue_images_like_")
            data_root = Path(tmp_root) / "stack1"
            raw_dir = data_root / "raw"
            labels_dir  = data_root / "label"

            try:
                raw_dir.mkdir(parents=True, exist_ok=True)
                labels_dir.mkdir(parents=True, exist_ok=True)

                H, W = 8, 12
                values = np.array([0, 50, 100, 191, 200, 255], dtype=np.uint8)

                # Save 5 RGB images in raw/
                for i in range(5):
                    # Simple non-uniform pattern for RGB channels
                    r = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
                    g = np.tile(np.linspace(255, 0, W, dtype=np.uint8), (H, 1))
                    b = np.full((H, W), i * 50, dtype=np.uint8)
                    img_rgb = np.stack([r, g, b], axis=-1)
                    Image.fromarray(img_rgb, mode="RGB") \
                        .save(raw_dir / f"rgb_{i}.png")

                # Save 5 grayscale images in raw/
                for i in range(5):
                    # Fill image cycling through values, then shift by i
                    arr = np.tile(values, H * W // len(values) + 1)[: H * W]
                    arr = np.roll(arr, i)  # shift pattern
                    arr = arr.reshape(H, W)
                    Image.fromarray(arr, mode="L") \
                        .save(labels_dir / f"gray_{i}.png")

                raw_path = str(raw_dir)
                seg_path = str(labels_dir)    

                ## PART 1
                # Loading image files into a pipeline.
                raw_paths = dt.sources.ImageFolder(root=raw_path)
                seg_paths = dt.sources.ImageFolder(root=seg_path)
                paths = dt.sources.Source(raw=raw_paths, label=seg_paths)
                train_paths, val_paths, test_paths = \
                    dt.sources.random_split(paths, [0.6, 0.2, 0.2])

                assert len(raw_paths) == 5
                assert len(seg_paths) == 5
                assert len(train_paths) == 3
                assert len(val_paths) == 1
                assert len(test_paths) == 1

                train_srcs = train_paths.product(
                    flip_ud=[True, False], flip_lr=[True, False],
                )
                val_srcs = val_paths.constants(flip_ud=False, flip_lr=False)
                test_srcs = test_paths.constants(flip_ud=False, flip_lr=False)

                sources = dt.sources.Join(train_srcs, val_srcs, test_srcs)

                ## PART 2
                # Testing pipelines and select_labels function.
                im_pip = dt.LoadImage(sources.raw.path) >> dt.NormalizeMinMax()
                seg_pip = (dt.LoadImage(sources.label.path)
                        >> dt.Lambda(select_labels, class_labels=[255, 191]))
                pip = ((im_pip & seg_pip) >> dt.FlipLR(sources.flip_lr)
                    >> dt.FlipUD(sources.flip_ud) >> dt.MoveAxis(2, 0)
                    >> dt.pytorch.ToTensor(dtype=torch.float))

                train_dataset = dt.pytorch.Dataset(pip, train_srcs)
                val_dataset = dt.pytorch.Dataset(pip, val_srcs)
                test_dataset = dt.pytorch.Dataset(pip, test_srcs)

                assert len(train_dataset) == 12  # 3 images * 4 augmentations
                assert len(val_dataset) == 1  # No augmentations
                assert len(test_dataset) == 1  # No augmentations

                for i in range(12):
                    im, seg = train_dataset[i]
                    assert im.min() >= 0.0 and im.max() <= 1.0
                    assert seg.ndim == 3  # (num_classes, H, W)
                    assert seg.shape[0] == 3  # 3 channels
                    assert set(np.unique(seg)).issubset({0, 1})

                im, seg = val_dataset[0]
                assert im.min() >= 0.0 and im.max() <= 1.0
                assert seg.ndim == 3  # (num_classes, H, W)
                assert seg.shape[0] == 3  # 3 channels
                assert set(np.unique(seg)).issubset({0, 1})

                im, seg = test_dataset[0]
                assert im.min() >= 0.0 and im.max() <= 1.0
                assert seg.ndim == 3  # (num_classes, H, W)
                assert seg.shape[0] == 3  # 3 channels
                assert set(np.unique(seg)).issubset({0, 1})

            except Exception:
                raise
            finally:
                # Clean up the temporary dataset tree
                shutil.rmtree(tmp_root, ignore_errors=True)

    def test_5_A(self):
        ## PART 1
        # Single quantum dot.

        optics = dt.Fluorescence(
            wavelength=600 * dt.units.nm,
            NA=0.9,
            magnification=1,
            resolution=0.1 * dt.units.um,
            output_region=(0, 0, 3, 3),
        )
        particle = dt.PointParticle(
            position=(2, 2),
            intensity=1.2e4,
            z=0,
        )
        sim_im_pip = (
            optics(particle)
            >> dt.Add(30)
            >> dt.Add(82)
        )

        expected = np.array(
            [[[ 7.01659596], [16.36273566], [21.04978789]],
             [[16.36273566], [32.64875852], [40.41116424]],
             [[21.04978789], [40.41116424], [49.51533565]]]
        ) + 30 + 82

        with self._suppress_expected_optics_warnings():
            np.testing.assert_allclose(sim_im_pip(), expected,
                                    rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(sim_im_pip.update()(), expected,
                                    rtol=1e-7, atol=1e-7)

        np.random.seed(123)  # Note that this seeding is not warratied
                             # to give reproducible results across platforms
                             # so the subsequent test might fail
        sim_im_pip = (
            optics(particle)
            >> dt.Add(30)
            >> np.random.poisson
            >> dt.Add(82)
        )
        expected = np.array(
            [[[123], [122],[138]],
            [[128], [141], [151]],
            [[131], [144], [162]]]
        )
        # This test might fail (see above)
        np.testing.assert_array_equal(sim_im_pip(), expected)

        ## PART 2
        # Multiple quantum dots.

        np.random.seed(123)  # Note that this seeding is not warratied
                             # to give reproducible results across platforms
                             # so the subsequent test might fail

        optics = dt.Fluorescence(
            wavelength=600 * dt.units.nm,
            NA=0.9,
            magnification=1,
            resolution=0.1 * dt.units.um,
            output_region=(0, 0, 8, 8),
        )
        particle = dt.PointParticle(
            position=lambda: np.random.uniform(0, 8, size=2),
            intensity=lambda: np.random.uniform(6e3, 3e4),
            z=lambda: np.random.uniform(-1.5, 1.5) * dt.units.um,
        )
        postprocess = (
            dt.Add(lambda: np.random.uniform(20, 40))
            >> np.random.poisson
            >> dt.Add(lambda: np.random.uniform(70, 90))
        )
        normalization = (
            dt.AsType("float")
            >> dt.Subtract(110)
            >> dt.Divide(250)
        )
        particles = particle ^ (lambda: np.random.randint(2, 5))
        sim_im_pip = (
            optics(particles)
            >> postprocess
            >> normalization
        )

        expected_1 = np.array(
            [[[-0.00362469], [ 0.11237531], [ 0.23637531], [ 0.32037531], [ 0.38037531], [ 0.34437531], [ 0.17637531], [ 0.16837531]],
             [[ 0.07237531], [ 0.15237531], [ 0.28837531], [ 0.48837531], [ 0.46837531], [ 0.48437531], [ 0.46037531], [ 0.30837531]],
             [[ 0.04837531], [ 0.12437531], [ 0.32837531], [ 0.43237531], [ 0.63637531], [ 0.65237531], [ 0.56437531], [ 0.37237531]], 
             [[ 0.02037531], [ 0.07637531], [ 0.16437531], [ 0.26837531], [ 0.57237531], [ 0.60837531], [ 0.67637531], [ 0.45237531]], 
             [[ 0.04437531], [ 0.02437531], [ 0.20437531], [ 0.30437531], [ 0.47237531], [ 0.54037531], [ 0.63237531], [ 0.40037531]],
             [[ 0.07237531], [ 0.12437531], [ 0.24837531], [ 0.22037531], [ 0.38037531], [ 0.39237531], [ 0.42037531], [ 0.31237531]],
             [[ 0.02837531], [ 0.12037531], [ 0.22837531], [ 0.33237531], [ 0.31637531], [ 0.22037531], [ 0.19637531], [ 0.19237531]],
             [[-0.01962469], [ 0.08037531], [ 0.16037531], [ 0.22437531], [ 0.27237531], [ 0.14037531], [ 0.10437531], [ 0.03637531]]]
        )

        with self._suppress_expected_optics_warnings():
            np.testing.assert_allclose(sim_im_pip(), expected_1,
                                   rtol=1e-7, atol=1e-7)

        expected_2 = np.array(
            [[[ 0.05024189], [ 0.03024189], [ 0.05824189], [ 0.13024189], [ 0.07824189], [ 0.12224189], [ 0.13424189], [ 0.11824189]],
             [[ 0.08224189], [ 0.02624189], [ 0.08624189], [ 0.11024189], [ 0.11024189], [ 0.13824189], [ 0.11824189], [ 0.16224189]],
             [[ 0.09024189], [ 0.05424189], [ 0.04224189], [ 0.04224189], [ 0.06624189], [ 0.15824189], [ 0.11424189], [ 0.04224189]],
             [[ 0.09024189], [ 0.00624189], [ 0.05424189], [ 0.05424189], [ 0.05424189], [ 0.05024189], [ 0.01424189], [ 0.02624189]],
             [[-0.00575811], [ 0.02224189], [ 0.03424189], [ 0.04224189], [ 0.07424189], [ 0.00624189], [ 0.03424189], [ 0.01824189]],
             [[-0.02175811], [-0.00575811], [ 0.01024189], [ 0.03024189], [ 0.05024189], [ 0.05424189], [ 0.08224189], [ 0.07024189]],
             [[ 0.04624189], [-0.04575811], [-0.00175811], [ 0.02624189], [ 0.05424189], [ 0.12224189], [ 0.15024189], [ 0.11424189]],
             [[-0.02175811], [-0.01775811], [-0.01375811], [-0.02175811], [ 0.04624189], [ 0.18624189], [ 0.22624189], [ 0.19024189]]]
        )

        with self._suppress_expected_optics_warnings():
            np.testing.assert_allclose(sim_im_pip.update()(), expected_2,
                                       rtol=1e-7, atol=1e-7)

        ## PART 3
        # Complete pipeline.
        sim_mask_pip = (
            particles
            >> dt.SampleToMasks(lambda: lambda particle: particle > 0,
                                output_region=optics.output_region,
                                merge_method="or")
            >> dt.AsType("int")
            >> dt.OneHot(num_classes=2)
        )

        if TORCH_AVAILABLE:
            sim_im_mask_pip = (
                (sim_im_pip & sim_mask_pip)
                >> dt.MoveAxis(2, 0)
                >> dt.pytorch.ToTensor(dtype=torch.float)
            )

            train_dataset = dt.pytorch.Dataset(
                sim_im_mask_pip, length=320, replace=.1,
            )

            im, mask = train_dataset[0]

            # Dataset length honored
            assert len(train_dataset) == 320

            # Types and shapes
            assert isinstance(im, torch.Tensor)
            assert isinstance(mask, torch.Tensor)
            assert im.ndim == 3 and mask.ndim == 3
            assert im.shape[0] == 1  # single image channel
            assert mask.shape[0] == 2  # one-hot: bg=0, fg=1
            assert im.shape[1:] == mask.shape[1:]

            # Mask must be one-hot per pixel and binary {0,1}
            u = set(mask.unique().tolist())
            assert u.issubset({0.0, 1.0})
            assert torch.allclose(mask.sum(dim=0), torch.ones_like(mask[0]))

            # Both background and foreground should be present
            fg_sum = int(mask[1].sum().item())
            bg_sum = int(mask[0].sum().item())
            assert fg_sum > 0 and bg_sum > 0

            # Foreground pixels should be brighter than background on average
            im2d = im[0]                              # (H, W)
            fg_mean = float(im2d[mask[1].bool()].mean())
            bg_mean = float(im2d[mask[0].bool()].mean())
            assert fg_mean > bg_mean

    def test_5_B(self):
        ## PART 1
        # Loading data images and masks from files.

        # Temporary root (deleted in finally)
        tmp_root = tempfile.mkdtemp(prefix="cell_counting_like_")
        data_root = Path(tmp_root) / "base"
        images_dir = data_root / "images"
        masks_dir  = data_root / "masks"

        try:
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)

            # Synthetic image (grayscale with some blobs)
            image = np.zeros((8, 12), dtype=np.uint8)
            image[1:3, 2:4] = 128   # blob 1
            image[4:6, 6:8] = 200   # blob 2
            image[6:8, 4:6] = 255   # blob 3

            # Synthetic label mask (integer IDs for blobs in red channel)
            mask_gray = np.zeros_like(image, dtype=np.uint8)
            mask_gray[1:3, 2:4] = 1
            mask_gray[4:6, 6:8] = 1
            mask_gray[6:8, 4:6] = 1

            # Expand to RGB, putting data in channel 0, zeros in channels 1 & 2
            mask_rgb = np.stack(
                [mask_gray,
                np.zeros_like(mask_gray),
                np.zeros_like(mask_gray)],
                axis=-1,
            )

            # Save images
            Image.fromarray(image, mode="L").save(images_dir / "image_0.png")
            for i in range(1, 5):
                Image.fromarray(np.zeros_like(image), mode="L") \
                    .save(images_dir / f"image_{i}.png")

            # Save labels
            Image.fromarray(mask_rgb, mode="RGB") \
                .save(masks_dir / "mask_0.png")
            for i in range(1, 5):
                Image.fromarray(np.zeros_like(mask_rgb), mode="RGB") \
                    .save(masks_dir / f"mask_{i}.png")

            ## PART 1.1
            # Loading image files into a pipeline.
            image_paths = dt.sources.ImageFolder(root=str(images_dir))
            mask_paths = dt.sources.ImageFolder(root=str(masks_dir))
            sources = dt.sources.Source(image=image_paths, label=mask_paths)

            assert len(image_paths) == 5
            assert len(mask_paths) == 5

            ## PART 2.2
            # Pipelines.
            if TORCH_AVAILABLE:
                image_pip = (
                    dt.LoadImage(sources.image.path)
                    >> dt.Divide(3000)
                    >> dt.Clip(0, 1)
                    >> dt.AsType("float")
                )
                mask_pip = (
                    dt.LoadImage(sources.label.path)[..., :1]
                    >> dt.AsType("float")
                )
                pip = (
                    (image_pip & mask_pip)
                    >> dt.Crop(crop=(4, 6, None), corner=(0, 0))
                    >> dt.MoveAxis(2, 0)
                    >> dt.pytorch.ToTensor(dtype=torch.float)
                )
                test_dataset = dt.pytorch.Dataset(pip, sources)

                assert len(test_dataset) == 5

                for i in range(5):
                    image, mask = test_dataset[i]

                    assert isinstance(image, torch.Tensor)
                    assert image.shape == torch.Size([1, 4, 6])
                    assert image.dtype == torch.float32
                    assert torch.all(image >= 0) and torch.all(image <= 1)

                    assert isinstance(mask, torch.Tensor)
                    assert mask.shape == torch.Size([1, 4, 6])
                    assert mask.dtype == torch.float32

        except Exception:
            raise
        finally:
            # Clean up the temporary dataset tree
            shutil.rmtree(tmp_root, ignore_errors=True)

        ## PART 2
        # Simulation pipeline.

        train_image_size = 6

        def random_ellipse_axes():
            """Return the three axes of an ellipse."""
            ellipse_area = (np.random.uniform(.5, 1)) ** 2
            radius_ratio = np.random.uniform(1, 1.5)
            major_axis = np.sqrt(ellipse_area) * radius_ratio
            minor_axis = np.sqrt(ellipse_area) / radius_ratio
            z_axis = np.sqrt(ellipse_area) * np.random.uniform(0.2, 0.4)
            return (major_axis, minor_axis, z_axis) * dt.units.um

        ## PART 2.1
        np.random.seed(123)  # Note that this seeding is not warratied
                             # to give reproducible results across
                             # platforms so the subsequent test might fail


        ellipse = dt.Ellipsoid(
            radius=random_ellipse_axes,
            intensity=lambda: np.random.uniform(0.5, 1.5),
            position=lambda: np.random.uniform(2, train_image_size - 2, 
                                               size=2),
            rotation=lambda: np.random.uniform(0, 2 * np.pi),
        )
        optics = dt.Fluorescence(
            resolution=1e-6,
            magnification=6,
            wavelength=400e-9,
            NA=lambda: np.random.uniform(0.9, 1.1),
            output_region=(0, 0, train_image_size, train_image_size),
        )
        sim_im_pip = optics(ellipse)

        # Checks
        expected_image = np.array(
                    [[[0.60265415], [0.94844141], [1.14489087], [1.16483931], [1.13598992], [0.90247759]],
                     [[1.199768  ], [1.51251191], [1.74839492], [1.77029627], [1.72956925], [1.42921194]],
                     [[1.73096144], [1.825617  ], [1.87179117], [1.87245093], [1.84518863], [1.74890568]],
                     [[1.77330325], [1.85308512], [1.87854141], [1.87606849], [1.82860277], [1.74103692]],
                     [[1.53892305], [1.76151488], [1.79291875], [1.77124261], [1.55013829], [1.30663407]],
                     [[1.02576262], [1.2719972 ], [1.3016064 ], [1.27185945], [0.99481222], [0.63890969]]]
        )
        with self._suppress_expected_optics_warnings():
            image = sim_im_pip()
        try:  # Occasional error in Ubuntu system
            assert np.allclose(image, expected_image, atol=1e-6)
        except AssertionError:
            if platform.system() != "Linux":
                raise
        with self._suppress_expected_optics_warnings():
            image = sim_im_pip()
        assert np.allclose(image, expected_image, atol=1e-6)
        with self._suppress_expected_optics_warnings():
            image = sim_im_pip.update()()
        assert not np.allclose(image, expected_image, atol=1e-6)

        ## PART 2.2
        import random

        np.random.seed(123)  # Note that this seeding is not warratied
        random.seed(123)     # to give reproducible results across
                             # platforms so the subsequent test might fail

        ellipse = dt.Ellipsoid(
            radius=random_ellipse_axes,
            intensity=lambda: np.random.uniform(0.5, 1.5),
            position=lambda: np.random.uniform(2, train_image_size - 2,
                                               size=2),
            rotation=lambda: np.random.uniform(0, 2 * np.pi),
        )
        synthetic_nuclei = (
            (ellipse ^ (lambda: np.random.randint(5, 10)))
            >> dt.Pad(px=(10, 10, 10, 10), keep_size=False)
            >> dt.ElasticTransformation(alpha=100, sigma=10, order=1)
            >> dt.CropTight()
        )
        optics = dt.Fluorescence(
            resolution=1e-6,
            magnification=6,
            wavelength=400e-9,
            NA=lambda: np.random.uniform(0.9, 1.1),
            output_region=(0, 0, train_image_size, train_image_size),
        )
        sim_im_pip = optics(synthetic_nuclei)

        # Checks
        expected_image = np.array(
            [[[2.40686748], [3.57908632], [4.82880076], [5.75153091], [6.06963462], [5.57287094]],
             [[3.3805992 ], [4.90299411], [6.20417148], [6.90634308], [6.83577577], [6.16933536]],
             [[4.18260712], [5.97708425], [7.23448674], [7.48806701], [6.93908065], [5.96343732]],
             [[4.27119652], [6.1665758 ], [7.29078817], [7.50901958], [6.86948897], [5.63460567]],
             [[3.87612061], [5.88381024],  [6.76433577], [7.00694866], [6.62352318], [5.28112149]],
             [[3.07807345], [5.21008639], [6.18438896], [6.43448107], [6.07102741], [4.76105099]]]
            )
        
        with self._suppress_expected_optics_warnings():
            image = sim_im_pip()
        try:  # Occasional error in Ubuntu system
            assert np.allclose(image, expected_image, atol=1e-6)
        except AssertionError:
            if platform.system() != "Linux":
                raise

        with self._suppress_expected_optics_warnings():
            image = sim_im_pip()
        try:  # Occasional error in Ubuntu system
            assert np.allclose(image, expected_image, atol=1e-6)
        except AssertionError:
            if platform.system() != "Linux":
                raise
            
        with self._suppress_expected_optics_warnings():
            image = sim_im_pip.update()()
        assert not np.allclose(image, expected_image, atol=1e-6)

        ## PART 2.3
        np.random.seed(123)  # Note that this seeding is not warratied
        random.seed(123)     # to give reproducible results across
                             # platforms so the subsequent test might fail

        ellipse = dt.Ellipsoid(
            radius=random_ellipse_axes,
            intensity=lambda: np.random.uniform(0.5, 1.5),
            position=lambda: np.random.uniform(2, train_image_size - 2,
                                               size=2),
            rotation=lambda: np.random.uniform(0, 2 * np.pi),
        )
        synthetic_nuclei = (
            (ellipse ^ (lambda: np.random.randint(5, 10)))
            >> dt.Pad(px=(10, 10, 10, 10), keep_size=False)
            >> dt.ElasticTransformation(alpha=100, sigma=10, order=1)
            >> dt.CropTight()
        )
        synthetic_nuclei_mask = synthetic_nuclei > 0
        long_range_noise = (
            synthetic_nuclei
            >> dt.Poisson(snr=0.2)
            >> dt.GaussianBlur(sigma=3.5)
        )
        short_range_noise = (
            synthetic_nuclei
            >> dt.Poisson(snr=1.0)
            >> dt.GaussianBlur(sigma=1.5)
        )
        random_range_noise = (
            synthetic_nuclei
            >> dt.Poisson(snr=lambda: np.random.uniform(0.5, 1.5))
            >> dt.GaussianBlur(sigma=lambda: np.random.uniform(0.75, 1.5))
        )
        noisy_synthetic_nuclei = (
            synthetic_nuclei_mask
            * (long_range_noise + short_range_noise + random_range_noise) / 3
        )

        optics = dt.Fluorescence(
            resolution=1e-6,
            magnification=6,
            wavelength=400e-9,
            NA=lambda: np.random.uniform(0.9, 1.1),
            output_region=(0, 0, train_image_size, train_image_size),
        )
        sim_im_pip = optics(noisy_synthetic_nuclei)

        # Checks
        expected_image = np.array(
            [[[1.93167944], [2.69410402], [3.66954369], [4.37636897], [4.48323595], [4.12289828]],
             [[2.53519759], [3.60325565], [4.58956314], [5.15477629], [5.08360439], [4.53870126]],
             [[3.21864851], [4.44624058], [5.32791401], [5.62321336], [5.41770116], [4.70481539]],
             [[3.46683641], [4.70335513], [5.51074196], [5.77536735], [5.50595722], [4.68637212]],
             [[3.34183827], [4.5430821 ], [5.33049864], [5.58676063], [5.30614662], [4.38580553]],
             [[2.96852351], [4.1349709 ], [4.83801129], [4.96868391], [4.6222409 ], [3.84192146]]]
        )

        with self._suppress_expected_optics_warnings():
            image = sim_im_pip()
            assert np.allclose(image, expected_image, atol=1e-6)
            image = sim_im_pip()
            assert np.allclose(image, expected_image, atol=1e-6)
            image = sim_im_pip.update()()
            assert not np.allclose(image, expected_image, atol=1e-6)

        ## PART 2.4
        np.random.seed(123)  # Note that this seeding is not warratied
        random.seed(123)     # to give reproducible results across
                            # platforms so the subsequent test might fail

        ellipse = dt.Ellipsoid(
            radius=random_ellipse_axes,
            intensity=lambda: np.random.uniform(0.5, 1.5),
            position=lambda: np.random.uniform(2, train_image_size - 2,
                                               size=2),
            rotation=lambda: np.random.uniform(0, 2 * np.pi),
        )
        synthetic_nuclei = (
            (ellipse ^ (lambda: np.random.randint(1, 2)))
            >> dt.Pad(px=(10, 10, 10, 10), keep_size=False)
            >> dt.ElasticTransformation(alpha=100, sigma=10, order=1)
            >> dt.CropTight()
        )

        long_range_noise = (synthetic_nuclei >> dt.Poisson(snr=0.2)
                            >> dt.GaussianBlur(sigma=3.5))
        short_range_noise = (synthetic_nuclei >> dt.Poisson(snr=1.0)
                            >> dt.GaussianBlur(sigma=1.5))
        random_range_noise = (
            synthetic_nuclei
            >> dt.Poisson(snr=lambda: np.random.uniform(0.5, 1.5))
            >> dt.GaussianBlur(sigma=lambda: np.random.uniform(0.75, 1.5))
        )
        noisy_synthetic_nuclei = (
            synthetic_nuclei
            * (long_range_noise + short_range_noise + random_range_noise) / 3
        )

        non_overlap_nuclei = dt.NonOverlapping(
            noisy_synthetic_nuclei, min_distance=6,
        )

        optics = dt.Fluorescence(
            resolution=1e-6, magnification=6, wavelength=400e-9,
            NA=lambda: np.random.uniform(0.9, 1.1),
            output_region=(0, 0, train_image_size, train_image_size),
        )
        sim_im_pip = (
            optics(non_overlap_nuclei)
            >> dt.Gaussian(sigma=lambda: np.random.uniform(0, 0.1))
            >> dt.Divide(lambda: np.random.uniform(14, 20))
            >> dt.Add(lambda: np.random.uniform(-0.05, 0.15))
            >> dt.Clip(0, 1) >> dt.AsType("float")
        )

        # Checks
        expected_image = np.array(
            [[[0.13199702], [0.1420024 ], [0.15640373], [0.15710884], [0.15771862], [0.15338107]],
            [[0.16063779], [0.1707068 ], [0.18537119], [0.19869939], [0.1960437 ], [0.18760834]],
            [[0.18653255], [0.2071835 ], [0.21618341], [0.22117799], [0.21667417], [0.20882602]],
            [[0.20211888], [0.22039408], [0.22713002], [0.2263781 ], [0.2210908 ], [0.21466281]],
            [[0.19227835], [0.21184996], [0.22195321], [0.22250827], [0.21844318], [0.20950961]],
            [[0.16802898], [0.18852521], [0.19970309], [0.19951212], [0.19412736], [0.18247772]]]
        )
        with self._suppress_expected_optics_warnings():
            image = sim_im_pip()
            assert np.allclose(image, expected_image, atol=1e-6)
            image = sim_im_pip()
            assert np.allclose(image, expected_image, atol=1e-6)
            image = sim_im_pip.update()()
            assert not np.allclose(image, expected_image, atol=1e-6)

        if TORCH_AVAILABLE:
            ## PART 2.5
            import warnings

            from skimage import morphology as skmorph

            np.random.seed(123)  # Note that this seeding is not warratied
            random.seed(123)     # to give reproducible results across
                                # platforms so the subsequent test might fail

            def get_mask(radius):
                """Apply isotropic erosion to a binary mask."""
                def inner(mask):
                    mask = np.sum(mask, -1, keepdims=True) > 0
                    mask = np.pad(mask, [(1, 1), (1, 1), (0, 0)],
                                  mode="constant")
                    mask = skmorph.isotropic_erosion(mask, radius=radius)
                    return mask[1:-1, 1:-1]
                return inner

            sim_mask_pip = (
                non_overlap_nuclei
                >> dt.SampleToMasks(
                    get_mask,
                    radius=1,
                    output_region=optics.output_region,
                    merge_method="or",
                )
                >> dt.AsType("float")
            )

            # Checks
            expected_mask = np.array(
                [[[0.], [0.], [0.], [0.], [0.], [0.]],
                [[0.], [1.], [1.], [0.], [0.], [0.]],
                [[1.], [1.], [1.], [1.], [0.], [0.]],
                [[1.], [1.], [1.], [1.], [1.], [1.]],
                [[1.], [1.], [1.], [1.], [1.], [1.]],
                [[1.], [1.], [1.], [1.], [1.], [1.]]]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mask = sim_mask_pip()
                assert np.allclose(mask, expected_mask, atol=1e-6)
                mask = sim_mask_pip()
                assert np.allclose(mask, expected_mask, atol=1e-6)
                mask = sim_mask_pip.update()()
                assert not np.allclose(mask, expected_mask, atol=1e-6)

            ## PART 2.6
            np.random.seed(123)  # Note that this seeding is not warratied
            random.seed(123)     # to give reproducible results across
                                # platforms so the subsequent test might fail

            sim_im_mask_pip = (
                (sim_im_pip & sim_mask_pip)
                >> dt.MoveAxis(2, 0)
                >> dt.pytorch.ToTensor(dtype=torch.float)
            )
            train_dataset = dt.pytorch.Dataset(
                sim_im_mask_pip, length=640, replace=0.01,
            )

            assert len(train_dataset) == 640

            image, mask = train_dataset[639]
            expected_image = torch.tensor(
               [[[0.1320, 0.1420, 0.1564, 0.1571, 0.1577, 0.1534],
                [0.1606, 0.1707, 0.1854, 0.1987, 0.1960, 0.1876],
                [0.1865, 0.2072, 0.2162, 0.2212, 0.2167, 0.2088],
                [0.2021, 0.2204, 0.2271, 0.2264, 0.2211, 0.2147],
                [0.1923, 0.2118, 0.2220, 0.2225, 0.2184, 0.2095],
                [0.1680, 0.1885, 0.1997, 0.1995, 0.1941, 0.1825]]]
            )
            expected_mask = torch.tensor(
                [[[0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1.]]]
            )
            assert torch.allclose(image, expected_image, rtol=1e-7, atol=1e-4)
            assert torch.allclose(mask, expected_mask, rtol=1e-7, atol=1e-4)

    def test_6_1(self):
        if TORCH_AVAILABLE:
            np.random.seed(123)  # Note that this seeding is not warratied
                                 # to give reproducible results across
                                 # platforms so the subsequent test might fail

            image_size = 5

            particle = dt.PointParticle(
                position=lambda: np.random.uniform(
                    image_size / 2 - 1,
                    image_size / 2 + 1,
                    size=2,
                ),
            )

            optics = dt.Fluorescence(
                output_region=(0, 0, image_size, image_size),
            )

            simulation = (
                optics(particle)
                >> dt.NormalizeMinMax()
                >> dt.Gaussian(sigma=0.1)
                >> dt.MoveAxis(-1, 0)
                >> dt.pytorch.ToTensor(dtype=torch.float32)
            )
            
            train_dataset = dt.pytorch.Dataset(simulation, length=2)
            test_dataset = dt.pytorch.Dataset(simulation & particle.position,
                                                length=10)

            # Test train dataset
            expected_image = torch.tensor(
                [[[ 0.0283, -0.0033,  0.1577,  0.3276, -0.2213],
                [ 0.1929,  0.6013,  0.4957,  0.4310,  0.2619],
                [ 0.5937,  0.6918,  0.8498,  0.7436,  0.7122],
                [ 0.7354,  0.9519,  1.0386,  0.9589,  0.7165],
                [ 0.3173,  0.8281,  0.7187,  0.6770,  0.5468]]],
                dtype=torch.float32,
            )
            
            with self._suppress_expected_optics_warnings():
               image = train_dataset[0]
            assert torch.allclose(image[0], expected_image,
                                  rtol=1e-4, atol=1e-4)

            assert len(train_dataset) == 2

            with self._suppress_expected_optics_warnings():
                for image in train_dataset:
                    image = image[0]
                    assert isinstance(image, torch.Tensor)
                    assert image.dtype == torch.float32
                    assert image.shape == torch.Size([1, image_size, image_size])

            # Test test dataset
            expected_image = torch.tensor(
                [[[0.0891, 0.2869, 0.3431, 0.3024, 0.0388],
                [0.2430, 0.3986, 0.3769, 0.6547, 0.4562],
                [0.3438, 0.6046, 0.7049, 0.8493, 0.6863],
                [0.3541, 0.8862, 0.8850, 0.7877, 0.8871],
                [0.3608, 0.7055, 0.8491, 0.7773, 0.8561]]],
                dtype=torch.float32,
            )
            expected_position = torch.tensor([3.2509, 2.5208])
            with self._suppress_expected_optics_warnings():
                image, position = test_dataset[0]
            assert torch.allclose(image, expected_image,
                                  rtol=1e-4, atol=1e-4)
            assert torch.allclose(position, expected_position,
                                  rtol=1e-4, atol=1e-4)

            assert len(test_dataset) == 10
            with self._suppress_expected_optics_warnings():
                for image, position in test_dataset:
                    assert isinstance(image, torch.Tensor)
                    assert image.shape == torch.Size([1, image_size, image_size])
                    assert image.dtype == torch.float32

                    assert isinstance(position, torch.Tensor)
                    assert position.shape == (2,)  # (x, y) particle position

    def test_6_A(self):
        # Temporary root (deleted in finally)
        tmp_root = tempfile.mkdtemp(prefix="cells_like_")
        data_root = Path(tmp_root) / "02"
        image_dir = data_root / "image"
        label_dir  = data_root / "label"

        try:
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            # Synthetic image (grayscale with some blobs)
            image = np.zeros((8, 12), dtype=np.uint8)
            image[1:3, 2:4] = 128   # blob 1
            image[4:6, 6:8] = 200  # blob 2
            image[6:8, 4:6] = 255  # blob 3

            # Synthetic label mask (integer IDs for blobs)
            label = np.zeros_like(image, dtype=np.uint8)
            label[1:3, 2:4] = 1
            label[4:6, 6:8] = 2
            label[6:8, 4:6] = 3

            # Save images
            Image.fromarray(image, mode="L").save(image_dir / "image_0.png")
            for i in range(1, 5):
                Image.fromarray(np.zeros_like(image), mode="L") \
                    .save(image_dir / f"image_{i}.png")

            # Save labels
            Image.fromarray(label, mode="L").save(label_dir / "label_0.png")
            for i in range(1, 5):
                Image.fromarray(np.zeros_like(label), mode="L") \
                    .save(label_dir / f"label_{i}.png")

            ## PART 1
            # Loading and analyzing the image and segmentaions.
            from skimage.measure import regionprops

            sources = dt.sources.Source(
                image_path=sorted(glob.glob(str(image_dir / "*.png"))),
                label_path=sorted(glob.glob(str(label_dir / "*.png"))),
            )

            image_pip = dt.LoadImage(sources.image_path)[1:, 2:-4] / 256
            props_pip = (
                dt.LoadImage(sources.label_path)[1:, 2:-4]
                >> regionprops
            )

            pip = image_pip & props_pip

            image, *props = pip()

            # The combined output should flatten to 1 image + 3 props = 4 items
            assert len(pip()) == 4

            assert isinstance(image, np.ndarray)
            expected_image = np.array(
                [[0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.78125, 0.78125],
                [0.0, 0.0, 0.0, 0.0, 0.78125, 0.78125],
                [0.0, 0.0, 0.99609375, 0.99609375, 0.0, 0.0],
                [0.0, 0.0, 0.99609375, 0.99609375, 0.0, 0.0]],
                dtype=np.float32,
            )
            assert np.allclose(image.squeeze(), expected_image, atol=1e-6)

            assert sorted([p.label for p in props]) == [1, 2, 3]

            ## PART 2
            # Cropping.
            crop_frame_index = 0
            crop_size = 2
            crop_x0 = 1
            crop_y0 = 1

            image, *props = pip(sources[crop_frame_index])
            crop = image[crop_x0:crop_x0 + crop_size,
                         crop_y0:crop_y0 + crop_size]

            expected_crop = np.array(
                [[0.5, 0.0],
                [0.0, 0.0]],
                dtype=np.float32,
            )
            assert np.allclose(crop.squeeze(), expected_crop, atol=1e-6)

            ## PART 3
            # Training pipeline.
            if TORCH_AVAILABLE:
                np.random.seed(123)  # Note that this seeding is not warratied
                                     # to give reproducible results across
                                     # platforms so the subsequent test might
                                     # fail

                train_pip = (
                    dt.Value(crop)
                    >> dt.Multiply(lambda: np.random.uniform(0.9, 1.1))
                    >> dt.Add(lambda: np.random.uniform(-0.1, 0.1))
                    >> dt.MoveAxis(-1, 0)
                    >> dt.pytorch.ToTensor(dtype=torch.float32)
                )

                train_dataset = \
                    dt.pytorch.Dataset(train_pip, length=40, replace=False)

                assert len(train_dataset) == 40

                expected_tensor_0 = torch.tensor(
                    [[[ 0.4769, -0.0428],
                     [-0.0428, -0.0428]]],
                )
                sample = train_dataset[0]
                assert torch.allclose(sample[0], expected_tensor_0,
                                      rtol=1e-4, atol=1e-4)

                expected_tensor_39 = torch.tensor(
                    [[[0.4829, 0.0103],
                     [0.0103, 0.0103]]],
                )
                sample = train_dataset[39]
                assert torch.allclose(sample[0], expected_tensor_39,
                                      rtol=1e-4, atol=1e-4)

        except Exception:
            raise
        finally:
            # Clean up the temporary dataset tree
            shutil.rmtree(tmp_root, ignore_errors=True)

    def test_7_1(self):
        # Small toy dataset
        # Shape: (num_samples=5, seq_len=3, num_features=2)
        in_sequences = np.array([
            [[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]],
            [[4.0, 20.0], [5.0, 21.0], [6.0, 22.0]],
            [[7.0, 30.0], [8.0, 31.0], [9.0, 32.0]],
            [[10.0, 40.0], [11.0, 41.0], [12.0, 42.0]],
            [[13.0, 50.0], [14.0, 51.0], [15.0, 52.0]],
        ])

        # Targets: one value per sample (e.g., "temperature")
        # Shape: (num_samples=5, 1)
        targets = np.array([
            [100.0],
            [200.0],
            [300.0],
            [400.0],
            [500.0],
        ])

        temp_idx = 0

        sources = dt.sources.Source(inputs=in_sequences, targets=targets)
        train_sources, val_sources = \
            dt.sources.random_split(sources, [0.8, 0.2])

        assert len(train_sources) == 4
        assert len(val_sources) == 1

        mean = np.mean([src["inputs"] for src in sources], axis=(0, 1))
        std = np.std([src["inputs"] for src in sources], axis=(0, 1))
        np.testing.assert_allclose(mean, np.array([ 8., 31.]),
                                   rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(std, np.array([ 4.3204938 , 14.16568624]),
                                   rtol=1e-7, atol=1e-7)

        train_mean = np.mean([src["inputs"] for src in train_sources],
                             axis=(0, 1))
        train_std = np.std([src["inputs"] for src in train_sources],
                           axis=(0, 1))

        if TORCH_AVAILABLE:
            inputs_pipeline = (
                dt.Value(sources.inputs - train_mean) / train_std
                >> dt.pytorch.ToTensor(dtype=torch.float)
            )
            targets_pipeline = (
                dt.Value(sources.targets - train_mean[temp_idx])
                / train_std[temp_idx]
            )

            train_dataset = dt.pytorch.Dataset(
                inputs_pipeline & targets_pipeline,
                inputs=train_sources,
            )
            val_dataset = dt.pytorch.Dataset(
                inputs_pipeline & targets_pipeline,
                inputs=val_sources,
            )

            assert len(train_dataset) == 4
            for inputs, target in train_dataset:
                assert isinstance(inputs, torch.Tensor)
                assert isinstance(target, torch.Tensor)
                assert inputs.shape == torch.Size([3, 2])
                assert target.shape == torch.Size([1])
                assert inputs.dtype == torch.float32
                assert target.dtype == torch.float32

            assert len(val_dataset) == 1
            for inputs, target in val_dataset:
                assert inputs.shape == torch.Size([3, 2])
                assert target.shape == torch.Size([1])

    def test_7_A(self):
        if TORCH_AVAILABLE:
            # Hardcoded sequences: shape (num_samples=10, seq_len=4)
            in_sequences = np.array([
                [1, 2, 3, 0],
                [4, 5, 6, 0],
                [7, 8, 9, 0],
                [10, 11, 12, 0],
                [13, 14, 15, 0],
                [16, 17, 18, 0],
                [19, 20, 21, 0],
                [22, 23, 24, 0],
                [25, 26, 27, 0],
                [28, 29, 30, 0],
            ])

            out_sequences = np.array([
                [101, 102, 103, 0],
                [104, 105, 106, 0],
                [107, 108, 109, 0],
                [110, 111, 112, 0],
                [113, 114, 115, 0],
                [116, 117, 118, 0],
                [119, 120, 121, 0],
                [122, 123, 124, 0],
                [125, 126, 127, 0],
                [128, 129, 130, 0],
            ])

            sources = dt.sources.Source(
                inputs=in_sequences,
                targets=out_sequences,
            )
            train_sources, test_sources = \
                dt.sources.random_split(sources, [.8, .2])

            assert len(train_sources) == 8
            assert len(test_sources) == 2

            inputs_pip = (
                dt.Value(sources.inputs)
                >> dt.pytorch.ToTensor(dtype=torch.int)
            )
            outputs_pip = (
                dt.Value(sources.targets)
                >> dt.pytorch.ToTensor(dtype=torch.int)
            )

            train_dataset = dt.pytorch.Dataset(
                inputs_pip & outputs_pip,
                inputs=train_sources,
            )
            test_dataset = dt.pytorch.Dataset(
                inputs_pip & outputs_pip,
                inputs=test_sources,
            )

            assert len(train_dataset) == 8
            for input, output in train_dataset:
                assert isinstance(input, torch.Tensor)
                assert isinstance(output, torch.Tensor)
                assert input.shape == torch.Size([4])
                assert output.shape == torch.Size([4])
                assert input.dtype == torch.int64
                assert output.dtype == torch.int64

            assert len(test_dataset) == 2
            for input, output in test_dataset:
                assert isinstance(input, torch.Tensor)
                assert isinstance(output, torch.Tensor)
                assert input.shape == torch.Size([4])
                assert output.shape == torch.Size([4])
                assert input.dtype == torch.int64
                assert output.dtype == torch.int64

    def test_8_A(self):
        pass  # Essentially same code as test_7_A


if __name__ == "__main__":
    unittest.main()
