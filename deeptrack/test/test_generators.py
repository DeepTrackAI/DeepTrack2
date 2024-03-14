import sys

# sys.path.append(".")  # Adds the module to path

import unittest

has_required_modules = True

try:
    from .. import features
    from .. import generators
    from ..optics import Fluorescence
    from ..scatterers import PointParticle
    from ..models import gnns
    import numpy as np
    import pandas as pd
except ImportError:
    has_required_modules = False

class TestGenerators(unittest.TestCase):
    def test_Generator(self):
        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 128, 128),
        )
        scatterer = PointParticle(
            intensity=100,
            position_unit="pixel",
            position=lambda: np.random.rand(2) * 128,
        )
        imaged_scatterer = optics(scatterer)
        imaged_scatterer.store_properties()

        def get_particle_position(result):
            for property in result.properties:
                if "position" in property:
                    return property["position"]

        generator = generators.Generator()
        particle_generator = generator.generate(
            imaged_scatterer, get_particle_position, batch_size=4
        )
        particles, positions = next(particle_generator)
        for particle, position in zip(particles, positions):
            self.assertEqual(particle.shape, (128, 128, 1))
            self.assertTrue(np.all(position >= 0))
            self.assertTrue(np.all(position <= 128))

    def test_ContinuousGenerator(self):
        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 128, 128),
        )
        scatterer = PointParticle(
            intensity=100,
            position_unit="pixel",
            position=lambda: np.random.rand(2) * 128,
        )
        imaged_scatterer = optics(scatterer)
        imaged_scatterer.store_properties()

        def get_particle_position(result):
            for property in result.properties:
                if "position" in property:
                    return property["position"]

        generator = generators.ContinuousGenerator(
            imaged_scatterer, get_particle_position, min_data_size=10, max_data_size=20
        )

        with generator:
            self.assertGreater(len(generator.data), 10)
            self.assertLess(len(generator.data), 21)
    

    def test_MultiInputs_ContinuousGenerator(self):
        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 128, 128),
        )
        scatterer_A = PointParticle(
            intensity=100,
            position_unit="pixel",
            position=lambda: np.random.rand(2) * 128,
        )
        scatterer_B = PointParticle(
            intensity=10,
            position_unit="pixel",
            position=lambda: np.random.rand(2) * 128,
        )
        imaged_scatterer_A = optics(scatterer_A)
        imaged_scatterer_B = optics(scatterer_B)

        def get_particle_position(result):
            result = result[0]
            for property in result.properties:
                if "position" in property:
                    return property["position"]
        
        imaged_scatterers = imaged_scatterer_A & imaged_scatterer_B
        imaged_scatterers.store_properties()

        generator = generators.ContinuousGenerator(
            imaged_scatterers, 
            get_particle_position,
            batch_size=8, 
            min_data_size=10, 
            max_data_size=20, 
            use_multi_inputs=True,
        )

        with generator:
            data, _ = generator[0]
            self.assertEqual(data[0].shape, (8, 128, 128, 1))
            self.assertEqual(data[1].shape, (8, 128, 128, 1))

    def test_CappedContinuousGenerator(self):

        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 128, 128),
            index=range(0, 200),
        )
        scatterer = PointParticle(
            intensity=100,
            position_unit="pixel",
            position=lambda: np.random.rand(2) * 128,
        )
        imaged_scatterer = optics(scatterer)

        def get_particle_position(result):
            for property in result.properties:
                if "position" in property:
                    return property["position"]

        generator = generators.ContinuousGenerator(
            imaged_scatterer,
            get_particle_position,
            batch_size=1,
            min_data_size=10,
            max_data_size=20,
            max_epochs_per_sample=5,
        )

        # with generator:
        #     self.assertGreater(len(generator.data), 10)
        #     self.assertLess(len(generator.data), 21)
        #     for _ in range(10):
        #         generator.on_epoch_end()
        #         for idx in range(len(generator)):
        #             a = generator[idx]

        #         [self.assertLess(d[-1], 8) for d in generator.data]
    

    def test_GraphGenerator(self):
        frame = np.arange(10)
        centroid = np.random.normal(0.5, 0.1, (10, 2))

        df = pd.DataFrame(
            {
                'frame': frame, 
                'centroid-0': centroid[:, 0], 
                'centroid-1': centroid[:, 1], 
                'label': 0, 
                'set': 0, 
                'solution': 0.0
            }
        )
        # remove consecutive frames
        df = df[~df["frame"].isin([3, 4, 5])]

        generator = gnns.generators.GraphGenerator(
	        nodesdf=df, 
	        properties=["centroid"], 
	        min_data_size=8,
	        max_data_size=9,
	        batch_size=8,
	        feature_function=gnns.augmentations.GetGlobalFeature,
	        radius=0.2,
            nofframes=3,
            output_type="edges"
        )
        self.assertIsInstance(generator, gnns.generators.ContinuousGraphGenerator)



    def test_training(self):
        from deeptrack.extras import datasets
        import tensorflow as tf

        datasets.load("BFC2Cells")
        nodesdf = pd.read_csv("datasets/BFC2DLMuSCTra/nodesdf.csv")

        # normalize centroids between 0 and 1
        nodesdf.loc[:, nodesdf.columns.str.contains("centroid")] = (
            nodesdf.loc[:, nodesdf.columns.str.contains("centroid")]
            / np.array([1000.0, 1000.0])
        )
        parenthood = pd.read_csv("datasets/BFC2DLMuSCTra/parenthood.csv")
        variables = features.DummyFeature(
            radius=0.2,
            output_type="edges",
            nofframes=3, # time window to associate nodes (in frames)
        )
        model = gnns.MAGIK(
            dense_layer_dimensions=(64, 96,),      # number of features in each dense encoder layer
            base_layer_dimensions=(96, 96, 96),    # Latent dimension throughout the message passing layers
            number_of_node_features=2,             # Number of node features in the graphs
            number_of_edge_features=1,             # Number of edge features in the graphs
            number_of_edge_outputs=1,              # Number of predicted features
            edge_output_activation="sigmoid",      # Activation function for the output layer
            output_type="edges",              # Output type. Either "edges", "nodes", or "graph"
        )

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss = 'binary_crossentropy',
            metrics=['accuracy'],
        )

        generator = gnns.generators.GraphGenerator(
                nodesdf=nodesdf,
                properties=["centroid"],
                parenthood=parenthood,
                min_data_size=8,#511,
                max_data_size=9,#512,
                batch_size=8,
                **variables.properties()
            )
        
        with generator:
            model.fit(generator, epochs=1)


if not has_required_modules:
    TestGenerators = None
    del TestGenerators

if __name__ == "__main__":
    unittest.main()
