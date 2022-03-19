import numpy as np
import pandas as pd

from .graphs import GraphExtractor

from deeptrack.generators import ContinuousGenerator
from deeptrack.utils import safe_call

from . import augmentations


def GraphGenerator(
    nodesdf: pd.DataFrame = None,
    properties: list = None,
    parenthood: np.array = np.ones((1, 2)) * -1,
    validation_mode: bool = False,
    min_data_size=1000,
    max_data_size=2000,
    feature_function=augmentations.GetFeature,
    **kwargs,
):
    """
    Returns a generator that generates graphs asynchronously.
    Parameters
    ----------
    nodesdf: pd.DataFrame
        Dataframe with containing the node properties.
    properties: list
        List of properties to be extracted from the nodesdf.
    parenthood: np.array (Optional)
        Array containing the parent-child relationships if any.
    validation_mode: bool (Optional)
        Specifies if the generator should be used for validation or not.
    min_data_size : int
        Minimum size of the training data before training starts
    max_data_size : int
        Maximum size of the training data before old data is replaced.
    kwargs : dict
        Keyword arguments to pass to the features.
    """
    full_graph = GraphExtractor(
        nodesdf=nodesdf,
        properties=properties,
        validation=validation_mode,
        parenthood=parenthood,
        **kwargs,
    )

    feature = feature_function(full_graph, **kwargs)

    args = {
        "batch_function": lambda graph: graph[0],
        "label_function": lambda graph: graph[1],
        "min_data_size": min_data_size,
        "max_data_size": max_data_size,
        **kwargs,
    }

    return ContinuousGraphGenerator(feature, **args)


def SelfDuplicateEdgeAugmentation(edges, w, maxnofedges=None, idxs=None):
    """
    Augments edges by randomly adding edges to the graph. The new edges
    are copies of the original edges, and their influence is set to 0.
    Parameters
    ----------
    edges : list of numpy arrays
        List of edges to augment
    maxnofedges : int, optional
        Maximum number of edges to add to the graph. If None, the maximum
        number of edges is set to the number of edges in the graph.
    idxs : list of numpy arrays, optional
        List of indices of the edges to be augmented. If None, the edges
        are selected randomly.
    """
    weights = []
    use_idxs = True if idxs else False
    idxs = idxs if use_idxs else []

    def inner(items):
        itr, (edge, w) = items
        edge = np.array(edge)
        w = np.array(w, dtype=np.float64)

        # Computes the number of additional edges to add
        nofedges = np.shape(edge)[0]
        offset = maxnofedges - nofedges

        # Randomly selects edges to duplicate
        if use_idxs:
            idx = idxs[itr]
        else:
            idx = np.random.choice(nofedges, offset, replace=True)

            idxs.append(idx)

        # Augment the weights and balances repeated edges
        w = np.concatenate((w, np.array([w[idx, 0], 0 * w[idx, 1]]).T), axis=0)
        weights.append(w)

        # Duplicate the edges
        edge = np.concatenate((edge, edge[idx]), axis=0)
        return edge

    return list(map(inner, enumerate(zip(edges, w)))), weights, idxs


class ContinuousGraphGenerator(ContinuousGenerator):
    """
    Generator that asynchronously generates graph representations.
    The generator aims to speed up the training of networks by striking a
    balance between the generalization gained by generating new images and
    the speed gained from reusing images. The generator will continuously
    create new trainingdata during training, until `max_data_size` is reached,
    at which point the oldest data point is replaced.
    Parameters
    ----------
    feature : dt.Feature
        The feature to resolve the graphs from.
    label_function : Callable
        Function that returns the label corresponding to a feature output.
    batch_function : Callable
        Function that returns the training data corresponding a feature output.
    min_data_size : int
        Minimum size of the training data before training starts
    max_data_set : int
        Maximum size of the training data before old data is replaced.
    batch_size : int or Callable[int, int] -> int
        Number of images per batch. A function is expected to accept the current epoch
        and the size of the training data as input.
    shuffle_batch : bool
        If True, the batches are shuffled before outputting.
    feature_kwargs : dict or list of dicts
        Set of options to pass to the feature when resolving
    ndim : int
        Number of dimensions of each batch (including the batch dimension).
    output_type : str
        Type of output. Either "nodes", "edges", or "graph". If 'key' is not a
        supported output type, then the output will be the concatenation of the
        node and edge labels.
    """

    def __init__(self, feature, *args, output_type="graph", **kwargs):
        self.output_type = output_type

        safe_call(super().__init__, positional_args=[feature, *args], **kwargs)

    def __getitem__(self, idx):
        batch, labels = super().__getitem__(idx)

        # Extracts minimum number of nodes in the batch
        cropNodesTo = np.min(
            list(map(lambda _batch: np.shape(_batch[0])[0], batch))
        )

        inputs = [[], [], [], []]
        outputs = [[], [], []]
        nofedges = []

        batch_size = 0
        for i in range(len(batch)):

            # Clip node features to the minimum number of nodes
            # in the batch
            nodef = batch[i][0][:cropNodesTo, :]

            last_node_idx = 0
            # Extracts index of the last node in the adjacency matrix
            try:
                last_node_idx = int(
                    np.where(batch[i][2][:, 1] <= cropNodesTo - 1)[0][-1] + 1
                )
            except IndexError:
                continue

            # Clips edge features and adjacency matrix to the index
            # of the last node
            edgef = batch[i][1][:last_node_idx]
            adjmx = batch[i][2][:last_node_idx]
            wghts = batch[i][3][:last_node_idx]

            # Clips node and edge solutions
            nodesol = labels[i][0][:cropNodesTo]
            edgesol = labels[i][1][:last_node_idx]
            globsol = labels[i][2].astype(np.float)

            inputs[0].append(nodef)
            inputs[1].append(edgef)
            inputs[2].append(adjmx)
            inputs[3].append(wghts)

            nofedges.append(np.shape(edgef)[0])

            outputs[0].append(nodesol)
            outputs[1].append(edgesol)
            outputs[2].append(globsol)

            batch_size += 1

        if batch_size == 0:
            return self.__getitem__((i + 1) % len(self))

        maxnOfedges = np.max(nofedges)

        # Edge augmentation
        inputs[1], weights, idxs = SelfDuplicateEdgeAugmentation(
            inputs[1], inputs[3], maxnofedges=maxnOfedges
        )
        inputs[2], *_ = SelfDuplicateEdgeAugmentation(
            inputs[2], inputs[3], maxnofedges=maxnOfedges, idxs=idxs
        )

        outputs[1], *_ = SelfDuplicateEdgeAugmentation(
            outputs[1], inputs[3], maxnofedges=maxnOfedges, idxs=idxs
        )
        inputs[3] = weights

        # Converts to numpy arrays
        inputs = tuple(map(np.array, inputs))
        outputs = tuple(map(np.array, outputs))

        output_dict = {
            "nodes": outputs[0],
            "edges": outputs[1],
            "graph": [outputs[0], outputs[1]],
            "global": outputs[2],
        }
        try:
            outputs = output_dict[self.output_type]
        except KeyError:
            outputs = output_dict["graph"]

        return inputs, outputs
