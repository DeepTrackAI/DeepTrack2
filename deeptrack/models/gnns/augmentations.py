import numpy as np
import deeptrack as dt
import random


def GetSubSet(randset):
    """
    Returns a function that takes a graph and returns a
    random subset of the graph.
    """

    def inner(data):
        graph, labels, sets = data

        nodeidxs = np.where(sets[0][:, 0] == randset)[0]
        edgeidxs = np.where(sets[1][:, 0] == randset)[0]

        node_features = graph[0][nodeidxs]
        edge_features = graph[1][edgeidxs]
        edge_connections = graph[2][edgeidxs]

        weights = graph[3][edgeidxs]

        node_labels = labels[0][nodeidxs]
        edge_labels = labels[1][edgeidxs]
        glob_labels = labels[2][randset]

        node_sets = sets[0][nodeidxs]
        edge_sets = sets[1][edgeidxs]

        return (node_features, edge_features, edge_connections, weights), (
            node_labels,
            edge_labels,
            glob_labels,
        ), (node_sets, edge_sets)

    return inner


def GetSubGraphFromLabel(samples):
    """
    Returns a function that takes a graph and returns a subgraph
    of the graph based on the trajectory labels.
    """

    def inner(data):
        graph, labels, (node_sets, edge_sets) = data

        node_idxs = np.in1d(node_sets[:, 1], samples)
        edge_idxs = np.logical_and.reduce(
            np.isin(edge_sets[:, 1:], samples), axis=1
        )

        node_features = graph[0][node_idxs]
        edge_features, edge_connections = (
            graph[1][edge_idxs],
            graph[2][edge_idxs],
        )

        unique = np.unique(edge_connections)
        for i, u in enumerate(unique):
            edge_connections[np.where(edge_connections == u)] = i

        weights = graph[3][edge_idxs]

        node_labels = labels[0][node_idxs]
        edge_labels = labels[1][edge_idxs]
        glob_labels = labels[2]

        return (node_features, edge_features, edge_connections, weights), (
            node_labels,
            edge_labels,
            glob_labels,
        )

    return inner


def NoisyNode(num_centroids=2, **kwargs):
    """
    Returns a function that takes a graph and adds noise to the
    node features except for the centroids.
    """

    def inner(data):
        graph, labels, *_ = data

        features = graph[0][:, num_centroids:]
        features += np.random.randn(*features.shape) * np.random.rand() * 0.1

        node_features = np.array(graph[0])
        node_features[:, 2:] = features

        return (node_features, *graph[1:]), labels

    return inner


def NodeDropout(dropout_rate=0.02, **kwargs):
    """
    Returns a function that takes a graph and drops nodes
    with a certain probability.
    """

    def inner(data):
        graph, labels, *_ = data

        # Get indexes of randomly dropped nodes
        idxs = np.array(list(range(len(graph[0]))))
        dropped_idxs = idxs[np.random.rand(len(graph[0])) < dropout_rate]

        node_f, edge_f, edge_adj, weights = graph
        node_labels, edge_labels, glob_labels = labels

        for dropped_node in dropped_idxs:

            # Find all edges connecting to the dropped node
            edge_connects_removed_node = np.any(
                edge_f == dropped_node, axis=-1
            )

            # Remove bad edges
            edge_f = edge_f[~edge_connects_removed_node]
            edge_adj = edge_adj[~edge_connects_removed_node]
            edge_labels = edge_labels[~edge_connects_removed_node]
            weights = weights[~edge_connects_removed_node]

        return (node_f, edge_f, edge_adj, weights), (
            node_labels,
            edge_labels,
            glob_labels,
        )

    return inner


def AugmentCentroids(rotate, translate, flip_x, flip_y):
    """
    Returns a function that takes a graph and augments the centroids
    by applying a random rotation, translation, and flip.
    """

    def inner(data):
        graph, labels, *_ = data

        centroids = graph[0][:, :2]

        centroids = centroids - 0.5
        centroids_x = (
            centroids[:, 0] * np.cos(rotate)
            + centroids[:, 1] * np.sin(rotate)
            + translate[0]
        )
        centroids_y = (
            centroids[:, 1] * np.cos(rotate)
            - centroids[:, 0] * np.sin(rotate)
            + translate[1]
        )
        if flip_x:
            centroids_x *= -1
        if flip_y:
            centroids_y *= -1

        node_features = np.array(graph[0])
        node_features[:, 0] = centroids_x + 0.5
        node_features[:, 1] = centroids_y + 0.5

        return (node_features, *graph[1:]), labels

    return inner


def GetFeature(full_graph, **kwargs):
    return (
        dt.Value(full_graph)
        >> dt.Lambda(
            GetSubSet,
            randset=lambda: np.random.randint(
                np.max(full_graph[-1][0][:, 0]) + 1),
        )
        >> dt.Lambda(
            GetSubGraphFromLabel,
            samples=lambda: np.array(
                sorted(
                    random.sample(
                        list(full_graph[-1][0][:, -1]),
                        np.random.randint(5, 12),
                    )
                )
            ),
        )
        >> dt.Lambda(
            AugmentCentroids,
            rotate=lambda: np.random.rand() * 2 * np.pi,
            translate=lambda: np.random.randn(2) * 0.05,
            flip_x=lambda: np.random.randint(2),
            flip_y=lambda: np.random.randint(2),
        )
        >> dt.Lambda(NoisyNode)
        >> dt.Lambda(NodeDropout, dropout_rate=0.03)
    )


def GetGlobalFeature(full_graph, **kwargs):
    return (
        dt.Value(full_graph)
        >> dt.Lambda(
            GetSubSet,
            randset=lambda: np.random.randint(
                np.max(full_graph[-1][0][:, 0]) + 1),
        )
        >> dt.Lambda(
            AugmentCentroids,
            rotate=lambda: np.random.rand() * 2 * np.pi,
            translate=lambda: np.random.randn(2) * 0.05,
            flip_x=lambda: np.random.randint(2),
            flip_y=lambda: np.random.randint(2),
        )
    )
