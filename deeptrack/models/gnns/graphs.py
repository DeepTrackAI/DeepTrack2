import numpy as np
import pandas as pd
import itertools

import tqdm


import more_itertools as mit
from operator import is_not
from functools import partial


def GetEdge(
    df: pd.DataFrame,
    start: int,
    end: int,
    radius: int,
    parenthood: pd.DataFrame,
    **kwargs,
):
    """
    Extracts the edges from a windowed sequence of frames
    Parameters
    ----------
    df: pd.DataFrame
        A dataframe containing the extracted node properties.
    start: int
        Start frame of the edge.
    end: int
        End frame of the edge.
    radius: int
        Search radius for the edge (pixel units).
    parenthood: list
        A list of parent-child relationships
        between nodes.
    Returns
    -------
    edges: pd.DataFrame
        A dataframe containing the extracted
        properties of the edges.
    """
    # Add a column for the indexes of the nodes
    df.loc[:, "index"] = df.index

    # Filter the dataframe to only include the the
    # frames, centroids, labels, and indexes
    df = df.loc[(df["frame"] >= start) & (df["frame"] <= end)].filter(
        regex="(frame|centroid|label|index)"
    )

    # Merge columns contaning the centroids into a single column of
    # numpy arrays, i.e., centroid = [centroid_x, centroid_y,...]
    df.loc[:, "centroid"] = df.filter(like="centroid").apply(np.array, axis=1)

    # Add key column to the dataframe
    df["key"] = 1

    # Group the dataframe by frame
    df = df.groupby(["frame"])
    dfs = [_df for _, _df in df]

    edges = []
    for dfi, dfj in itertools.product(dfs[0:1], dfs[1:]):

        # Merge the dataframes for frames i and j
        combdf = pd.merge(dfi, dfj, on="key").drop("key", axis=1)

        # Compute distances between centroids
        combdf.loc[:, "diff"] = combdf.centroid_x - combdf.centroid_y
        combdf.loc[:, "feature-dist"] = combdf["diff"].apply(
            lambda diff: np.linalg.norm(diff, ord=2)
        )

        # Filter out edges with a feature-distance less than scale * radius
        combdf = combdf[combdf["feature-dist"] < radius].filter(
            regex=("frame|label|index|feature")
        )
        edges.append(combdf)
    # Concatenate the dataframes in a single
    # dataframe for the whole set of edges
    edgedf = pd.concat(edges)

    # Merge columns contaning the labels into a single column
    # of numpy arrays, i.e., label = [label_x, label_y]
    edgedf.loc[:, "label"] = edgedf.filter(like="label").apply(
        np.array, axis=1
    )

    # Returns a solution for each edge. If label is the parenthood
    # array or if label_x == label_y, the solution is 1, otherwise
    # it is 0 representing edges are not connected.
    def GetSolution(x):
        if np.any(np.all(x["label"][::-1] == parenthood, axis=1)):
            solution = 1.0
        elif x["label_x"] == x["label_y"]:
            solution = 1.0
        else:
            solution = 0.0

        return solution

    # Initialize solution column
    edgedf["solution"] = 0.0

    return AppendSolution(edgedf, GetSolution)


def EdgeExtractor(nodesdf, nofframes=3, **kwargs):
    """
    Extracts edges from a sequence of frames
    Parameters
    ----------
    nodesdf: pd.DataFrame
        A dataframe containing the extracted node properties.
    noframes: int
        Number of frames to be used for
        the edge extraction.
    """
    # Create a copy of the dataframe to avoid overwriting
    df = nodesdf.copy()

    edgedfs = []
    sets = np.unique(df["set"])
    for setid in tqdm.tqdm(sets):
        df_set = df.loc[df["set"] == setid].copy()

        # Create subsets from the frame list, with
        # "nofframes" elements each
        maxframe = range(0, df_set["frame"].max() + 1 + nofframes)
        windows = mit.windowed(maxframe, n=nofframes, step=1)
        windows = map(
            lambda x: list(filter(partial(is_not, None), x)), windows
        )
        windows = list(windows)[:-2]

        for window in windows:
            # remove excess frames
            window = [elem for elem in window if elem <= df_set["frame"].max()]

            # Compute the edges for each frames window
            edgedf = GetEdge(df_set, start=window[0], end=window[-1], **kwargs)
            edgedf["set"] = setid
            edgedfs.append(edgedf)

    # Concatenate the dataframes in a single
    # dataframe for the whole set of edges
    edgesdfs = pd.concat(edgedfs)

    return edgesdfs


def AppendSolution(df, func, **kwargs):
    """
    Appends a solution to the dataframe
    Parameters
    ----------
    df: pd.DataFrame
        A dataframe containing the extracted
        properties for the edges/nodes.
    func: function
        A function that takes a dataframe
        and returns a solution.
    Returns
    -------
    df: pd.DataFrame
        A dataframe containing the extracted
        properties for the edges/nodes with
        a solution.
    """
    # Get solution
    df.loc[:, "solution"] = df.apply(lambda x: func(x), axis=1, **kwargs)

    return df


def DataframeSplitter(df, props: tuple, to_array=True, **kwargs):
    """
    Splits a dataframe into features and labels
    Parameters
    ----------
    dt: pd.DataFrame
        A dataframe containing the extracted
        properties for the edges/nodes.
    atts: list
        A list of attributes to be used as features.
    to_array: bool
        If True, the features are converted to numpy arrays.
    Returns
    -------
    X: np.ndarray or pd.DataFrame
        Features.
    """
    # Extract features from the dataframe
    if len(props) == 1:
        features = df.filter(like=props[0])
    else:
        regex = ""
        for prop in props[0:]:
            regex += prop + "|"
        regex = regex[:-1]
        features = df.filter(regex=regex)

    # Extract labels from the dataframe
    label = df["solution"]

    if "index_x" in df:
        outputs = [features, df.filter(like="index"), label]
    else:
        outputs = [features, label]

    # Convert features to numpy arrays if needed
    if to_array:
        outputs = list(
            map(
                lambda x: np.stack(x.apply(np.array, axis=1).values),
                outputs[:-1],
            )
        ) + [
            np.stack(outputs[-1].values),
        ]

    return outputs


def GraphExtractor(
    nodesdf: pd.DataFrame = None,
    properties: list = None,
    parenthood: np.array = np.ones((1, 2)) * -1,
    validation: bool = False,
    global_property: np.array = None,
    **kwargs,
):
    """
    Extracts the graph from a sequence of frames
    Parameters
    ----------
    sequence: dt.Feature
        A sequence of frames.
    """

    # Extract edges and edge features from nodes
    print("Creating graph edges...")
    edgesdf = EdgeExtractor(nodesdf, parenthood=parenthood, **kwargs)

    # Split the nodes dataframe into features and labels
    nodefeatures, nfsolution = DataframeSplitter(
        nodesdf, props=properties, **kwargs
    )

    # Split the edges dataframe into features, sparse adjacency
    # matrix, and labels
    edgefeatures, sparseadjmtx, efsolution = DataframeSplitter(
        edgesdf, props=("feature",), **kwargs
    )

    if validation:
        # Add frames to the adjacency matrix
        frames = edgesdf.filter(like="frame").to_numpy()
        sparseadjmtx = np.concatenate((frames, sparseadjmtx), axis=-1)

        # Add frames to the node features matrix
        frames = nodesdf.filter(like="frame").to_numpy()
        nodefeatures = np.concatenate((frames, nodefeatures), axis=-1)

    # Create edge weights matrix
    edgeweights = np.ones(sparseadjmtx.shape[0])
    edgeweights = np.stack(
        (np.arange(0, edgeweights.shape[0]), edgeweights), axis=1
    )

    # Extract set ids
    nodesets = nodesdf[["set", "label"]].to_numpy()
    edgesets = edgesdf[["set", "label_x", "label_y"]].to_numpy()

    if global_property is None:
        global_property = np.zeros(np.unique(nodesdf["set"]).shape[0])

    return (
        (nodefeatures, edgefeatures, sparseadjmtx, edgeweights),
        (nfsolution, efsolution, global_property),
        (nodesets, edgesets),
    )
