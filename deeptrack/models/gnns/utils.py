from matplotlib import cm
import numpy as np
import itertools
import pandas as pd

from .graphs import GraphExtractor


def f(d):
    for frame_y in range(int(d["frame_y"].min()), int(d["frame_y"].max()) + 1):
        x = d.loc[d["frame_y"] == frame_y]
        d.loc[
            x.loc[x["score"] < x["score"].max(), "prediction"].index,
            "prediction",
        ] = False
    return d


def to_trajectories(
    edges_df: pd.DataFrame,
    assert_no_splits=True,
    _type="prediction",
):

    edges_df["frame_diff"] = edges_df["frame_y"] - edges_df["frame_x"]
    if assert_no_splits and _type != "gt":
        edges_df_grouped = edges_df.groupby(["node_x"])
        edges_df = edges_df_grouped.apply(f)
    edges_df["frame_diff"] = edges_df["frame_y"] - edges_df["frame_x"]
    edges_df_grouped = edges_df.groupby(["node_y"])
    edges_dfs = [_edge_df for _, _edge_df in edges_df_grouped]

    color = cm.viridis(np.linspace(0, 1, 250))
    color = itertools.cycle(color)
    t = {}

    x_iterator = iter(range(10000000))
    tr = {}
    sc = {}
    alpha = 0.1
    parents = []
    d_alpha = (1 - alpha) / edges_df["prediction"].max()
    trajectories = []
    for _edge_df in edges_dfs:
        solutions = _edge_df.loc[_edge_df[_type] == 1.0, :]
        solutions = solutions.loc[
            solutions["frame_diff"] == solutions["frame_diff"].min(), :
        ]
        if _type == "prediction":
            solutions = solutions.loc[
                solutions["score"] == solutions["score"].max(), :
            ]
        for i in range(len(solutions)):
            alpha = 0.1 + d_alpha * int(solutions[i : i + 1]["frame_y"])

            if not (str(int(solutions[i : i + 1]["node_x"])) in t.keys()):
                t[str(int(solutions[i : i + 1]["node_x"]))] = next(color)
                tr[int(solutions[i : i + 1]["node_x"])] = next(x_iterator)
                sc[int(solutions[i : i + 1]["node_x"])] = float(
                    solutions[i : i + 1]["score"]
                )

            if int(solutions[i : i + 1]["node_x"]) in parents:
                t[str(int(solutions[i : i + 1]["node_x"]))] = next(color)
                tr[int(solutions[i : i + 1]["node_x"])] = next(x_iterator)
                sc[int(solutions[i : i + 1]["node_x"])] = float(
                    solutions[i : i + 1]["score"]
                )
                break

            t[str(int(solutions[i : i + 1]["node_y"]))] = t[
                str(int(solutions[i : i + 1]["node_x"]))
            ]

            key = int(solutions[i : i + 1]["node_y"])
            tr[key] = tr[int(solutions[i : i + 1]["node_x"])]
            parents.append(int(solutions[i : i + 1]["node_x"]))

    traj_dict = {}

    for key, val in tr.items():
        if val not in traj_dict:
            traj_dict[val] = []
        traj_dict[val].append(key)

    for val in traj_dict.values():
        trajectories.append(val)

    return trajectories


def get_predictions(dfs, properties, model, graph_kwargs):
    """
    Get predictions from nodes dataframe.
    Parameters
    ----------
    dfs: DataFrame
        DataFrame containing the nodes.
    properties: dict
        Dictionary containing names of the properties to be used as features.
    model: tf.keras.Model
        Model to be used for predictions.
    graph_kwargs: dict
        Extra arguments to be passed to the graph extractor.
    """
    grapht = GraphExtractor(
        nodesdf=dfs,
        properties=properties,
        validation=True,
        **graph_kwargs.properties()
    )

    v = [
        np.expand_dims(grapht[0][0][:, 1:], 0),
        np.expand_dims(grapht[0][1], 0),
        np.expand_dims(grapht[0][2][:, 2:], 0),
        np.expand_dims(grapht[0][3], 0),
    ]
    output_edge_f = model(v).numpy()
    pred = (output_edge_f > 0.5)[0, ...]
    g = grapht[1][1]

    return pred, g, output_edge_f, grapht


def df_from_results(pred, g, output_edge_features, graph_test):
    nodes = graph_test[0][0]
    edges = graph_test[0][2]

    edges = np.append(edges, np.expand_dims(g, axis=-1), axis=-1)
    edges = np.append(edges, pred, axis=-1)

    frame_edges = edges[edges[:, 1] <= np.max(nodes[:, 0])]

    edges_df = pd.DataFrame(
        frame_edges,
        columns=["frame_x", "frame_y", "node_x", "node_y", "gt", "prediction"],
    )
    edges_df["score"] = output_edge_features[0]

    edges_df["frame_diff"] = edges_df["frame_y"] - edges_df["frame_x"]

    return edges_df, nodes, edges


def get_traj(edges_df, th=5):
    traj = []
    traj = to_trajectories(edges_df)
    traj = list(filter(lambda t: len(t) > th, traj))
    traj = [sorted(t) for t in traj]

    num_traj = len(traj)

    top = cm.get_cmap("Oranges_r")
    bottom = cm.get_cmap("Blues_r")

    colors = np.vstack(
        (
            top(np.linspace(0, 1, int(np.ceil(num_traj / 2)))),
            bottom(np.linspace(0, 1, int(np.ceil(num_traj / 2)))),
        )
    )
    np.random.shuffle(colors)

    traj = [[t, colors[i]] for i, t in enumerate(traj)]

    return traj
