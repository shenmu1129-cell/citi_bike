from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .regional_common import PATHS, ensure_dirs, haversine_m, log


def normalized_adjacency(adj: np.ndarray) -> np.ndarray:
    degrees = adj.sum(axis=1)
    degrees = np.where(degrees <= 0, 1.0, degrees)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    return d_inv_sqrt @ adj @ d_inv_sqrt


def build_region_graph(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    region_info = pd.read_csv(PATHS["tables"] / "region_grid_info.csv")
    region_info = region_info.sort_values("grid_id").reset_index(drop=True)
    n = len(region_info)
    k = min(int(config.get("knn_k", 5)), max(n - 1, 1))
    od_k = min(int(config.get("od_k", 5)), max(n - 1, 1))
    sigma = float(config.get("sigma_m", 1500.0))
    distance_graph_weight = float(config.get("distance_graph_weight", 0.55))
    od_graph_weight = float(config.get("od_graph_weight", 0.45))
    lat = region_info["grid_center_lat"].to_numpy()
    lng = region_info["grid_center_lng"].to_numpy()
    dist = np.zeros((n, n), dtype="float64")
    for i in range(n):
        dist[i] = haversine_m(lat[i], lng[i], lat, lng)

    distance_adj = np.eye(n, dtype="float64")
    edge_rows = []
    for i in range(n):
        nearest = np.argsort(dist[i])[1 : k + 1]
        for j in nearest:
            weight = float(np.exp(-dist[i, j] / sigma))
            distance_adj[i, j] = max(distance_adj[i, j], weight)
            distance_adj[j, i] = max(distance_adj[j, i], weight)
            edge_rows.append(
                {
                    "source_grid_id": region_info.loc[i, "grid_id"],
                    "target_grid_id": region_info.loc[j, "grid_id"],
                    "distance_m": float(dist[i, j]),
                    "distance_weight": weight,
                    "od_weight": 0.0,
                    "combined_weight": weight,
                    "edge_type": "distance_knn",
                }
            )

    od_adj = np.eye(n, dtype="float64")
    od_path = PATHS["tables"] / "region_od_edges.csv"
    if od_path.exists():
        od = pd.read_csv(od_path)
        id_to_idx = {grid_id: idx for idx, grid_id in enumerate(region_info["grid_id"])}
        od = od[od["source_grid_id"].isin(id_to_idx) & od["target_grid_id"].isin(id_to_idx)].copy()
        if not od.empty:
            max_count = max(float(od["trip_count"].max()), 1.0)
            od["od_weight"] = np.log1p(od["trip_count"]) / np.log1p(max_count)
            for source, group in od.groupby("source_grid_id"):
                group = group[group["target_grid_id"] != source].sort_values("od_weight", ascending=False).head(od_k)
                i = id_to_idx[source]
                for row in group.itertuples(index=False):
                    j = id_to_idx[row.target_grid_id]
                    weight = float(row.od_weight)
                    od_adj[i, j] = max(od_adj[i, j], weight)
                    od_adj[j, i] = max(od_adj[j, i], weight)
                    edge_rows.append(
                        {
                            "source_grid_id": source,
                            "target_grid_id": row.target_grid_id,
                            "distance_m": float(dist[i, j]),
                            "distance_weight": float(np.exp(-dist[i, j] / sigma)),
                            "od_weight": weight,
                            "combined_weight": weight,
                            "edge_type": "od_flow",
                        }
                    )

    adj = distance_graph_weight * distance_adj + od_graph_weight * od_adj
    np.fill_diagonal(adj, 1.0)
    adj_norm = normalized_adjacency(adj)
    adj_df = pd.DataFrame(adj_norm, index=region_info["grid_id"], columns=region_info["grid_id"])
    edges = pd.DataFrame(edge_rows)
    if not edges.empty:
        edges = (
            edges.groupby(["source_grid_id", "target_grid_id"], as_index=False)
            .agg(
                distance_m=("distance_m", "min"),
                distance_weight=("distance_weight", "max"),
                od_weight=("od_weight", "max"),
                combined_weight=("combined_weight", "max"),
                edge_type=("edge_type", lambda s: "+".join(sorted(set(s)))),
            )
        )
    od_matrix = pd.DataFrame(od_adj, index=region_info["grid_id"], columns=region_info["grid_id"])
    adj_df.to_csv(PATHS["tables"] / "region_adjacency_matrix.csv")
    od_matrix.to_csv(PATHS["tables"] / "region_od_matrix.csv")
    edges.to_csv(PATHS["tables"] / "region_edges.csv", index=False)
    log(f"saved hybrid region graph: nodes={n}, edges={len(edges)}")
    return adj_df, edges
