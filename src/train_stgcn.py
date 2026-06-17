from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .regional_common import PATHS, ensure_dirs, log, regression_metrics


NODE_FEATURES = [
    "pickup_count",
    "dropoff_count",
    "net_flow",
    "pickup_member_count",
    "pickup_casual_count",
    "pickup_electric_count",
    "pickup_classic_count",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "is_rush_hour",
    "pickup_lag_1",
    "pickup_lag_24",
    "pickup_lag_168",
    "dropoff_lag_1",
    "dropoff_lag_24",
    "dropoff_lag_168",
    "net_flow_lag_1",
    "net_flow_lag_24",
    "net_flow_lag_168",
    "pickup_rolling_mean_24",
    "dropoff_rolling_mean_24",
    "net_flow_rolling_mean_24",
    "net_flow_rolling_std_24",
]


class GatedTemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, nodes, features]
        x = x.permute(0, 3, 2, 1)  # [batch, features, nodes, time]
        x = self.conv(x)
        value, gate = torch.chunk(x, 2, dim=1)
        x = value * torch.sigmoid(gate)
        return x.permute(0, 3, 2, 1)  # [batch, time, nodes, channels]


class GraphConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.linear = nn.Linear(channels, channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # A_hat @ X @ W
        x = torch.einsum("ij,btjf->btif", adj, x)
        return self.act(self.linear(x))


class STConvBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.15):
        super().__init__()
        self.temporal1 = GatedTemporalConv(in_channels, hidden_channels)
        self.graph = GraphConv(hidden_channels)
        self.temporal2 = GatedTemporalConv(hidden_channels, hidden_channels)
        self.residual = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity()
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.temporal1(x)
        out = self.graph(out, adj)
        out = self.temporal2(out)
        out = self.dropout(out)
        return self.norm(out + residual)


class FullSTGCN(nn.Module):
    def __init__(self, num_features: int, hidden_channels: int, dropout: float = 0.15):
        super().__init__()
        self.block1 = STConvBlock(num_features, hidden_channels, dropout)
        self.block2 = STConvBlock(hidden_channels, hidden_channels, dropout)
        self.readout = GatedTemporalConv(hidden_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.block1(x, adj)
        x = self.block2(x, adj)
        x = self.norm(self.readout(x))
        x = x[:, -1, :, :]
        return self.fc(x).squeeze(-1)


@dataclass
class TensorBundle:
    X: np.ndarray
    y: np.ndarray
    y_scaled: np.ndarray
    y_baseline: np.ndarray
    datetimes: np.ndarray
    grid_ids: List[str]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


def add_stgcn_node_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["grid_id", "datetime"]).reset_index(drop=True)
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    for lag in [1, 24, 168]:
        df[f"pickup_lag_{lag}"] = df.groupby("grid_id")["pickup_count"].shift(lag)
        df[f"dropoff_lag_{lag}"] = df.groupby("grid_id")["dropoff_count"].shift(lag)
        df[f"net_flow_lag_{lag}"] = df.groupby("grid_id")["net_flow"].shift(lag)
    df["pickup_rolling_mean_24"] = df.groupby("grid_id")["pickup_count"].transform(lambda s: s.shift(1).rolling(24).mean())
    df["dropoff_rolling_mean_24"] = df.groupby("grid_id")["dropoff_count"].transform(lambda s: s.shift(1).rolling(24).mean())
    df["net_flow_rolling_mean_24"] = df.groupby("grid_id")["net_flow"].transform(lambda s: s.shift(1).rolling(24).mean())
    df["net_flow_rolling_std_24"] = df.groupby("grid_id")["net_flow"].transform(lambda s: s.shift(1).rolling(24).std())
    for feature in NODE_FEATURES:
        if feature not in df.columns:
            df[feature] = 0
    df[NODE_FEATURES] = df[NODE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def build_stgcn_tensors(config: Dict) -> TensorBundle:
    panel = pd.read_csv(PATHS["processed"] / "regional_hourly_panel.csv")
    panel = add_stgcn_node_features(panel)
    region_info = pd.read_csv(PATHS["tables"] / "region_grid_info.csv").sort_values("grid_id")
    grid_ids = region_info["grid_id"].tolist()
    times = np.array(sorted(pd.to_datetime(panel["datetime"]).unique()))
    feature_arrays = []
    for feature in NODE_FEATURES:
        pivot = (
            panel.pivot_table(index="datetime", columns="grid_id", values=feature, aggfunc="first")
            .reindex(index=times, columns=grid_ids)
            .fillna(0)
        )
        feature_arrays.append(pivot.to_numpy(dtype="float32"))
    values = np.stack(feature_arrays, axis=-1)  # [time, nodes, features]
    target = (
        panel.pivot_table(index="datetime", columns="grid_id", values="net_flow", aggfunc="first")
        .reindex(index=times, columns=grid_ids)
        .fillna(0)
        .to_numpy(dtype="float32")
    )
    lookback = int(config["lookback"])
    X_parts = []
    y_parts = []
    y_baseline_parts = []
    y_times = []
    for end_idx in range(lookback - 1, len(times) - 1):
        X_parts.append(values[end_idx - lookback + 1 : end_idx + 1])
        y_parts.append(target[end_idx + 1])
        y_baseline_parts.append(target[end_idx])
        y_times.append(times[end_idx + 1])
    X = np.stack(X_parts, axis=0)
    y = np.stack(y_parts, axis=0)
    y_baseline = np.stack(y_baseline_parts, axis=0)
    y_residual = y - y_baseline
    n = len(X)
    train_end = int(n * float(config["train_ratio"]))
    val_end = int(n * (float(config["train_ratio"]) + float(config["val_ratio"])))
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)
    mean = X[train_idx].mean(axis=(0, 1, 2), keepdims=True)
    std = X[train_idx].std(axis=(0, 1, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    X = (X - mean) / std
    y_mean = y_residual[train_idx].mean(axis=0, keepdims=True)
    y_std = y_residual[train_idx].std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    y_scaled = (y_residual - y_mean) / y_std
    metadata = {
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "num_nodes": len(grid_ids),
        "num_features": len(NODE_FEATURES),
        "node_features": NODE_FEATURES,
        "lookback": lookback,
        "target_scaled_by_node": True,
        "target_mode": "residual_next_minus_current",
        "architecture": "FullSTGCN(gated_temporal_conv, graph_conv, residual_stconv_blocks, layer_norm, dropout)",
    }
    (PATHS["tables"] / "stgcn_tensor_shapes.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log(f"STGCN tensors: X={X.shape}, y={y.shape}")
    return TensorBundle(X, y, y_scaled, y_baseline, np.array(y_times), grid_ids, train_idx, val_idx, test_idx, mean, std, y_mean, y_std)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False) -> DataLoader:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate_loss(model: nn.Module, loader: DataLoader, adj: torch.Tensor, loss_fn, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            losses.append(float(loss_fn(model(bx, adj), by).item()))
    return float(np.mean(losses)) if losses else float("inf")


def train_stgcn(config: Dict, quick: bool = False) -> pd.DataFrame:
    ensure_dirs()
    torch.manual_seed(int(config["random_state"]))
    bundle = build_stgcn_tensors(config)
    adj = pd.read_csv(PATHS["tables"] / "region_adjacency_matrix.csv", index_col=0)
    adj = adj.reindex(index=bundle.grid_ids, columns=bundle.grid_ids).to_numpy(dtype="float32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"train STGCN on {device}")
    model = FullSTGCN(len(NODE_FEATURES), int(config["hidden_channels"])).to(device)
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    train_loader = make_loader(bundle.X[bundle.train_idx], bundle.y_scaled[bundle.train_idx], int(config["batch_size"]), shuffle=True)
    val_loader = make_loader(bundle.X[bundle.val_idx], bundle.y_scaled[bundle.val_idx], int(config["batch_size"]))
    test_loader = make_loader(bundle.X[bundle.test_idx], bundle.y_scaled[bundle.test_idx], int(config["batch_size"]))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    epochs = int(config["quick_epochs"] if quick else config["epochs"])
    patience = int(config["patience"])
    best_val = float("inf")
    best_state = None
    wait = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            pred = model(bx, adj_t)
            loss = loss_fn(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        train_loss = float(np.mean(train_losses))
        val_loss = evaluate_loss(model, val_loader, adj_t, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        log(f"STGCN epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": model.state_dict(),
            "node_features": NODE_FEATURES,
            "grid_ids": bundle.grid_ids,
            "y_mean": bundle.y_mean,
            "y_std": bundle.y_std,
            "architecture": "FullSTGCN(gated_temporal_conv, graph_conv, residual_stconv_blocks, layer_norm, dropout)",
            "config": config,
        },
        PATHS["models"] / "stgcn_best.pt",
    )
    pd.DataFrame(history).to_csv(PATHS["tables"] / "stgcn_training_loss.csv", index=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for bx, by in test_loader:
            pred = model(bx.to(device), adj_t).cpu().numpy()
            preds.append(pred)
    pred_arr = np.concatenate(preds, axis=0)
    pred_arr = pred_arr * bundle.y_std + bundle.y_mean
    pred_arr = pred_arr + bundle.y_baseline[bundle.test_idx]
    y_true = bundle.y[bundle.test_idx]
    rows = []
    for sample_i, dt in enumerate(bundle.datetimes[bundle.test_idx]):
        for node_i, grid_id in enumerate(bundle.grid_ids):
            y_t = float(y_true[sample_i, node_i])
            y_p = float(pred_arr[sample_i, node_i])
            rows.append(
                {
                    "datetime": pd.Timestamp(dt),
                    "grid_id": grid_id,
                    "y_true_net_flow": y_t,
                    "y_pred_net_flow": y_p,
                    "error": y_p - y_t,
                    "abs_error": abs(y_p - y_t),
                }
            )
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(PATHS["tables"] / "stgcn_predictions.csv", index=False)
    metrics = regression_metrics(pred_df["y_true_net_flow"], pred_df["y_pred_net_flow"])
    metrics_row = pd.DataFrame([{ "model": "STGCN", **metrics, "notes": "spatio-temporal graph neural network" }])
    metrics_path = PATHS["tables"] / "regional_model_metrics.csv"
    if metrics_path.exists():
        base = pd.read_csv(metrics_path)
        base = base[base["model"] != "STGCN"]
        out = pd.concat([base, metrics_row], ignore_index=True).sort_values("RMSE")
    else:
        out = metrics_row
    out.to_csv(metrics_path, index=False)
    log("saved STGCN model, predictions, and metrics")
    return pred_df
