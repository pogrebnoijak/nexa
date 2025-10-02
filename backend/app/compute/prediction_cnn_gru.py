from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from app.compute.preprocessing import consider_global_events_simple, interpolate_signals


# FIXME It is necessary to properly disassemble the windows: the number of values ​​was approximately the same depending on the time - just pad now
def build_sequence_dataset(
    records: List[Dict],
    window_size_s: int = 600,
    step_s: int = 60,
    horizon_s: int = 600,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    records: список словарей по пациенткам (очищенные)
    window_size_s: длина окна (например 600 секунд = 10 минут)
    step_s: шаг (например 60 секунд = 1 минута)
    horizon_s: горизонт прогноза (например 600 секунд)
    return: X, y, groups
      X: (N, window_size, 2) — [fhr, toco]
      y: (N,) — 0/1 (будет ли децелерация в будущем интервале)
      groups: (N,) — patient_id
    """
    all_dts = []
    for rec in records:
        ts = np.asarray(rec["ts"])
        if len(ts) > 1:
            all_dts.extend(np.diff(ts))
    base_dt = np.median(all_dts)
    expected_len = int(round(window_size_s / base_dt))

    X, y, groups = [], [], []

    for rec in records:
        ts, fhr, toco, events, pid = (
            rec["ts"],
            rec["fhr"],
            rec["toco"],
            rec["events"],
            rec["patient_id"],
        )
        ts = np.asarray(ts)

        t_min, t_max = ts[0], ts[-1]
        cur_end = t_min + window_size_s
        while cur_end + horizon_s <= t_max:
            window_start = cur_end - window_size_s
            sel = (ts >= window_start) & (ts <= cur_end)
            fhr_w = fhr[sel]
            toco_w = toco[sel]

            if len(fhr_w) == 0:
                continue

            if len(fhr_w) > expected_len:
                fhr_w = fhr_w[-expected_len:]
                toco_w = toco_w[-expected_len:]
            elif len(fhr_w) < expected_len:
                pad_len = expected_len - len(fhr_w)
                fhr_w = np.pad(fhr_w, (pad_len, 0), constant_values=fhr_w[0])
                toco_w = np.pad(toco_w, (pad_len, 0), constant_values=toco_w[0])

            X.append(np.stack([fhr_w, toco_w], axis=-1))

            future_start, future_end = cur_end, cur_end + horizon_s
            y.append(
                1
                if any(
                    ev.kind == "decel" and future_start <= ev.t_start_s < future_end
                    for ev in events
                )
                else 0
            )
            groups.append(pid)

            cur_end += step_s

    return np.array(X), np.array(y), np.array(groups)


class CTGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class CNN_GRU_Model(nn.Module):
    def __init__(
        self, input_channels: int = 2, hidden_size: int = 64, num_layers: int = 1
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.gru = nn.GRU(64, hidden_size, num_layers=num_layers, batch_first=True)

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, channels)
        x = x.permute(0, 2, 1)  # -> (batch, channels, seq_len)
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))

        x = x.permute(0, 2, 1)  # -> (batch, seq_len', features)

        _, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden_size)
        x = h_n[-1]  # (batch, hidden_size)

        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x)).squeeze(-1)
        return x


def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu"):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{epochs} - Train loss: {total_loss / len(train_loader):.4f}, Val loss: {val_loss:.4f}"
        )

    return model


def evaluate_loss(model, loader, criterion, device="cpu"):
    model.eval()
    losses = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            losses.append(loss.item())
    return np.mean(losses)


def predict(model, loader, device="cpu"):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            preds.append(y_pred.cpu().numpy())
            trues.append(y_batch.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def evaluate_predictions(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)

    return {
        "auc": roc_auc_score(y_true, y_pred_proba)
        if len(np.unique(y_true)) > 1
        else None,
        "ap": average_precision_score(y_true, y_pred_proba)
        if len(np.unique(y_true)) > 1
        else None,
        "brier": brier_score_loss(y_true, y_pred_proba),
        "report": classification_report(y_true, y_pred, digits=3),
    }


def predict_on_data(
    model,
    ts: np.ndarray,
    fhr_raw: np.ndarray,
    toco_raw: np.ndarray,
    window_size_s: int = 600,
    device="cpu",
) -> Optional[float]:
    """
    Применяет обученную модель к новым данным.

    Args:
        model: обученная model
        ts: массив времени
        fhr_raw: сырой сигнал FHR
        toco_raw: сырой сигнал TOCO
        window_size_s: размер окна (сек), должно быть как в модели

    Returns:
        float: вероятность
    """
    fhr_clean, toco_clean, events = consider_global_events_simple(ts, fhr_raw, toco_raw)
    interpolated = interpolate_signals(ts, fhr=fhr_clean, toco=toco_clean)

    window_end = float(ts.tail(1))
    sel = ts >= window_end - window_size_s
    sel[max(0, sel.count() - sel.sum() - 1)] = True  # first element
    X, y, _ = build_sequence_dataset(
        [
            {
                "patient_id": id,
                "ts": ts[sel],
                "fhr": interpolated["fhr"][sel],
                "toco": interpolated["toco"][sel],
                "events": events,
            }
        ],
        window_size_s=window_size_s,
        horizon_s=0,
    )

    if len(X) == 0:
        return None

    test_ds = CTGDataset(X, y)  # y useless, ignore it
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    y_pred_proba, _ = predict(model, test_loader, device)
    return float(y_pred_proba[0])


def train_cnn_on_dataset(
    X, y, groups, test_size=0.2, batch_size=32, epochs=20, device="cpu"
):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_ds = CTGDataset(X_train, y_train)
    test_ds = CTGDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = CNN_GRU_Model(input_channels=X.shape[2])
    model = train_model(model, train_loader, test_loader, epochs=epochs, device=device)

    y_pred_proba, y_true = predict(model, test_loader, device)
    results = evaluate_predictions(y_true, y_pred_proba)

    return model, results
