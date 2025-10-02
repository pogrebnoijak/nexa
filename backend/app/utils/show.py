from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.abstract import Contraction, Event
from app.utils.other import contiguous_spans


def plot_series(
    ts: np.ndarray, fhr_raw: np.ndarray, fhr_clean: np.ndarray, baseline: np.ndarray
) -> None:
    """
    Рисует исходный и очищенный FHR и базовую линию.
    """
    plt.figure()
    plt.plot(pd.to_datetime(ts, unit="s"), fhr_raw, label="FHR raw")
    plt.plot(pd.to_datetime(ts, unit="s"), fhr_clean, label="FHR clean")
    plt.plot(pd.to_datetime(ts, unit="s"), baseline, label="baseline")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("bpm")
    plt.title("FHR raw / clean / baseline")
    plt.tight_layout()
    plt.show()


def plot_toco(ts: np.ndarray, toco: np.ndarray) -> None:
    """
    Рисует TOCO.
    """
    plt.figure()
    plt.plot(pd.to_datetime(ts, unit="s"), toco, label="TOCO")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("toco")
    plt.title("TOCO")
    plt.tight_layout()
    plt.show()


def plot_variability(ts: np.ndarray, stv: np.ndarray, ltv: np.ndarray) -> None:
    """
    Рисует STV и LTV.
    """
    plt.figure()
    plt.plot(pd.to_datetime(ts, unit="s"), stv, label="STV")
    plt.plot(pd.to_datetime(ts, unit="s"), ltv, label="LTV")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("bpm")
    plt.title("Variability metrics")
    plt.tight_layout()
    plt.show()


def plot_events_on_fhr(
    ts: np.ndarray, fhr_clean: np.ndarray, events: List[Event]
) -> None:
    """
    Рисует FHR с вертикальными линиями начала событий.
    """
    plt.figure()
    plt.plot(pd.to_datetime(ts, unit="s"), fhr_clean, label="FHR clean")
    for ev in events:
        t0 = ts[ev.t_start]
        plt.axvline(pd.to_datetime(t0, unit="s"))
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("bpm")
    plt.title("Events on FHR")
    plt.tight_layout()
    plt.show()


def plot_ctg(
    timestamps: np.ndarray,
    fhr_raw: Optional[np.ndarray],
    fhr_clean: np.ndarray,
    toco: Optional[np.ndarray],
    baseline: Optional[np.ndarray],
    events: Optional[List[Event]] = None,
    contractions: Optional[List[Contraction]] = None,
    stv: Optional[np.ndarray] = None,
    ltv: Optional[np.ndarray] = None,
    quality: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (14, 8),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Рисует CTG в привычном виде:
      - Верхняя панель: FHR (raw faded, clean), baseline (линия), события (децелерации/акцелерации) — прямоугольники и nadir/peaks
      - Нижняя панель: TOCO (шкала), отмеченные пики схваток (если contractions)
      - Нижняя полоска: STV/LTV (если заданы) и качество (серые интервалы)
    Возвращает matplotlib.Figure.
    """
    ts_dt = (
        pd.to_datetime(timestamps, unit="s")
        if np.issubdtype(timestamps.dtype, np.number)
        else pd.to_datetime(timestamps)
    )
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 0.8], hspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    if fhr_raw is not None:
        ax1.plot(ts_dt, fhr_raw, color="0.7", linewidth=0.8, label="FHR raw", alpha=0.6)
    ax1.plot(ts_dt, fhr_clean, color="C0", linewidth=1.2, label="FHR clean")
    if baseline is not None:
        ax1.plot(
            ts_dt, baseline, color="C3", linestyle="--", linewidth=1.5, label="Baseline"
        )
    if quality is not None:
        bad_mask = ~quality
        if np.any(bad_mask):
            spans = contiguous_spans(bad_mask)
            for s, e in spans:
                ax1.axvspan(ts_dt[s], ts_dt[e - 1], color="grey", alpha=0.15)

    if events:
        for ev in events:
            color = {
                "decel": "red",
                "accel": "green",
                "tachy": "orange",
                "brady": "purple",
            }.get(ev.kind, "black")
            ax1.axvspan(ts_dt[ev.t_start], ts_dt[ev.t_end - 1], color=color, alpha=0.18)
            if ev.t_nadir is not None:
                ax1.plot(
                    ts_dt[ev.t_nadir],
                    (ev.min_val or ev.max_val or np.nan),
                    marker="v",
                    color=color,
                )

    ax1.set_ylabel("FHR (bpm)")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    if toco is not None:
        ax2.plot(ts_dt, toco, color="C2", linewidth=1.0, label="TOCO")
        ax2.set_ylabel("TOCO")
        if contractions:
            for c in contractions:
                ax2.axvline(ts_dt[c.peak], color="C2", linestyle="--", alpha=0.6)
    ax2.grid(True)
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    if stv is not None:
        ax3.plot(ts_dt, stv, label="STV")
    if ltv is not None:
        ax3.plot(ts_dt, ltv, label="LTV")
    if quality is not None:
        bad_mask = ~quality
        if np.any(bad_mask):
            spans = contiguous_spans(bad_mask)
            for s, e in spans:
                ax3.axvspan(ts_dt[s], ts_dt[e - 1], color="grey", alpha=0.2)
    ax3.set_ylabel("Variability")
    ax3.set_xlabel("time")
    ax3.legend(loc="upper right")
    ax3.grid(True)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_with_mask(
    data,
    mask,
    baseline=None,
    figsize=(12, 6),
    data_color="blue",
    baseline_color="green",
    mask_color="red",
    alpha=0.3,
):
    assert len(data) == len(mask)

    x = np.arange(len(data))

    _, ax = plt.subplots(figsize=figsize)
    ax.plot(x, data, color=data_color, linewidth=1.5, label="data")
    if baseline is not None:
        ax.plot(x, baseline, color=baseline_color, linewidth=1, label="baseline")
    ax.fill_between(
        x,
        np.min(data),
        np.max(data),
        where=mask,
        color=mask_color,
        alpha=alpha,
        label="mask",
    )
    ax.set_title("mask plot", fontsize=14, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
