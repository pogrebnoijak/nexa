from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    """
    Возвращает скользящую медиану с окном win. Минимальные края заполняются ближайшими значениями.
    """
    win = max(3, win if win % 2 == 1 else win + 1)
    s = pd.Series(x)
    y = s.rolling(win, center=True, min_periods=1).median().to_numpy()
    y = pd.Series(y).fillna(method="bfill").fillna(method="ffill").to_numpy()
    return y


def generate_ctg_stream(
    duration_min: float = 30.0, fs: float = 4.0, seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Генерирует синтетические ряды FHR и TOCO с реалистичными особенностями: схватки, акцелерации, децелерации, тахи/бради эпизоды и артефакты.
    Возвращает timestamps (сек с эпохи, float), fhr_raw, toco_raw.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_min * 60.0 * fs))

    t = np.arange(n) / fs
    t0 = pd.Timestamp.now().timestamp()
    ts = t0 + t

    base = (
        140.0
        + 2.0 * np.sin(2 * np.pi * t / (10 * 60.0))
        + 1.0 * np.sin(2 * np.pi * t / (6 * 60.0))
    )
    drift = np.cumsum(rng.normal(0.0, 0.02, size=n))
    base = base + _rolling_median(drift, int(max(3, fs * 20)))

    hf = rng.normal(0.0, 1.5, size=n)
    hf = pd.Series(hf).rolling(int(max(3, fs * 2)), min_periods=1).mean().to_numpy()
    fhr = base + hf

    toco = np.zeros(n, dtype=float)
    pos = 30.0
    peaks = []
    interval_s = rng.uniform(90.0, 180.0, size=int(duration_min / 2) + 3)
    for d in interval_s:
        p = int(min(n - 1, round((pos + d) * fs)))
        if p >= n:
            break
        peaks.append(p)
        pos += d
    for p in peaks:
        width_s = rng.uniform(40.0, 70.0)
        width = int(round(width_s * fs))
        amp = rng.uniform(10.0, 35.0)
        left = max(0, p - width // 2)
        right = min(n, p + width // 2)
        x = np.linspace(-1.0, 1.0, right - left)
        bump = amp * (1 - x**2)
        toco[left:right] += bump
        kind = rng.choice(["early", "late", "none"], p=[0.35, 0.35, 0.30])
        if kind != "none":
            shift = (
                int(round(rng.uniform(-10.0, 10.0) * fs))
                if kind == "early"
                else int(round(rng.uniform(15.0, 35.0) * fs))
            )
            nadir = p + shift
            if 0 < nadir < n:
                dec_duration_s = rng.uniform(20.0, 50.0)
                dec_width = int(round(dec_duration_s * fs))
                dleft = max(0, nadir - dec_width // 2)
                dright = min(n, nadir + dec_width // 2)
                x2 = np.linspace(-1.0, 1.0, dright - dleft)
                depth = rng.uniform(15.0, 30.0)
                dip = -depth * (1 - x2**2)
                fhr[dleft:dright] += dip

    num_acc = int(duration_min // 5) + 1
    for _ in range(num_acc):
        c = rng.integers(0, n)
        width = int(round(rng.uniform(10.0, 30.0) * fs))
        left = max(0, c - width // 2)
        right = min(n, c + width // 2)
        x = np.linspace(-1.0, 1.0, right - left)
        amp = rng.uniform(15.0, 25.0)
        fhr[left:right] += amp * (1 - x**2)
    if rng.random() < 0.7:
        s = rng.integers(int(5 * 60 * fs), int(10 * 60 * fs))
        e = min(n, s + int(3 * 60 * fs))
        fhr[s:e] += 20.0
    if rng.random() < 0.7:
        s = rng.integers(int(15 * 60 * fs), int(20 * 60 * fs))
        e = min(n, s + int(2 * 60 * fs))
        fhr[s:e] -= 25.0
    if rng.random() < 0.8:
        for _ in range(3):
            s = rng.integers(0, n - int(5 * fs))
            e = s + int(rng.uniform(3.0, 8.0) * fs)
            fhr[s:e] = np.nan

    fhr = np.clip(fhr, 60.0, 210.0)
    toco = (
        pd.Series(toco).rolling(int(max(3, fs * 1.5)), min_periods=1).mean().to_numpy()
    )
    return ts.astype(float), fhr.astype(float), toco.astype(float)
