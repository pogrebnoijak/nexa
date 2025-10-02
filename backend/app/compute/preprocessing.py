from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, medfilt
from sortedcontainers import SortedList

import app.consts as c
from app.abstract import Contraction, Event
from app.utils import contiguous_spans

# 0


def _odd_kernel(n: int) -> int:
    return 3 if n < 3 else n if n % 2 == 1 else n + 1


def _remove_short_true_runs_time(
    mask: np.ndarray, ts: np.ndarray, min_duration: float
) -> np.ndarray:
    """Очищает mask от слишком коротких фрагментов (< min_duration секунд)."""
    mask = mask.copy()
    spans = contiguous_spans(mask)
    for s, e in spans:
        dur_s = ts[e - 1] - ts[s] if e > s else 0
        if dur_s < min_duration:
            mask[s:e] = False
    return mask


def interpolate_signals(ts: np.ndarray, **signals) -> dict:
    """
    Интерполирует несколько сигналов (например, fhr, toco) по временной шкале ts.

    Args:
        ts: np.ndarray - временная шкала (равномерная или с пропусками, но отсортированная)
        **signals: np.ndarray - сигналы в виде именованных аргументов

    Returns:
        dict: {имя_сигнала: интерполированный np.ndarray}
    """
    df = pd.DataFrame({"ts": ts, **signals})
    df = df.interpolate(method="linear", limit_direction="both")
    return {col: df[col].to_numpy() for col in signals}


# 1


def clean_signal(
    fhr: np.ndarray,
    ts: np.ndarray,
    delta_max_bpm_per_s: float = 15.0,
    gap_max_s: float = 2.5,
    artifact_gap_max_s: float = 3.0,
    median_win_s: float = 1.0,
    smooth_win_s: float = 0.5,
    jump_expand: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает FHR_clean и quality_mask.
    ts - массив временных меток в секундах (может быть неравномерный шаг).

    1) Помечает скачки быстрее delta_max_bpm_per_s.
    2) Медианный фильтр по окну median_win_s.
    3) Сглаживание скользящим средним smooth_win_s.
    4) Интерполяция только коротких провалов ≤ gap_max_s.
    """
    fhr = np.asarray(fhr, dtype=float)
    ts = np.asarray(ts, dtype=float)
    fhr_size = fhr.size
    dt_mean = np.median(np.diff(ts))

    valid_idx = np.where(np.isfinite(fhr))[0]
    if valid_idx.size > 1:
        local_dt = np.diff(ts[valid_idx])
        local_diff = np.abs(np.diff(fhr[valid_idx]))
        jump_idx = valid_idx[1:][local_diff > delta_max_bpm_per_s * local_dt]
    else:
        jump_idx = np.array([], dtype=int)

    segments = []
    if jump_idx.size > 0:
        start, prev = jump_idx[0], jump_idx[0]
        for idx in jump_idx[1:]:
            if ts[idx] - ts[prev] <= artifact_gap_max_s:
                prev = idx
            else:
                segments.append((start, prev))
                start, prev = idx, idx
        segments.append((start, prev))
    if segments:
        if ts[segments[0][0]] - ts[0] <= gap_max_s:
            segments[0] = (0, segments[0][1])
        if ts[-1] - ts[segments[-1][1]] <= gap_max_s:
            segments[-1] = (segments[-1][0], fhr_size - 1)
    jump_mask = np.zeros(fhr_size, dtype=bool)
    for s, e in segments:
        s = max(0, s - jump_expand)
        e = min(fhr_size - 1, e + jump_expand)
        jump_mask[s : e + 1] = True
    valid = np.isfinite(fhr) & (~jump_mask)

    fhr_filtered = fhr.copy()
    starts = np.where((valid) & np.concatenate(([False], ~valid[:-1])))[0]
    ends = np.where((valid) & np.concatenate((~valid[1:], [False])))[0] + 1

    k_med = _odd_kernel(max(1, int(round(median_win_s / dt_mean))))
    w = max(1, int(round(smooth_win_s / dt_mean)))
    kernel = np.ones(w, dtype=float) / float(w)

    for start, end in zip(starts, ends):
        seg = fhr_filtered[start:end].copy()
        if seg.size < 2:
            continue
        seg = (
            pd.Series(seg)
            .rolling(k_med, center=True, min_periods=1)
            .median()
            .to_numpy()
        )
        fhr_filtered[start:end] = np.convolve(
            np.pad(seg, pad_width=((w - 1) // 2, w // 2), mode="edge"),
            kernel,
            mode="valid",
        )

    fhr_filtered[~valid] = np.nan
    if not valid.all():
        starts = np.where((~valid) & np.concatenate(([True], valid[:-1])))[0]
        ends = np.where((~valid) & np.concatenate((valid[1:], [True])))[0] + 1

        for start, end in zip(starts, ends):
            gap_len_s = ts[end - 1] - ts[start] if end > start else 0
            if gap_len_s <= gap_max_s:
                left = start - 1
                right = end
                left_val = (
                    fhr_filtered[left]
                    if left >= 0 and np.isfinite(fhr_filtered[left])
                    else np.nan
                )
                right_val = (
                    fhr_filtered[right]
                    if right < fhr_size and np.isfinite(fhr_filtered[right])
                    else np.nan
                )

                if np.isfinite(left_val) and np.isfinite(right_val):
                    fhr_filtered[start:end] = np.linspace(
                        left_val, right_val, end - start + 2
                    )[1:-1]
                elif np.isfinite(left_val):
                    fhr_filtered[start:end] = left_val
                elif np.isfinite(right_val):
                    fhr_filtered[start:end] = right_val

    quality = np.isfinite(fhr_filtered)
    return fhr_filtered, quality.astype(bool)


def clean_toco(
    toco: np.ndarray,
    ts: np.ndarray,
    phys_min: float = 0.0,
    phys_max: float = 100.0,
    diff_thr: float = 30.0,
    med_win_s: float = 3.0,
) -> np.ndarray:
    """
    Очищает TOCO от артефактов:
      - ограничение диапазона [phys_min, phys_max],
      - удаление выбросов по скачкам,
      - медианное сглаживание на коротком окне.

    Аргументы:
        toco: сигнал TOCO (массив, может содержать NaN).
        ts: временная ось (секунды).
        phys_min, phys_max: допустимый физиологический диапазон.
        diff_thr: допустимый скачок между соседними точками.
        med_win_s: окно медианного фильтра (секунды).
    """
    toco = np.asarray(toco, dtype=float)
    ts = np.asarray(ts, dtype=float)
    cleaned = toco.copy()

    mask = (cleaned < phys_min) | (cleaned > phys_max)
    cleaned[mask] = np.nan

    diffs = np.abs(np.diff(cleaned, prepend=cleaned[0]))
    cleaned[diffs > diff_thr] = np.nan

    dt = np.median(np.diff(ts))
    win = int(round(med_win_s / dt))
    if win % 2 == 0:
        win += 1

    finite = np.isfinite(cleaned)
    if finite.any():
        smoothed = cleaned.copy()
        smoothed[finite] = medfilt(cleaned[finite], kernel_size=win)
        cleaned = smoothed
    return cleaned


# 2


def compute_global_baseline(
    fhr: np.ndarray,
    ts: np.ndarray,
    diff_thr_bpm_per_s: float = 10.0,
    min_stable_s: float = 5.0,
    accel_dec_thr_bpm: float = 15.0,
    min_event_dur_s: float = 15.0,
    osc_std_window_s: float = 60.0,
    osc_std_factor: float = 2.0,
    max_iter: int = 5,
) -> Tuple[float, np.ndarray]:
    """
    Находит глобальный baseline (медиану) и маску стабильных точек (True = точка считается для baseline).
    Алгоритм:
      1) помечает "стабильные" точки по |ΔFHR/Δt| < diff_thr_bpm_per_s для соседних семплов;
      2) убирает слишком короткие стабильные фрагменты (< min_stable_s);
      3) вычисляет локальную std (окно osc_std_window_s) и исключает участки с std > osc_std_factor * median_std;
      4) итеративно: вычисляет медиану по оставшимся, удаляет продолжительные окна (>= min_event_dur_s) где
         |FHR - медиана| >= accel_dec_thr_bpm (т.е. вероятные акцел/дец), повторяет пока не стабилизируется.
    Параметры:
      fhr: одномерный numpy массив (float). Допустимы NaN/Inf — они игнорируются.
      ts: одномерный numpy массив (float) временных меток в секундах (может быть неравномерный шаг).
      diff_thr_bpm_per_s: порог bpm/сек для соседних разностей (по умолчанию 10 bpm/s).
      min_stable_s: минимальная длина стабильного сегмента в секундах.
      accel_dec_thr_bpm: порог в bpm для обнаружения акцел/децел относительно медианы.
      min_event_dur_s: минимальная длительность события (сек) при фильтрации.
      osc_std_window_s: окно (сек) для оценки локальной std (для поиска осцилляций).
      osc_std_factor: множитель для определения "высокой" локальной std.
      max_iter: максимальное число итераций удаления событий.

    Возвращает:
      (global_baseline, stable_mask)
        global_baseline: float (медиана по stable_mask, либо медиана по всему сигналу если stable_mask пуст)
        stable_mask: np.ndarray[bool] длины fhr, True = используется в baseline
    """
    fhr = np.asarray(fhr, dtype=float)
    ts = np.asarray(ts, dtype=float)
    n = fhr.size
    if n == 0:
        return float("nan"), np.zeros(0, dtype=bool)

    finite = np.isfinite(fhr)
    if not finite.any():
        return float("nan"), np.zeros(n, dtype=bool)

    dt = np.diff(ts, prepend=ts[0])
    dt_median = np.median(np.diff(ts))

    valid_pairs = finite & np.isfinite(np.roll(fhr, 1))
    rate = np.zeros_like(fhr)
    rate[valid_pairs] = np.abs(
        fhr[valid_pairs] - np.roll(fhr, 1)[valid_pairs]
    ) / np.maximum(dt[valid_pairs], c.ZERO_PLUS)
    stable_mask = (rate < diff_thr_bpm_per_s) & finite

    stable_mask = _remove_short_true_runs_time(stable_mask, ts, min_stable_s)

    osc_win = max(1, int(round(osc_std_window_s / dt_median)))
    local_std = (
        pd.Series(fhr)
        .rolling(osc_win, center=True, min_periods=1)
        .std(ddof=0)
        .to_numpy()
    )
    median_std = float(np.nanmedian(local_std[np.isfinite(local_std)]))
    if not np.isfinite(median_std) or median_std <= 0.0:
        median_std = c.ZERO_PLUS

    osc_mask = local_std > (osc_std_factor * median_std)
    stable_mask = stable_mask & (~osc_mask)

    baseline_val = (
        float(np.nanmedian(fhr[stable_mask]))
        if stable_mask.any()
        else float(np.nanmedian(fhr[finite]))
    )

    for _ in range(max_iter):
        if stable_mask.any():
            baseline_val = float(np.nanmedian(fhr[stable_mask]))
        else:
            baseline_val = float(np.nanmedian(fhr[finite]))
            break

        deviation_mask = np.abs(fhr - baseline_val) >= accel_dec_thr_bpm
        event_spans = contiguous_spans(deviation_mask)

        removed_any = False
        for s, e in event_spans:
            dur_s = ts[e - 1] - ts[s] if e > s else 0
            if dur_s >= min_event_dur_s:
                stable_mask[s:e] = False
                removed_any = True

        if not removed_any:
            break

        stable_mask = _remove_short_true_runs_time(stable_mask, ts, min_stable_s)
        stable_mask = stable_mask & (~osc_mask)

    if stable_mask.any():
        global_baseline = float(np.nanmedian(fhr[stable_mask]))
    else:
        global_baseline = float(np.nanmedian(fhr[finite]))
        stable_mask[:] = False

    return global_baseline, stable_mask


def compute_window_baseline(
    fhr: np.ndarray,
    mask: np.ndarray,
    ts: np.ndarray,
    global_baseline: float,
    window: Optional[Tuple[int, int]] = None,
    max_dev: float = 15.0,
) -> Tuple[float, bool]:
    """
    Baseline по окну в секундах [t_start, t_end].
    Возвращает значение и флаг доверия (True=надежен).
    Если отклонение > max_dev от глобального baseline, ограничиваем.
    """
    if window is None:
        window_vals = fhr[mask]
    else:
        t_start, t_end = window
        idx = (ts >= t_start) & (ts < t_end) & mask
        window_vals = fhr[idx]
    window_vals = window_vals[np.isfinite(window_vals)]

    if window_vals.size == 0:
        return global_baseline, False

    local_med = float(np.median(window_vals))
    if abs(local_med - global_baseline) > max_dev:
        return global_baseline, False

    return local_med, True


def compute_baseline_curve(
    fhr: np.ndarray,
    stable_mask: np.ndarray,
    ts: np.ndarray,
    global_baseline: float,
    win_size_s: float,
    step_s: float = 1.0,
    max_dev: float = 15.0,
    max_rate: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Baseline по всему сигналу, скользящим окном.

    win_size   : размер окна (в отсчетах)
    step       : шаг смещения окна
    max_dev    : макс. допустимое отклонение от глобального baseline
    max_rate   : макс. изменение baseline на шаг (bpm)

    Возвращает:
        baseline_curve : np.ndarray
        reliable_mask  : np.ndarray (True = baseline надежен)
        curve_ts       : np.ndarray (времена центров окон)
    """
    t_min, t_max = ts[0], ts[-1]
    centers = np.arange(t_min, t_max - win_size_s + step_s, step_s)
    baseline_curve = np.full(centers.shape, np.nan, dtype=float)
    reliable_mask = np.zeros(centers.shape, dtype=bool)
    max_step = max_rate * step_s

    prev_val = global_baseline
    for i, t_start in enumerate(centers):
        t_end = t_start + win_size_s
        val, reliable = compute_window_baseline(
            fhr, stable_mask, ts, global_baseline, (t_start, t_end), max_dev
        )
        if not reliable:
            val = prev_val

        if reliable:
            if val > prev_val + max_step:
                val = prev_val + max_step
                reliable = False
            elif val < prev_val - max_step:
                val = prev_val - max_step
                reliable = False

        baseline_curve[i] = val
        reliable_mask[i] = reliable
        prev_val = val

    return baseline_curve, reliable_mask, centers


def compute_baseline_curve_iterative(
    fhr: np.ndarray,
    stable_mask: np.ndarray,
    ts: np.ndarray,
    global_baseline: float,
    win_size_s: float,
    step_s: float = 1.0,
    max_dev: float = 15.0,
    max_rate: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Оптимальная версия: baseline скользящим окном, медиана поддерживается итеративно.
    Возвращает baseline_curve, reliable_mask и centers.
    """
    t_min, t_max = ts[0], ts[-1]
    centers = np.arange(t_min, t_max - win_size_s + step_s, step_s)
    baseline_curve = np.full(centers.shape, np.nan, dtype=float)
    reliable_mask = np.zeros(centers.shape, dtype=bool)
    max_step = max_rate * step_s

    idx0 = (ts >= centers[0]) & (ts < centers[0] + win_size_s) & stable_mask
    window = SortedList(fhr[idx0][np.isfinite(fhr[idx0])])

    def get_median() -> Tuple[float, bool]:
        if len(window) == 0:
            return global_baseline, False

        mid = len(window) // 2
        if len(window) % 2 == 1:
            local_med = float(window[mid])
        else:
            local_med = 0.5 * (window[mid - 1] + window[mid])

        if abs(local_med - global_baseline) > max_dev:
            return global_baseline, False
        return local_med, True

    prev_val = global_baseline
    for i, t_start in enumerate(centers):
        t_end = t_start + win_size_s
        if i > 0:
            idx_out = (ts >= centers[i - 1]) & (ts < t_start) & stable_mask
            for v in fhr[idx_out]:
                if np.isfinite(v):
                    window.remove(v)
            idx_in = (ts >= t_end - step_s) & (ts < t_end) & stable_mask
            for v in fhr[idx_in]:
                if np.isfinite(v):
                    window.add(v)

        val, reliable = get_median()
        if not reliable:
            val = prev_val

        if reliable:
            if val > prev_val + max_step:
                val = prev_val + max_step
                reliable = False
            elif val < prev_val - max_step:
                val = prev_val - max_step
                reliable = False

        baseline_curve[i] = val
        reliable_mask[i] = reliable
        prev_val = val

    return baseline_curve, reliable_mask, centers


def compute_baseline_line(
    fhr: np.ndarray,
    stable_mask: np.ndarray,
    ts: np.ndarray,
    global_baseline: float,
    win_size: int = 200,
    step: int = 20,
    max_dev: float = 15.0,
    max_rate: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    baseline_curve, reliable_mask, _ = compute_baseline_curve_iterative(
        fhr, stable_mask, ts, global_baseline, win_size, step, max_dev, max_rate
    )
    # baseline_curve, reliable_mask, _ = compute_baseline_curve(fhr, stable_mask, ts, global_baseline, win_size, step, max_dev, max_rate)
    if win_size == 1 and step == 1:
        return baseline_curve, reliable_mask

    n = len(fhr)
    baseline_line = np.zeros(n, dtype=float)
    reliable_line = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=int)
    for i, (val, rel) in enumerate(zip(baseline_curve, reliable_mask)):
        start = i * step
        end = min(start + win_size, n)
        baseline_line[start:end] += val
        reliable_line[start:end] += float(rel)
        counts[start:end] += 1

    nonzero = counts > 0
    baseline_line[nonzero] /= counts[nonzero]
    reliable_line[nonzero] /= counts[nonzero]
    reliable_line = reliable_line >= c.RELIABLE_K

    if not np.all(nonzero):
        idx = np.arange(n)
        valid_idx = idx[nonzero]
        not_valid_idx = idx[~nonzero]
        baseline_line[~nonzero] = np.interp(
            not_valid_idx, valid_idx, baseline_line[nonzero]
        )
        reliable_line[~nonzero] = reliable_line[nonzero][
            np.searchsorted(valid_idx, not_valid_idx, side="left") - 1
        ]

    return baseline_line, reliable_line


# 3


def variability_metrics(  # TODO add metrics variability_metrics_window (rmssd, amp, freq)
    fhr: np.ndarray,
    ts: np.ndarray,
    stv_win_s: float = 60.0,
    ltv_win_s: float = 120.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает STV, LTV как ряды.
    STV: среднее абсолютное приращение в окне stv_win_s.
    LTV: стандартное отклонение в окне ltv_win_s.
    """
    n = len(fhr)
    stv = np.full(n, np.nan, dtype=float)
    ltv = np.full(n, np.nan, dtype=float)

    for i in range(n):
        t0 = ts[i] - stv_win_s / 2.0
        t1 = ts[i] + stv_win_s / 2.0
        idx = (ts >= t0) & (ts <= t1)
        if idx.any():
            vals = fhr[idx]
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                stv[i] = np.abs(np.diff(vals, prepend=vals[0])).mean()

        t0 = ts[i] - ltv_win_s / 2.0
        t1 = ts[i] + ltv_win_s / 2.0
        idx = (ts >= t0) & (ts <= t1)
        if idx.any():
            vals = fhr[idx]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 1:
                ltv[i] = np.std(vals, ddof=1)
            else:
                ltv[i] = 0.0
    return stv, ltv


def variability_metrics_window(
    fhr: np.ndarray,
    ts: np.ndarray,
    baseline: float,
    window: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float, float, float, float]:
    """
    Возвращает метрики вариабельности:
    - STV: среднее абсолютное приращение за последнюю 1 минуту.
    - LTV: стандартное отклонение за последние 10 минут.
    - RMSSD: корень из среднего квадрата разностей за последнюю 1 минуту.
    - Амплитуда осцилляций: среднее |fhr - baseline| пиков за последние 30 минут.
    - Частота осцилляций: число колебаний в минуту за последние 30 минут.
    """
    if window is not None:
        t0, t1 = window
        mask = (ts >= t0) & (ts < t1)
        ts = ts[mask]
        fhr = fhr[mask]

    finite_mask = np.isfinite(fhr)
    ts = ts[finite_mask]
    fhr = fhr[finite_mask]

    if ts.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    t_end = ts[-1]
    stv, rmssd = np.nan, np.nan
    mask_1min = ts >= (t_end - c.MINUTE)
    vals_1min = fhr[mask_1min]
    if vals_1min.size > 1:
        diffs = np.diff(vals_1min, prepend=vals_1min[0])
        stv = np.mean(np.abs(diffs))
        rmssd = np.sqrt(np.mean(diffs**2))

    ltv = np.nan
    mask_10min = ts >= (t_end - c.MINUTE * 10)
    vals_10min = fhr[mask_10min]
    if vals_10min.size > 1:
        ltv = np.std(vals_10min, ddof=1)

    amp, freq = np.nan, np.nan
    mask_30min = ts >= (t_end - c.MINUTE * 30)
    vals_30min = fhr[mask_30min]
    ts_30min = ts[mask_30min]

    if vals_30min.size > 1:
        peaks, _ = find_peaks(vals_30min)
        troughs, _ = find_peaks(-vals_30min)
        extrema = np.sort(np.concatenate([peaks, troughs]))

        if extrema.size > 0:
            deviations = np.abs(vals_30min[extrema] - baseline)
            amp = deviations.mean() if deviations.size > 0 else np.nan

        dur_min = (ts_30min[-1] - ts_30min[0]) / c.MINUTE
        if dur_min > 0:
            freq = extrema.size / dur_min

    return stv, ltv, rmssd, amp, freq


# 4


def detect_events(
    fhr: np.ndarray,
    ts: np.ndarray,
    baseline: np.ndarray,
    accel_thr_bpm: float = 15.0,
    decel_thr_bpm: float = 15.0,
    accel_min_dur_s: float = 15.0,
    decel_min_dur_s: float = 15.0,
    tachy_bpm: float = 160.0,
    brady_bpm: float = 110.0,
    tachy_min_dur_s: float = 600.0,
    brady_min_dur_s: float = 180.0,
) -> List[Event]:
    """
    Выделяет акцелерации, децелерации, тахи- и бради-эпизоды.
    """
    events: List[Event] = []

    above = (fhr - baseline) >= accel_thr_bpm
    for start, end in contiguous_spans(above):
        dur = ts[end - 1] - ts[start]
        if dur >= accel_min_dur_s:
            seg = fhr[start:end]
            base = baseline[start:end]
            mask = np.isfinite(seg) & np.isfinite(base)
            if not mask.any():
                continue
            seg, base = seg[mask], base[mask]
            rel = seg - base
            mx = float(np.max(seg))
            depth = float(np.max(rel))
            area = float(np.trapezoid(rel, x=ts[start:end][mask]))
            events.append(
                Event(
                    kind="accel",
                    t_start=start,
                    t_end=end,
                    t_start_s=ts[start],
                    duration_s=dur,
                    min_val=None,
                    max_val=mx,
                    depth=depth,
                    area=area,
                    t_nadir=None,
                    toco_rel=None,
                )
            )

    below = (baseline - fhr) >= decel_thr_bpm
    for start, end in contiguous_spans(below):
        dur = ts[end - 1] - ts[start]
        if dur >= decel_min_dur_s:
            seg = fhr[start:end]
            base = baseline[start:end]
            mask = np.isfinite(seg) & np.isfinite(base)
            if not mask.any():
                continue
            seg, base = seg[mask], base[mask]
            rel = base - seg
            nadir_rel_idx = int(np.argmax(rel))
            mn = float(np.min(seg))
            depth = float(np.max(rel))
            area = float(np.trapezoid(rel, x=ts[start:end][mask]))
            events.append(
                Event(
                    kind="decel",
                    t_start=start,
                    t_end=end,
                    t_start_s=ts[start],
                    duration_s=dur,
                    min_val=mn,
                    max_val=None,
                    depth=depth,
                    area=area,
                    t_nadir=start + nadir_rel_idx,
                    toco_rel=None,
                )
            )

    tachy_mask = fhr >= tachy_bpm
    for start, end in contiguous_spans(tachy_mask):
        dur = ts[end - 1] - ts[start]
        if dur >= tachy_min_dur_s:
            seg = fhr[start:end]
            base = baseline[start:end]
            mask = np.isfinite(seg) & np.isfinite(base)
            if not mask.any():
                continue
            seg, base = seg[mask], base[mask]
            mx = float(np.max(seg))
            depth = float(np.max(seg - base))
            events.append(
                Event(
                    kind="tachy",
                    t_start=start,
                    t_end=end,
                    t_start_s=ts[start],
                    duration_s=dur,
                    min_val=None,
                    max_val=mx,
                    depth=depth,
                    area=None,
                    t_nadir=None,
                    toco_rel=None,
                )
            )

    brady_mask = fhr <= brady_bpm
    for start, end in contiguous_spans(brady_mask):
        dur = ts[end - 1] - ts[start]
        if dur >= brady_min_dur_s:
            seg = fhr[start:end]
            base = baseline[start:end]
            mask = np.isfinite(seg) & np.isfinite(base)
            if not mask.any():
                continue
            seg, base = seg[mask], base[mask]
            mn = float(np.min(seg))
            depth = float(np.max(base - seg))
            events.append(
                Event(
                    kind="brady",
                    t_start=start,
                    t_end=end,
                    t_start_s=ts[start],
                    duration_s=dur,
                    min_val=mn,
                    max_val=None,
                    depth=depth,
                    area=None,
                    t_nadir=None,
                    toco_rel=None,
                )
            )

    return events


def extract_contractions(
    toco: np.ndarray,
    ts: np.ndarray,
    min_peak_prominence: float = 5.0,
    min_distance_s: float = 45.0,
    start_threshold_rel: float = 0.3,
    cluster_distance_s: float = 10.0,
    use_clustering: bool = True,
    min_dur_s: float = 20.0,
    max_dur_s: float = 180.0,
) -> List[Contraction]:
    """
    Выделяет схватки по TOCO:
    - пики с минимальной выраженностью,
    - объединение близких пиков в одну схватку (опционально),
    - старт/конец по относительному порогу,
    - фильтр по длительности.
    """
    toco = np.asarray(toco, dtype=float)
    ts = np.asarray(ts, dtype=float)
    toco_size = toco.size
    dt = float(np.median(np.diff(ts)))

    toco_proc = np.where(np.isfinite(toco), toco, -np.inf)

    distance = max(1, int(round(min_distance_s / dt)))
    peaks, _ = find_peaks(toco_proc, distance=distance, prominence=min_peak_prominence)
    if peaks.size == 0:
        return []

    if use_clustering:
        cluster_dist = max(1, int(round(cluster_distance_s / dt)))
        clusters: List[List[int]] = []
        current = [peaks[0]]
        for p in peaks[1:]:
            if p - current[-1] <= cluster_dist:
                current.append(p)
            else:
                clusters.append(current)
                current = [p]
        clusters.append(current)
        peak_list = [int(cluster[np.nanargmax(toco[cluster])]) for cluster in clusters]
    else:
        peak_list = peaks.tolist()

    contractions: List[Contraction] = []
    for p in peak_list:
        left_lim = max(0, p - distance)
        right_lim = min(toco_size, p + distance + 1)

        left_min = (
            float(np.nanmin(toco[left_lim:p])) if p > left_lim else float(toco[p])
        )
        right_min = (
            float(np.nanmin(toco[p + 1 : right_lim]))
            if p + 1 < right_lim
            else float(toco[p])
        )
        base = min(left_min, right_min)
        prom = float(toco[p] - base)
        th = base + prom * start_threshold_rel

        i = p
        while i > 0 and np.isfinite(toco[i]) and toco[i] > th:
            i -= 1
        j = p
        while j < toco_size - 1 and np.isfinite(toco[j]) and toco[j] > th:
            j += 1

        dur_s = ts[j] - ts[i]
        if min_dur_s <= dur_s <= max_dur_s:
            contractions.append(Contraction(start=i, peak=p, end=j))
    return contractions


def classify_decelerations_wrt_toco(
    events: List[Event],
    toco: Optional[np.ndarray],
    contractions: List[Contraction],
    ts: np.ndarray,
    d1: float = 30.0,
    d2: float = 60.0,
    min_duration: float = 15.0,
    min_depth: float = 15.0,
) -> List[Event]:
    """
    Классификация децелераций относительно схваток: early / late / variable.

    Правила:
      - Early: начало децелерации ≈ начало схватки (±d1), nadir ≈ peak схватки.
      - Late: начало децелерации после начала схватки, nadir > peak + d1.
      - Variable: остальное (или ближайшая схватка дальше d2).
      - Если децелерация слишком короткая (<15с) и мелкая (<15 bpm) → artifact.
    """
    if toco is None or len(toco) == 0 or not np.any(np.isfinite(toco)):
        for ev in events:
            if ev.kind == "decel":
                if ev.duration_s < min_duration and (
                    ev.depth is None or ev.depth < min_depth
                ):
                    ev.toco_rel = "artifact"
                else:
                    ev.toco_rel = "variable"
        return events

    for ev in events:
        if ev.kind != "decel":
            continue
        if ev.t_nadir is None:
            ev.toco_rel = None
            continue

        if ev.duration_s < min_duration and (ev.depth is None or ev.depth < min_depth):
            ev.toco_rel = "artifact"
            continue

        overlaps = [
            c for c in contractions if not (ev.t_end < c.start or ev.t_start > c.end)
        ]
        if overlaps:
            overlaps.sort(
                key=lambda c: min(ev.t_end, c.end) - max(ev.t_start, c.start),
                reverse=True,
            )
            target = overlaps[0]
        else:
            if contractions:
                dists = [(abs(ev.t_nadir - c.peak), c) for c in contractions]
                target = min(dists, key=lambda x: x[0])[1]
                if abs(ts[ev.t_nadir] - ts[target.peak]) > d2:
                    ev.toco_rel = "variable"
                    continue
            else:
                ev.toco_rel = "variable"
                continue

        start_offset = ts[ev.t_start] - ts[target.start]
        nadir_offset = ts[ev.t_nadir] - ts[target.peak]

        if abs(start_offset) <= d1 and abs(nadir_offset) <= d1:
            ev.toco_rel = "early"
        elif start_offset >= 0 and nadir_offset > d1:
            ev.toco_rel = "late"
        else:
            ev.toco_rel = "variable"
    return events


# scores


def fisher_score(
    baseline_bpm: float,
    osc_amp: Optional[float],  # амплитуда осцилляций (уд/мин)
    osc_freq: Optional[float],  # частота осцилляций (осц/мин)
    num_accel_per_30min: Optional[
        float
    ],  # число акцелераций за 30 минут (можно float, если экстраполировано)
    decel_severity: Optional[
        int
    ] = None,  # 0/1/2 — подготовленный код для "урежения"/децелераций; None -> не учитывать
) -> Tuple[Dict[str, int], int]:
    """
    Возвращает компонентные оценки по Fischer (Fisher-like) и общий балл
    """
    comp: Dict[str, int] = {}

    if baseline_bpm is None or not np.isfinite(baseline_bpm):
        comp["baseline"] = 0
    else:
        if baseline_bpm < 100 or baseline_bpm > 180:
            comp["baseline"] = 0
        elif 100 <= baseline_bpm < 120 or 160 < baseline_bpm <= 180:
            comp["baseline"] = 1
        elif 120 <= baseline_bpm <= 160:
            comp["baseline"] = 2
        else:
            comp["baseline"] = 0

    if osc_amp is None or not np.isfinite(osc_amp):
        comp["osc_amp"] = 0
    else:
        if osc_amp < 5.0:
            comp["osc_amp"] = 0
        elif 5.0 <= osc_amp < 10.0 or osc_amp > 25.0:
            comp["osc_amp"] = 1
        elif 10.0 <= osc_amp <= 25.0:
            comp["osc_amp"] = 2
        else:
            comp["osc_amp"] = 0

    if osc_freq is None or not np.isfinite(osc_freq):
        comp["osc_freq"] = 0
    else:
        if osc_freq < 3.0:
            comp["osc_freq"] = 0
        elif 3.0 <= osc_freq <= 6.0:
            comp["osc_freq"] = 1
        else:
            comp["osc_freq"] = 2

    if num_accel_per_30min is None or not np.isfinite(num_accel_per_30min):
        comp["accels"] = 0
    else:
        if num_accel_per_30min <= 0:
            comp["accels"] = 0
        elif 1 <= num_accel_per_30min <= 4:
            comp["accels"] = 1
        else:
            comp["accels"] = 2

    if decel_severity is None:
        comp["decels"] = 0
    else:
        comp["decels"] = int(np.clip(decel_severity, 0, 2))

    total = sum(comp.values())
    return comp, total


# window


def clean_and_consider_global(
    ts: np.ndarray, fhr_raw: np.ndarray, toco_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    fhr_clean, quality = clean_signal(fhr_raw, ts)
    toco_clean = clean_toco(toco_raw, ts)

    global_baseline, mask = compute_global_baseline(
        fhr_clean, ts, min_event_dur_s=15, osc_std_window_s=30
    )
    return fhr_clean, toco_clean, global_baseline, mask


def window_consider(
    ts: np.ndarray,
    fhr_clean: np.ndarray,
    toco_clean: np.ndarray,
    global_baseline: float,
    mask: np.ndarray,
    ignore_not_reliable: bool = True,
) -> Tuple[float, float, float, float, float, float, List[Contraction], List[Event]]:
    baseline_value, reliable_value = compute_window_baseline(
        fhr_clean, mask, ts, global_baseline
    )
    if not reliable_value and ignore_not_reliable:
        return None

    baseline = np.ones_like(fhr_clean, float) * baseline_value

    stv, ltv, rmssd, amp, freq = variability_metrics_window(
        fhr_clean, ts, baseline_value
    )

    contractions = extract_contractions(toco_clean, ts)
    events = detect_events(
        fhr_clean,
        ts,
        baseline,
        accel_min_dur_s=5,
        decel_min_dur_s=5,
        accel_thr_bpm=10,
        decel_thr_bpm=10,
    )
    events = classify_decelerations_wrt_toco(events, toco_clean, contractions, ts)

    return baseline_value, stv, ltv, rmssd, amp, freq, contractions, events


def consider_global_events_simple(
    ts, fhr_raw, toco_raw
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fhr_clean, toco_clean, baseline_value, _ = clean_and_consider_global(
        ts, fhr_raw, toco_raw
    )
    baseline = np.ones_like(fhr_clean, float) * baseline_value
    events = detect_events(
        fhr_clean,
        ts,
        baseline,
        accel_min_dur_s=5,
        decel_min_dur_s=5,
        accel_thr_bpm=10,
        decel_thr_bpm=10,
    )

    return fhr_clean, toco_clean, events
