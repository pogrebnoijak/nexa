from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import app.consts as c
from app.abstract import Event
from app.compute.preprocessing import clean_and_consider_global, window_consider

try:
    import lightgbm as lgb

    GB_CLASS = lgb.LGBMClassifier
except BaseException:
    from sklearn.ensemble import GradientBoostingClassifier

    GB_CLASS = GradientBoostingClassifier

FEATURE_COLS = [
    "baseline",
    # basic FHR stats
    "fhr_mean",
    "fhr_median",
    "fhr_std",
    "fhr_min",
    "fhr_max",
    "fhr_last",
    "fhr_slope",
    "fhr_iqr",
    # variability
    "stv",
    "ltv",
    "rmssd",
    "amp",
    "freq",
    # decel/accel related (from window_consider.events)
    "num_decel",
    "num_accel",
    "avg_decel_depth",
    "max_decel_depth",
    "avg_decel_dur",
    "max_decel_dur",
    "sum_decel_area",  # if area available in Event
    "recent_decel_fraction",  # portion of window spent in decel
    # time since last decel (global)
    "time_since_last_decel_s",
    # contraction / toco
    "num_contractions",
    "toco_mean",
    "toco_std",
    # signal quality / missing
    "nan_fraction",
]


def compute_window_features_from_window(
    ts: np.ndarray,
    fhr_clean: np.ndarray,
    toco_clean: np.ndarray,
    mask: np.ndarray,
    global_baseline: float,
    window_start: float,
    window_end: float,
    prev_events: Optional[List[Event]] = None,
) -> Optional[Dict]:
    """
    Возвращает dict из признаков (keys = FEATURE_COLS) для данного окна.
    Если window_consider вернул None (ненадёжный baseline), возвращает None.
    prev_events: список всех Event для всей записи (можно получить вызовом window_consider на всей записи)
    """
    sel = (ts >= window_start) & (ts <= window_end)
    if sel.sum() < 2:
        return None
    ts_w = ts[sel]
    fhr_w = fhr_clean[sel]
    toco_w = toco_clean[sel] if toco_clean is not None else np.array([])
    mask_w = mask[sel] if mask is not None else np.ones_like(fhr_w, dtype=bool)

    res = window_consider(ts_w, fhr_w, toco_w, global_baseline, mask_w)
    if res is None:
        return None
    baseline_value, stv, ltv, rmssd, amp, freq, contractions, events = res

    fhr_valid_idx = ~np.isnan(fhr_w)
    fhr_valid = fhr_w[fhr_valid_idx]
    out = {}
    out["baseline"] = baseline_value
    out["fhr_mean"] = float(np.nanmean(fhr_w)) if fhr_valid.size > 0 else np.nan
    out["fhr_median"] = float(np.nanmedian(fhr_w)) if fhr_valid.size > 0 else np.nan
    out["fhr_std"] = float(np.nanstd(fhr_w)) if fhr_valid.size > 0 else np.nan
    out["fhr_min"] = float(np.nanmin(fhr_w)) if fhr_valid.size > 0 else np.nan
    out["fhr_max"] = float(np.nanmax(fhr_w)) if fhr_valid.size > 0 else np.nan
    out["fhr_last"] = (
        float(fhr_w[fhr_valid_idx.nonzero()[0][-1]]) if fhr_valid.size > 0 else np.nan
    )

    if fhr_valid.size > 1:
        x = ts_w[fhr_valid_idx] - ts_w[fhr_valid_idx][0]
        y = fhr_valid
        try:
            m, _ = np.polyfit(x, y, 1)
            out["fhr_slope"] = float(m)
        except Exception:
            out["fhr_slope"] = np.nan
    else:
        out["fhr_slope"] = np.nan

    if fhr_valid.size > 0:
        q75, q25 = np.nanpercentile(fhr_valid, [75, 25])
        out["fhr_iqr"] = float(q75 - q25)
    else:
        out["fhr_iqr"] = np.nan

    out["stv"] = float(stv) if stv is not None else np.nan
    out["ltv"] = float(ltv) if ltv is not None else np.nan
    out["rmssd"] = float(rmssd) if ltv is not None else np.nan
    out["amp"] = float(amp) if ltv is not None else np.nan
    out["freq"] = float(freq) if ltv is not None else np.nan

    decel_events = [e for e in events if e.kind == "decel"]
    accel_events = [e for e in events if e.kind == "accel"]
    out["num_decel"] = len(decel_events)
    out["num_accel"] = len(accel_events)

    if decel_events:
        depths = [float(e.depth) for e in decel_events if e.depth is not None]
        durs = [float(e.duration_s) for e in decel_events if e.duration_s is not None]
        areas = [float(e.area) for e in decel_events if e.area is not None]
        out["avg_decel_depth"] = float(np.nanmean(depths)) if depths else 0.0
        out["max_decel_depth"] = float(np.nanmax(depths)) if depths else 0.0
        out["avg_decel_dur"] = float(np.nanmean(durs)) if durs else 0.0
        out["max_decel_dur"] = float(np.nanmax(durs)) if durs else 0.0
        out["sum_decel_area"] = float(np.nansum(areas)) if areas else 0.0
        total_decel_time = sum(durs)
    else:
        out["avg_decel_depth"] = 0.0
        out["max_decel_depth"] = 0.0
        out["avg_decel_dur"] = 0.0
        out["max_decel_dur"] = 0.0
        out["sum_decel_area"] = 0.0
        total_decel_time = 0.0

    window_duration = float(ts_w[-1] - ts_w[0]) if len(ts_w) > 1 else 0.0
    out["recent_decel_fraction"] = (
        (total_decel_time / window_duration) if window_duration > 0 else 0.0
    )

    if prev_events and len(prev_events) > 0:
        prior_decel = [
            e
            for e in prev_events
            if e.kind == "decel" and e.t_start_s <= float(window_end)
        ]
        if prior_decel:
            last = prior_decel[-1]
            out["time_since_last_decel_s"] = float(window_end - last.t_start_s)
        else:
            out["time_since_last_decel_s"] = c.INF
    else:
        out["time_since_last_decel_s"] = c.INF

    if toco_w is not None and toco_w.size > 0:
        out["toco_mean"] = float(np.nanmean(toco_w))
        out["toco_std"] = float(np.nanstd(toco_w))
        out["num_contractions"] = len(contractions)
    else:
        out["toco_mean"] = np.nan
        out["toco_std"] = np.nan
        out["num_contractions"] = 0

    out["nan_fraction"] = float(np.isnan(fhr_w).mean())

    for k in FEATURE_COLS:
        if k not in out:
            out[k] = np.nan
    return out


def build_dataset_from_record(
    ts_raw: np.ndarray,
    fhr_raw: np.ndarray,
    toco_raw: np.ndarray,
    patient_id: Optional[str] = None,
    window_size_s: int = 600,
    step_s: int = 60,
    horizon_s: int = 600,
) -> pd.DataFrame:
    """
    Возвращает DataFrame: каждая строка — окно со всеми признаками + метка 'y'
    y = 1 если в (window_end, window_end + horizon_s] начинается new decel (по глобально найденным events)
    """
    ts = np.asarray(ts_raw)

    fhr_clean, toco_clean, global_baseline, mask = clean_and_consider_global(
        ts, fhr_raw, toco_raw
    )

    full_res = window_consider(
        ts, fhr_clean, toco_clean, global_baseline, mask
    )  # global
    if full_res is None:
        return pd.DataFrame(columns=["patient_id", "window_end", "y"] + FEATURE_COLS)
    _, _, _, _, _, _, _, events_all = full_res
    decel_starts = [e.t_start_s for e in events_all if e.kind == "decel"]

    rows = []
    t_min, t_max = ts[0], ts[-1]
    cur_end = t_min + window_size_s
    while cur_end + horizon_s <= t_max:
        window_start = cur_end - window_size_s
        feats = compute_window_features_from_window(
            ts=ts,
            fhr_clean=fhr_clean,
            toco_clean=toco_clean,
            mask=mask,
            global_baseline=global_baseline,
            window_start=window_start,
            window_end=cur_end,
            prev_events=events_all,
        )
        if feats is None:
            cur_end += step_s
            continue

        label = 0
        for s in decel_starts:
            if s > cur_end and s <= cur_end + horizon_s:
                label = 1
                break

        row = {"patient_id": patient_id, "window_end": float(cur_end), "y": int(label)}
        row.update(feats)
        rows.append(row)
        cur_end += step_s

    if not rows:
        return pd.DataFrame(columns=["patient_id", "window_end", "y"] + FEATURE_COLS)
    return pd.DataFrame(rows)


def build_dataset(
    records: List[Dict],
    window_size_s: int = 600,
    step_s: int = 60,
    horizon_s: int = 600,
) -> pd.DataFrame:
    """
    records: list of dicts {'patient_id': str, 'ts': np.ndarray, 'fhr': np.ndarray, 'toco': np.ndarray}
    """
    dfs = []
    for rec in records:
        pid = rec.get("patient_id", None)
        ts = rec["ts"]
        fhr = rec["fhr"]
        toco = rec["toco"]
        ds = build_dataset_from_record(
            ts,
            fhr,
            toco,
            patient_id=pid,
            window_size_s=window_size_s,
            step_s=step_s,
            horizon_s=horizon_s,
        )
        if not ds.empty:
            dfs.append(ds)
    if not dfs:
        return pd.DataFrame(columns=["patient_id", "window_end", "y"] + FEATURE_COLS)
    return pd.concat(dfs, ignore_index=True)


def train_rf_on_dataset(
    df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
    group_col: str = "patient_id",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, pd.DataFrame, Dict]:
    """
    Возвращает:
     - trained pipeline
     - test DataFrame (subset of df with predicted proba/pred columns added)
     - eval dict (auc, ap, brier, tp/tn/fp/fn)
    """
    if df.empty:
        raise ValueError("Empty dataset: nothing to train on")

    df.reset_index(drop=True, inplace=True)
    groups = df[group_col].fillna("__nogroup__").to_numpy()
    X = df[feature_cols].to_numpy()
    y = df["y"].to_numpy().astype(int)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train_df, X_test_df = df.iloc[train_idx], df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train_df[feature_cols], y_train)

    tr = X_test_df[feature_cols]
    probs = pipeline.predict_proba(tr)[:, 1]
    preds = (probs >= c.SCORE_BOUNDARY).astype(int)

    res_df = X_test_df.copy()
    res_df["y_pred_proba"] = probs
    res_df["y_pred"] = preds
    res_df["y_true"] = y_test

    eval_res = evaluate_predictions(
        res_df["y_true"].to_numpy(), res_df["y_pred_proba"].to_numpy()
    )
    return pipeline, res_df, eval_res


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    res = {}
    try:
        res["auc"] = (
            float(roc_auc_score(y_true, y_score))
            if len(np.unique(y_true)) > 1
            else np.nan
        )
    except Exception:
        res["auc"] = np.nan
    res["ap"] = (
        float(average_precision_score(y_true, y_score))
        if len(np.unique(y_true)) > 1
        else np.nan
    )
    res["brier"] = float(brier_score_loss(y_true, y_score))

    y_pred = (y_score >= c.SCORE_BOUNDARY).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except Exception:
        tn = fp = fn = tp = 0
    res.update({"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)})
    return res


def predict_latest_for_record(
    pipeline: Pipeline,
    ts_raw: np.ndarray,
    fhr_raw: np.ndarray,
    toco_raw: np.ndarray,
    window_size_s: int = 600,
    feature_cols: List[str] = FEATURE_COLS,
) -> Dict:
    """
    Возвращает {'proba': float, 'pred': int, 'features': dict}
    """
    ts = np.asarray(ts_raw)
    fhr_clean, toco_clean, global_baseline, mask = clean_and_consider_global(
        ts, fhr_raw, toco_raw
    )
    full_res = window_consider(ts, fhr_clean, toco_clean, global_baseline, mask)
    events_all = full_res[7] if full_res is not None else []

    window_end = float(ts[-1])
    window_start = window_end - window_size_s
    feats = compute_window_features_from_window(
        ts,
        fhr_clean,
        toco_clean,
        mask,
        global_baseline,
        window_start,
        window_end,
        prev_events=events_all,
    )
    if feats is None:
        return {"proba": None, "pred": None, "features": None}

    feat_df = pd.DataFrame([feats])[feature_cols]
    proba = float(pipeline.predict_proba(feat_df)[:, 1][0])
    pred = int(proba >= c.SCORE_BOUNDARY)
    return {"proba": proba, "pred": pred, "features": feats}


#


def aggregate_windows_to_episode(
    windows_df: pd.DataFrame, feature_cols: List[str]
) -> Optional[Dict]:
    """
    windows_df: DataFrame, как вернёт build_dataset_from_record (строки — окна, cols include feature_cols и 'window_end' если нужно)
    feature_cols: список колонок с фичами для агрегирования (те же FEATURE_COLS)
    Возвращает dict с агрегированными признаками или None, если windows_df пуст.
    Для каждой фичи делает: mean, std, min, max, trend (slope).
    Также добавляет: n_windows, frac_windows_with_decel (если есть 'num_decel').
    """
    if windows_df is None or windows_df.empty:
        return None

    out = {}
    vals = windows_df[feature_cols]

    if "window_end" in windows_df.columns:
        vals = vals.copy()
        vals["__order"] = windows_df["window_end"]
        vals = vals.sort_values("__order").drop(columns="__order")

    n_windows = len(vals)
    out["n_windows"] = int(n_windows)

    for col in feature_cols:
        col_arr = vals[col].to_numpy(dtype=float)
        mask = ~np.isnan(col_arr)
        if mask.sum() == 0:
            out[f"{col}_mean"] = np.nan
            out[f"{col}_std"] = np.nan
            out[f"{col}_min"] = np.nan
            out[f"{col}_max"] = np.nan
            out[f"{col}_trend"] = np.nan
        else:
            arr = col_arr[mask]
            out[f"{col}_mean"] = float(np.mean(arr))
            out[f"{col}_std"] = float(np.std(arr))
            out[f"{col}_min"] = float(np.min(arr))
            out[f"{col}_max"] = float(np.max(arr))
            idxs = np.arange(len(col_arr))[mask]
            if len(idxs) >= 2:
                slope = np.polyfit(idxs, arr, 1)[0]
                out[f"{col}_trend"] = float(slope)
            else:
                out[f"{col}_trend"] = 0.0

    if "num_decel" in vals.columns:
        dec_mask = vals["num_decel"].to_numpy() > 0
        out["frac_windows_with_decel"] = float(np.sum(dec_mask) / max(1, n_windows))
    else:
        out["frac_windows_with_decel"] = np.nan

    return out


def build_longterm_dataset(
    records: List[Dict],
    window_size_s: int = 1200,  # 20 min
    step_s: int = 60,
    feature_cols: List[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    records: список dict {'ts','fhr','toco','patient_id','label'}
    window_size_s: фиксированное окно в минутах (по умолчанию 20)
    step_s: шаг для нарезки окон (передаётся в build_dataset_from_record)
    feature_cols: список фич (по умолчанию FEATURE_COLS)
    Возвращает: (X_df, y_array, groups_array)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    rows = []
    labels = []
    groups = []

    for rec in records:
        ts = rec["ts"]
        fhr = rec["fhr"]
        toco = rec.get("toco", np.array([]))
        pid = rec.get("patient_id", None)
        label = rec.get("label", None)

        windows_df = build_dataset_from_record(
            ts, fhr, toco, pid, window_size_s=window_size_s, step_s=step_s, horizon_s=0
        )

        agg = aggregate_windows_to_episode(windows_df, feature_cols)
        if agg is None:
            continue

        rows.append(agg)
        labels.append(int(label) if label is not None else np.nan)
        groups.append(pid)

    if not rows:
        return pd.DataFrame(), np.array([]), np.array([])

    X_df = pd.DataFrame(rows)
    return X_df.reset_index(drop=True), np.array(labels), np.array(groups)


def train_longterm_gbm_on_dataset(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Обучает GBM (LightGBM если есть, иначе sklearn's GradientBoosting).
    Возвращает: pipeline, res_df (test rows with proba/pred), eval_res (via evaluate_predictions).
    """
    if df.empty:
        raise ValueError("Empty dataset: nothing to train on")
    if feature_cols is None:
        feature_cols = list(df.columns)
    X = df[feature_cols].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GB_CLASS(random_state=random_state)),
        ]
    )
    pipeline.fit(X_train[feature_cols], y_train)

    tr = X_test[feature_cols]
    proba_all = pipeline.predict_proba(tr)  # (n, n_classes)
    clf = pipeline.named_steps["clf"]
    classes = list(clf.classes_)
    if 1 in classes:
        idx_pos = classes.index(1)
        proba_pos = proba_all[:, idx_pos]
    else:
        proba_pos = np.zeros(proba_all.shape[0], dtype=float)

    preds = (proba_pos >= c.SCORE_BOUNDARY).astype(int)
    res_df = X_test.copy().reset_index(drop=True)
    res_df["y_true"] = y_test
    res_df["y_pred_proba"] = proba_pos
    res_df["y_pred"] = preds

    eval_res = evaluate_predictions(
        res_df["y_true"].to_numpy(), res_df["y_pred_proba"].to_numpy()
    )
    return pipeline, res_df, eval_res


def predict_longterm_for_record(
    pipeline,
    record: Dict,
    window_size_s: int = 1200,
    step_s: int = 300,
    feature_cols: List[str] = None,
) -> Dict:
    """
    Возвращает {'proba': float, 'pred': int, 'features': dict}
    """
    X_df, _, _ = build_longterm_dataset(
        [record], window_size_s=window_size_s, step_s=step_s, feature_cols=feature_cols
    )
    if X_df.empty:
        return {"proba": None, "pred": None, "features": None}

    proba_all = pipeline.predict_proba(X_df)
    clf = pipeline.named_steps["clf"]
    classes = list(clf.classes_)
    if 1 in classes:
        idx_pos = classes.index(1)
        proba_pos = float(proba_all[:, idx_pos][0])
    else:
        proba_pos = 0.0

    pred = int(proba_pos >= c.SCORE_BOUNDARY)
    return {"proba": proba_pos, "pred": pred, "features": X_df.iloc[0].to_dict()}
