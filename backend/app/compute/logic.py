from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd

import app.consts as c
from app.abstract import Analyze, Predicts
from app.compute.preprocessing import (
    clean_and_consider_global,
    fisher_score,
    window_consider,
)
from app.compute.training import predict_long, predict_short, predict_short_cnn


def parse_models_list(
    models: str,
) -> Set[
    Tuple[str, int]
]:  # models: l or s{horizon_m} or c{horizon_m} separated ":". Example: "s10:c20:c40:l"
    res = set()
    for model in models.split(":"):
        if model == "l":
            res.add((model, None))
        else:
            try:
                horizon_m = int(model[1:])
                res.add((model[0], horizon_m))
            except BaseException:
                pass
    return res


def metrics(
    df: pd.DataFrame,
    with_records: bool = False,
    with_predicts: bool = False,
    *,
    models: Optional[str] = None,
) -> Optional[Analyze]:
    res = {}
    ts_, fhr, toco = df["ts"], df["fhr"], df["toco"]
    ts = np.asarray(ts_)

    # FIXME call functions twice
    fhr_clean, toco_clean, global_baseline, mask = clean_and_consider_global(
        ts, fhr, toco
    )
    if with_records:
        res["ts"] = [float(x) for x in ts]
        res["fhr"] = [None if x is None or np.isnan(x) else float(x) for x in fhr_clean]
        res["toco"] = [
            None if x is None or np.isnan(x) else float(x) for x in toco_clean
        ]
    window_res = window_consider(
        ts, fhr_clean, toco_clean, global_baseline, mask, ignore_not_reliable=False
    )
    if window_res is None:
        return None

    baseline_value, stv, ltv, rmssd, amp, freq, contractions, events = window_res
    res["baseline"] = None if np.isnan(baseline_value) else float(baseline_value)
    res["stv"] = None if np.isnan(stv) else float(stv)
    res["ltv"] = None if np.isnan(ltv) else float(ltv)
    res["rmssd"] = None if np.isnan(rmssd) else float(rmssd)
    res["amp"] = None if np.isnan(amp) else float(amp)
    res["freq"] = None if np.isnan(freq) else float(freq)
    res["num_decel"] = len([e for e in events if e.kind == "decel"])
    res["num_accel"] = len([e for e in events if e.kind == "accel"])
    res["contractions"] = [c.model_dump() for c in contractions]
    res["events"] = [e.model_dump() for e in events]
    if ts[-1] > c.FISHER_MINIMUM:
        start_ts = ts[-1] - c.FISHER_MAX_WINDOW
        num_accel_per_window = len(
            [e for e in events if e.kind == "accel" and e.t_start_s >= start_ts]
        )
        if start_ts < 0:
            num_accel_per_window *= c.FISHER_MAX_WINDOW / ts[-1]
        decel_k = [
            1 if e.toco_rel == "early" else 0 for e in events if e.kind == "decel"
        ]
        decel_severity = 2 if len(decel_k) == 0 else min(decel_k)
        _, score = fisher_score(
            baseline_value, amp, freq, num_accel_per_window, decel_severity
        )
        res["fisher_points"] = score

    res["data_quality"] = 0 if len(mask) == 0 else sum(mask) / len(mask)

    if with_predicts:
        preds = __predicts(ts_, fhr, toco, models=models)
        res.update(preds.model_dump())
    return Analyze(**res)


def predicts(df: pd.DataFrame, *, models: Optional[str] = None) -> Predicts:
    ts_, fhr, toco = df["ts"], df["fhr"], df["toco"]
    return __predicts(ts_, fhr, toco, models=models)


def __predicts(
    ts, fhr, toco, *, models: Optional[str] = None
) -> Predicts:  # TODO improve models, more metrics, force None in start
    res = {}
    with ThreadPoolExecutor() as executor:  # TODO optimize, ONNX + ORT, FP16 / INT8
        futures = {}
        if models is None:
            for horizon_m in c.PREDICTION_SHORT_HORIZON_TO_WINDOW_SIZE_AND_STEP.keys():
                futures[
                    executor.submit(predict_short, ts, fhr, toco, horizon_m=horizon_m)
                ] = ("proba_short", horizon_m)
            for (
                horizon_m
            ) in c.PREDICTION_SHORT_HORIZON_TO_WINDOW_SIZE_AND_STEP_CNN.keys():
                futures[
                    executor.submit(
                        predict_short_cnn, ts, fhr, toco, horizon_m=horizon_m
                    )
                ] = ("proba_short_cnn", horizon_m)
            futures[executor.submit(predict_long, ts, fhr, toco)] = ("proba_long", None)
        else:
            for type, horizon_m in parse_models_list(models):
                if (
                    type == "s"
                    and horizon_m in c.PREDICTION_SHORT_HORIZON_TO_WINDOW_SIZE_AND_STEP
                ):
                    futures[
                        executor.submit(
                            predict_short, ts, fhr, toco, horizon_m=horizon_m
                        )
                    ] = ("proba_short", horizon_m)
                elif (
                    type == "c"
                    and horizon_m
                    in c.PREDICTION_SHORT_HORIZON_TO_WINDOW_SIZE_AND_STEP_CNN
                ):
                    futures[
                        executor.submit(
                            predict_short_cnn, ts, fhr, toco, horizon_m=horizon_m
                        )
                    ] = ("proba_short_cnn", horizon_m)
                elif type == "l":
                    futures[executor.submit(predict_long, ts, fhr, toco)] = (
                        "proba_long",
                        None,
                    )

        res["proba_short"] = {}
        res["proba_short_cnn"] = {}
        res["proba_long"] = None
        if len(futures) > 0:
            for future in as_completed(futures):
                kind, horizon_m = futures[future]
                result = future.result()
                if kind == "proba_long":
                    res[kind] = result
                elif kind == "proba_short":
                    res[kind][horizon_m] = result
                elif kind == "proba_short_cnn":
                    res[kind][horizon_m] = result
    return Predicts(**res)
