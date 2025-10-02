from typing import Dict, Optional

import joblib
import pandas as pd
import torch
from sklearn.pipeline import Pipeline

import app.consts as c
from app.compute.prediction import (
    build_dataset,
    build_longterm_dataset,
    predict_latest_for_record,
    predict_longterm_for_record,
    train_longterm_gbm_on_dataset,
    train_rf_on_dataset,
)
from app.compute.prediction_cnn_gru import (
    CNN_GRU_Model,
    build_sequence_dataset,
    predict_on_data,
    train_cnn_on_dataset,
)
from app.compute.preprocessing import consider_global_events_simple, interpolate_signals
from app.utils import get_all_data


class Models:
    def __init__(self) -> None:
        self.short_decel: Dict[int, Pipeline] = {}  # horizon_m -> model
        self.short_decel_cnn: Dict[int, CNN_GRU_Model] = {}  # horizon_m -> model
        self.long: Pipeline = None


models = Models()


def save_torch_model(model: torch.nn.Module, path: str) -> None:
    """Сохраняет веса модели в файл"""
    torch.save(model.state_dict(), path)


def load_torch_model(model_class, path: str, *args, **kwargs) -> torch.nn.Module:
    """
    Загружает веса модели в новый экземпляр.

    Args:
        model_class: класс модели (например MyModel)
        path: путь к файлу .pt/.pth
        *args, **kwargs: параметры для инициализации model_class
    """
    model = model_class(*args, **kwargs)
    state = torch.load(path, map_location=c.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def save_pipeline_model(pipeline: Pipeline, path: str) -> None:
    """
    Сохраняет sklearn Pipeline в файл.
    """
    joblib.dump(pipeline, path)


def load_pipeline_model(path: str) -> Pipeline:
    """
    Загружает sklearn Pipeline из файла.
    """
    return joblib.load(path)


def predict_short(
    ts, fhr, toco, horizon_m: int = None
) -> Optional[float]:  # horizon_m - горизонт предсказания в минутах
    if horizon_m is None:
        horizon_m = next(iter(c.PREDICTION_SHORT_HORIZON_TO_WINDOW_SIZE_AND_STEP))

    model = models.short_decel.get(horizon_m)
    if model is None:
        return None

    window_size_m, _ = c.PREDICTION_SHORT_HORIZON_TO_WINDOW_SIZE_AND_STEP[horizon_m]
    res = predict_latest_for_record(
        model, ts, fhr, toco, window_size_s=c.MINUTE * window_size_m
    )
    return res["proba"]


def predict_short_cnn(ts, fhr, toco, horizon_m: int = None) -> Optional[float]:
    if horizon_m is None:
        horizon_m = next(iter(c.PREDICTION_SHORT_HORIZON_TO_WINDOW_SIZE_AND_STEP_CNN))

    model = models.short_decel_cnn.get(horizon_m)
    if model is None:
        return None

    window_size_m, _ = c.PREDICTION_SHORT_HORIZON_TO_WINDOW_SIZE_AND_STEP_CNN[horizon_m]
    return predict_on_data(
        model, ts, fhr, toco, window_size_s=c.MINUTE * window_size_m, device=c.DEVICE
    )


def predict_long(ts, fhr, toco) -> Optional[float]:
    res = predict_longterm_for_record(
        models.long,
        {"patient_id": "#", "ts": ts, "fhr": fhr, "toco": toco},
        window_size_s=c.PREDICTION_LONG_WINDOW_SIZE,
        step_s=c.PREDICTION_LONG_STEP,
    )
    return res["proba"]


def load_models():
    for horizon_m in c.PREDICTION_SHORT_HORIZON_TO_WINDOW_SIZE_AND_STEP.keys():
        models.short_decel[horizon_m] = load_pipeline_model(
            c.SHORT_ASSET_PATH_F % horizon_m
        )
    for horizon_m in c.PREDICTION_SHORT_HORIZON_TO_WINDOW_SIZE_AND_STEP_CNN.keys():
        models.short_decel_cnn[horizon_m] = load_torch_model(
            CNN_GRU_Model, c.SHORT_CNN_ASSET_PATH_F % horizon_m
        )
    models.long = load_pipeline_model(c.LONG_ASSET_PATH)


def train_short(
    data: Dict[str, pd.DataFrame],
    window_size_m: int,
    step_m: int,
    horizon_m: int,
    save: bool = True,
) -> Pipeline:
    records = []
    for id, record in data.items():
        records.append(
            {
                "patient_id": id,
                "ts": record["time_sec"],
                "fhr": record["bpm"],
                "toco": record["uterus"],
            }
        )
    dataset = build_dataset(
        records,
        window_size_s=c.MINUTE * window_size_m,
        step_s=c.MINUTE * step_m,
        horizon_s=c.MINUTE * horizon_m,
    )
    model, _, _ = train_rf_on_dataset(dataset)
    models.short_decel[horizon_m] = model
    if save:
        save_pipeline_model(model, c.SHORT_ASSET_PATH_F % horizon_m)
    return model


def train_short_cnn(
    data: Dict[str, pd.DataFrame],
    window_size_m: int,
    step_m: int,
    horizon_m: int,
    save: bool = True,
) -> CNN_GRU_Model:
    records = []
    for id, record in data.items():
        ts = record["time_sec"]
        fhr_clean, toco_clean, events = consider_global_events_simple(
            ts, record["bpm"], record["uterus"]
        )
        interpolated = interpolate_signals(ts, fhr=fhr_clean, toco=toco_clean)
        records.append(
            {
                "patient_id": id,
                "ts": ts,
                "fhr": interpolated["fhr"],
                "toco": interpolated["toco"],
                "events": events,
            }
        )
    dataset, y, groups = build_sequence_dataset(
        records,
        window_size_s=c.MINUTE * window_size_m,
        step_s=c.MINUTE * step_m,
        horizon_s=c.MINUTE * horizon_m,
    )
    model, _ = train_cnn_on_dataset(
        dataset, y, groups, epochs=c.PREDICTION_SHORT_CNN_EPOCHS, device=c.DEVICE
    )
    models.short_decel_cnn[horizon_m] = model
    if save:
        save_torch_model(model, c.SHORT_CNN_ASSET_PATH_F % horizon_m)
    return model


def train_long(
    data_hypoxia: Dict[str, pd.DataFrame],
    data_regular: Dict[str, pd.DataFrame],
    save: bool = True,
) -> Pipeline:
    records = []
    for id, record in data_hypoxia.items():
        records.append(
            {
                "patient_id": id,
                "ts": record["time_sec"],
                "fhr": record["bpm"],
                "toco": record["uterus"],
                "label": 1,
            }
        )
    for id, record in data_regular.items():
        records.append(
            {
                "patient_id": id,
                "ts": record["time_sec"],
                "fhr": record["bpm"],
                "toco": record["uterus"],
                "label": 0,
            }
        )
    dataset, y, groups = build_longterm_dataset(
        records,
        window_size_s=c.PREDICTION_LONG_WINDOW_SIZE,
        step_s=c.PREDICTION_LONG_STEP,
    )
    model, _, _ = train_longterm_gbm_on_dataset(dataset, y, groups)
    models.long = model
    if save:
        save_pipeline_model(model, c.LONG_ASSET_PATH)
    return model


def train_all(save: bool = True):
    data_hypoxia = get_all_data(c.DATA_HYPOXIA_PATH)
    data_regular = get_all_data(c.DATA_REGULAR_PATH)
    data = data_hypoxia | data_regular

    for horizon_m, (
        window_size_m,
        step_m,
    ) in c.PREDICTION_SHORT_HORIZON_TO_WINDOW_SIZE_AND_STEP.items():
        print(f"start training short, horizon_m={horizon_m}")
        train_short(data, window_size_m, step_m, horizon_m, save=save)
        print(f"finish training short, horizon_m={horizon_m}")
    for horizon_m, (
        window_size_m,
        step_m,
    ) in c.PREDICTION_SHORT_HORIZON_TO_WINDOW_SIZE_AND_STEP_CNN.items():
        print(f"start training short_cnn, horizon_m={horizon_m}")
        train_short_cnn(data, window_size_m, step_m, horizon_m, save=save)
        print(f"finish training short_cnn, horizon_m={horizon_m}")
    print("start training long")
    train_long(data_hypoxia, data_regular, save=True)
    print("finish training long")
