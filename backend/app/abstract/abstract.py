from typing import Dict, List, Optional

from pydantic import BaseModel

Stirring = Dict
Meta = Dict[str, str]


class Event(BaseModel):
    kind: str  # accel, decel, tachy, brady
    t_start: int  # index
    t_end: int  # index
    toco_rel: Optional[str]  # artifact, variable, early, late

    t_start_s: float
    duration_s: float
    min_val: Optional[float]
    max_val: Optional[float]
    depth: Optional[float]
    area: Optional[float]
    t_nadir: Optional[int]


class Contraction(BaseModel):  # indices
    start: int
    peak: int
    end: int


class Predicts(BaseModel):
    proba_short: Optional[Dict[int, Optional[float]]] = None
    proba_short_cnn: Optional[Dict[int, Optional[float]]] = None
    proba_long: Optional[float] = None


class Analyze(Predicts):
    # statistic
    baseline: Optional[float]
    stv: Optional[float]
    ltv: Optional[float]
    rmssd: Optional[float]
    amp: Optional[float]
    freq: Optional[float]
    num_decel: int
    num_accel: int
    contractions: List[Contraction]
    events: List[Event]
    fisher_points: Optional[int] = None

    # quality
    data_quality: float

    # records
    ts: Optional[List[float]] = None
    fhr: Optional[List[Optional[float]]] = None
    toco: Optional[List[Optional[float]]] = None

    # other
    stirrings: List[Stirring] = []
    meta: Optional[Meta] = None


class Ids(BaseModel):
    data: List[str]


class Id(BaseModel):
    id: str


class PatientData(BaseModel):
    ts: List[float]
    fhr: List[Optional[float]]
    toco: List[Optional[float]]
    stirrings: List[Stirring] = None
    meta: Meta = {}
