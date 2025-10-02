from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import pandas as pd

import app.consts as c
from app.abstract import Meta, Stirring
from app.utils import get_all_data

_mapping = {"time_sec": "ts", "bpm": "fhr", "uterus": "toco"}


class Storage:
    def __init__(self) -> None:
        self.cur = pd.DataFrame(columns=["ts", "fhr", "toco"])
        self.stirrings: List[Stirring] = []
        self._data: Dict[
            str, Tuple[pd.DataFrame, List[Stirring], Optional[Meta]]
        ] = {}  # TODO database later

    def get(
        self, id: str
    ) -> Optional[Tuple[pd.DataFrame, List[Stirring], Optional[Meta]]]:
        return self._data.get(id)

    def put(
        self,
        df: pd.DataFrame,
        stirrings: List[Stirring] = None,
        meta: Optional[Meta] = None,
        id: Optional[str] = None,
    ) -> str:  # may overwrite data
        if stirrings is None:
            stirrings = []
        if id is None:
            id = uuid4().hex
        self._data[id] = (df, stirrings, meta)
        return id

    def ids(self) -> List[str]:
        return list(self._data.keys())

    def put_current(self, id: Optional[str] = None) -> str:  # may overwrite data
        if id is None:
            id = uuid4().hex
        self._data[id] = (self.cur, self.stirrings, None)
        return id

    def delete(
        self, id: str
    ) -> Optional[Tuple[pd.DataFrame, List[Stirring], Optional[Meta]]]:
        return self._data.pop(id)

    def clear(self) -> None:
        self._data.clear()

    def reset(self) -> None:
        self.cur = pd.DataFrame(columns=["ts", "fhr", "toco"])
        self.stirrings = []


base = Storage()


def add_base_data():  # only for demo
    data_hypoxia = get_all_data(c.DATA_HYPOXIA_PATH)
    data_regular = get_all_data(c.DATA_REGULAR_PATH)

    for id, df in data_hypoxia.items():
        df.rename(columns=_mapping, inplace=True)
        base.put(df, id=f"{id}_hypoxia")
    for id, df in data_regular.items():
        df.rename(columns=_mapping, inplace=True)
        base.put(df, id=f"{id}_regular")
