from typing import List, Tuple

import numpy as np


def contiguous_spans(mask: np.ndarray) -> List[Tuple[int, int]]:
    if mask.size == 0:
        return []

    transitions = np.where(
        np.diff(np.concatenate(([False], mask, [False])).astype(int)) != 0
    )[0]
    return list(zip(transitions[::2], transitions[1::2]))
