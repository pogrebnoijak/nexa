import asyncio
import json
import os
import random
import sys
import time

import numpy as np
import websockets
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backend.app.consts as c
from backend.app.utils.data import get_data

_prefix = os.path.join(c.DATA_HYPOXIA_PATH, "1")    # change path
_gen_sleep_k = 0.1
id = None


async def run_gen():
    uri = "ws://localhost:8000/ws/generator"
    if id is not None:
        uri += f"?id={id}"
    async with websockets.connect(uri) as ws:
        while True:
            data = {
                "ts": time.time(),
                "fhr": random.randint(110, 160),
                "toco": random.randint(0, 100)
            }
            await ws.send(json.dumps(data))
            await asyncio.sleep(_gen_sleep_k)


async def run_data(real_sleep: bool = True):
    df = get_data(_prefix)
    uri = "ws://localhost:8000/ws/generator"
    if id is not None:
        uri += f"?id={id}"
    async with websockets.connect(uri) as ws:
        last_ts = 0.0
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            ts = row["time_sec"]
            if real_sleep:
                await asyncio.sleep(_gen_sleep_k * (ts - last_ts))
                last_ts = ts
            else:
                await asyncio.sleep(_gen_sleep_k)
            data = {
                "ts": ts,
                "fhr": None if np.isnan(fhr := row["bpm"]) else float(fhr),
                "toco": None if np.isnan(toco := row["uterus"]) else float(toco),
            }
            await ws.send(json.dumps(data))


if __name__ == "__main__":
    # asyncio.run(run_gen())
    asyncio.run(run_data())
