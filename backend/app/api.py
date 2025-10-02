import asyncio
from typing import Optional, Set

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_200_OK,
    HTTP_204_NO_CONTENT,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
)

import app.consts as c
from app.abstract import Analyze, Id, Ids, PatientData, Predicts, Stirring
from app.compute.logic import metrics, predicts
from app.compute.training import load_models
from app.storage import add_base_data, base


class State:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.front_clients: Set[WebSocket] = set()
        self.cur_ws: Optional[WebSocket] = None


app = FastAPI()
s = State()


def init():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,  # Allows cookies and authorization headers
        allow_methods=["*"],  # Allows all standard HTTP methods
        allow_headers=["*"],  # Allows all headers
    )

    load_models()
    if c.IS_DEMO:
        add_base_data()


init()


async def clients_send(data):
    to_remove = []
    for client in s.front_clients:
        try:
            await client.send_json(data)
        except WebSocketDisconnect:
            to_remove.append(client)
    for client in to_remove:
        s.front_clients.remove(client)


@app.get("/ping", status_code=HTTP_200_OK)
async def ping() -> str:
    return "pong"


@app.websocket("/ws/generator")
async def generator_ws(
    ws: WebSocket, id: Optional[str] = Query(None, description="Optional id")
):
    print("id", id)

    await ws.accept()
    async with s.lock:
        if s.cur_ws is not None:
            try:
                await s.cur_ws.close()
            except BaseException:
                pass
        s.cur_ws = ws
        base.reset()

    try:
        while True:
            data = await ws.receive_json()
            if (
                "ts" not in data
                or "fhr" not in data
                or "toco" not in data
                or (ts := data["ts"]) is None
                or np.isnan(ts)
            ):
                print(f"wrong data, ignore: {data}")
                continue

            try:
                data = {
                    "ts": float(data["ts"]),
                    "fhr": None
                    if (fhr := data["fhr"]) is None or np.isnan(fhr)
                    else float(fhr),
                    "toco": None
                    if (toco := data["toco"]) is None or np.isnan(toco)
                    else float(toco),
                }
            except BaseException:
                print(f"wrong data, ignore: {data}")
                continue

            base.cur.loc[len(base.cur)] = [data["ts"], data["fhr"], data["toco"]]
            await clients_send(data)
    except WebSocketDisconnect:
        base.put_current(id)
        async with s.lock:
            if s.cur_ws == ws:
                s.cur_ws = None
        for client in s.front_clients:
            await client.close()
        s.front_clients.clear()


@app.websocket("/stream/ctg")
async def frontend_ws(ws: WebSocket):
    await ws.accept()

    # TODO write more carefully so that the meaning is not lost
    for _, row in base.cur.iterrows():
        try:
            await ws.send_json(row.to_dict())
        except WebSocketDisconnect:
            return
    if s.cur_ws is None:
        await ws.close()
        return

    s.front_clients.add(ws)
    try:
        while True:
            await asyncio.sleep(c.SLEEP_WS)
    except WebSocketDisconnect:
        s.front_clients.remove(ws)


# id: Optional[str] = Query(None, description="Optional id")
@app.get("/analyze", status_code=HTTP_200_OK)
async def analyze(
    records: bool = Query(False, description="return records or not"),
    predicts: Optional[bool] = Query(False, description="return predicts or not"),
    models: Optional[str] = None,
) -> Analyze:
    if base.cur.empty:
        return JSONResponse({"error": "no data"}, status_code=HTTP_400_BAD_REQUEST)

    res = metrics(base.cur, with_records=records, with_predicts=predicts, models=models)
    if res is None:
        return JSONResponse({"error": "bad data"}, status_code=HTTP_400_BAD_REQUEST)
    res.stirrings = base.stirrings
    return res


@app.get("/analyze/ids/{id}", status_code=HTTP_200_OK)
async def analyze_id(id: str, models: Optional[str] = None) -> Analyze:
    data = base.get(id)
    if data is None:
        return JSONResponse({"error": "wrong id"}, status_code=HTTP_404_NOT_FOUND)

    df, stirring, meta = data
    res = metrics(df, with_records=True, with_predicts=True, models=models)
    if res is None:
        return JSONResponse({"error": "bad data"}, status_code=HTTP_400_BAD_REQUEST)
    res.stirrings = stirring
    res.meta = meta
    return res


@app.get("/analyze/ids", status_code=HTTP_200_OK)
async def analyze_ids() -> Ids:
    return Ids(data=base.ids())


@app.post("/analyze/ids", status_code=HTTP_200_OK)
async def analyze_ids_post(data: PatientData) -> Id:
    if len(data.ts) != len(data.fhr) or len(data.ts) != len(data.toco):
        return JSONResponse(
            {"error": "wrong data sizes"}, status_code=HTTP_400_BAD_REQUEST
        )

    df = pd.DataFrame(
        {
            "ts": data.ts,
            "fhr": data.fhr,
            "toco": data.toco,
        }
    )
    id = base.put(df, data.stirrings, data.meta)
    return Id(id=id)


@app.delete("/analyze/ids/{id}", status_code=HTTP_204_NO_CONTENT)
async def analyze_id_delete(id: str):
    base.delete(id)


@app.delete("/analyze/ids", status_code=HTTP_204_NO_CONTENT)
async def analyze_ids_delete():
    base.clear()


@app.get("/analyze/predicts", status_code=HTTP_200_OK)
async def analyze_predicts(
    models: Optional[str] = None,
) -> Predicts:
    if base.cur.empty:
        return JSONResponse({"error": "no data"}, status_code=HTTP_400_BAD_REQUEST)

    res = predicts(base.cur, models=models)
    if res is None:
        return JSONResponse({"error": "bad data"}, status_code=HTTP_400_BAD_REQUEST)
    return res


@app.get("/analyze/predicts/{id}", status_code=HTTP_200_OK)
async def analyze_predicts_id(id: str, models: Optional[str] = None) -> Predicts:
    data = base.get(id)
    if data is None:
        return JSONResponse({"error": "wrong id"}, status_code=HTTP_404_NOT_FOUND)

    df, _, _ = data
    res = predicts(df, models=models)
    if res is None:
        return JSONResponse({"error": "bad data"}, status_code=HTTP_400_BAD_REQUEST)
    return res


@app.post("/stirring", status_code=HTTP_204_NO_CONTENT)
async def stirring_post(data: Stirring):
    if s.cur_ws is None:
        return JSONResponse(
            {"error": "data generator is not active"}, status_code=HTTP_400_BAD_REQUEST
        )
    base.stirrings.append(data)
