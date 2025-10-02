import asyncio
import json

import aiohttp
import websockets

_analyze_sleep = 10
stop_event = asyncio.Event()


async def ctg():
    uri = "ws://localhost:8000/stream/ctg"
    async with websockets.connect(uri) as ws:
        try:
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                if len(data) == 0:
                    print("clear")
                else:
                    print("get:", data)
        except websockets.ConnectionClosed:
            print("close")
            stop_event.set()


async def analyze():
    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            async with session.get("http://localhost:8000/analyze", params={"records": 0, "predicts": 1}, json=None) as response:
                response.raise_for_status()
                if response.ok:
                    result = await response.json()
                    print(result)
            await asyncio.sleep(_analyze_sleep)


async def run():
    await asyncio.gather(ctg(), analyze())


if __name__ == "__main__":
    asyncio.run(run())
