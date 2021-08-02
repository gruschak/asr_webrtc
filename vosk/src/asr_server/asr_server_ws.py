#!/usr/bin/env python3

import os
import asyncio
import websockets
import concurrent.futures
import functools
import logging
from typing import Tuple, Any
from vosk import Model, KaldiRecognizer
from asr_env import get_env

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def process_chunk(rec, message) -> Tuple[Any, bool]:
    if message == '{"eof" : 1}':
        return rec.FinalResult(), True
    elif rec.AcceptWaveform(message):
        return rec.Result(), False
    else:
        return rec.PartialResult(), False


async def accept_connection(websocket, path, recognizer, pool, connected):

    logging.info(f'Connection from {websocket.remote_address}')
    connected.add(websocket)

    try:

        loop = asyncio.get_running_loop()
        while True:
            message = await websocket.recv()
            response, stop = await loop.run_in_executor(pool, process_chunk, recognizer, message)
            for ws in connected:
                await ws.send(response)

            if stop:
                break

    except websockets.ConnectionClosed as e:
        logging.info(f'Lost connection {websocket.remote_address}: {e}')
    finally:
        connected.remove(websocket)


def start_ws_server():

    # Gpu part, uncomment if vosk-api has gpu support
    #
    # from vosk import GpuInit, GpuInstantiate
    # GpuInit()
    # def thread_init():
    #     GpuInstantiate()
    # pool = concurrent.futures.ThreadPoolExecutor(initializer=thread_init)

    env = get_env()
    model = Model(env.model_path)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=env.max_workers)
    loop = asyncio.get_event_loop()
    connected = set()

    # Create the recognizer
    recognizer = KaldiRecognizer(model, env.samplerate)
    recognizer.SetMaxAlternatives(env.max_alternatives)

    handle_connection = functools.partial(accept_connection, recognizer=recognizer, pool=pool, connected=connected)
    start_server = websockets.serve(handle_connection, env.ip_addr, env.port)

    logging.info(f'Listening on {env.ip_addr}:{env.port}')
    loop.run_until_complete(start_server)
    loop.run_forever()


if __name__ == "__main__":
    start_ws_server()
