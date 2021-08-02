#!/usr/bin/env python3

import json
import logging
import ssl
import sys
import os
import concurrent.futures
import asyncio

from pathlib import Path
from vosk import KaldiRecognizer, Model
from aiohttp import web
from aiohttp.web_exceptions import HTTPServiceUnavailable
from aiortc import RTCSessionDescription, RTCPeerConnection
from av.audio.resampler import AudioResampler

from asr_env import get_env
env = get_env()
ROOT = Path(__file__).parent

# vosk_interface = env.ip_addr
# vosk_sample_rate = float(os.environ.get("VOSK_SAMPLE_RATE", 8000))

model = Model(env.model_path)
pool = concurrent.futures.ThreadPoolExecutor((os.cpu_count() or 1))
loop = asyncio.get_event_loop()


def process_chunk(recognition, message):
    if recognition.AcceptWaveform(message):
        data = json.loads(recognition.Result())
        if "result" in data:
            return '{"text": "' + data["text"] + '"}'
        return recognition.Result()
    else:
        return recognition.PartialResult()


class KaldiTask:

    def __init__(self, user_connection):
        self.__resampler = AudioResampler(format='s16', layout='mono', rate=41000)
        self.__pc = user_connection
        self.__audio_task = None
        self.__track = None
        self.__channel = None
        self.__recognizer = KaldiRecognizer(model, 41000)

    async def set_audio_track(self, track):
        self.__track = track

    async def set_text_channel(self, channel):
        self.__channel = channel

    async def start(self):
        self.__audio_task = asyncio.create_task(self.__run_audio_xfer())

    async def stop(self):
        if self.__audio_task is not None:
            self.__audio_task.cancel()
            self.__audio_task = None

    async def __run_audio_xfer(self):
        dataframes = bytearray(b"")
        while True:
            frame = await self.__track.recv()
            frame = self.__resampler.resample(frame)
            max_frames_len = 8000
            message = frame.planes[0].to_bytes()
            recv_frames = bytearray(message)
            dataframes += recv_frames
            if len(dataframes) > max_frames_len:
                wave_bytes = bytes(dataframes)
                response = await loop.run_in_executor(pool, process_chunk, self.__recognizer, wave_bytes)
                print(response)
                self.__channel.send(response)
                dataframes = bytearray(b"")


async def index(request):
    with open(str(ROOT / 'static' / 'index.html')) as content:
        return web.Response(content_type='text/html', text=content.read())


async def offer(request):

    params = await request.json()
    rtc_offer = RTCSessionDescription(
        sdp=params['sdp'],
        type=params['type']
    )

    pc = RTCPeerConnection()

    kaldi = KaldiTask(pc)

    @pc.on('datachannel')
    async def on_datachannel(channel):
        """ This is called when the datachannel event occurs on an RTCPeerConnection.
            This event, of type RTCDataChannelEvent, is sent when an RTCDataChannel
            is added to the connection by the remote peer calling createDataChannel()"""
        await kaldi.set_text_channel(channel)
        await kaldi.start()

    @pc.on('iceconnectionstatechange')
    async def on_iceconnectionstatechange():
        """ This called when the iceconnectionstatechange event is fired on an RTCPeerConnection
            instance. This happens when the state of the connection's ICE agent,
            as represented by the iceConnectionState property, changes.
        """
        if pc.iceConnectionState == 'failed':
            await pc.close()

    @pc.on('track')
    async def on_track(track):
        """ This is called when the track event occurs, indicating that a track has been added
            to the RTCPeerConnection.  this event is sent when a new incoming MediaStreamTrack
            has been created and associated with an RTCRtpReceiver object which has been added
            to the set of receivers on connection.
        """
        if track.kind == 'audio':
            await kaldi.set_audio_track(track)

        @track.on('ended')
        async def on_ended():
            await kaldi.stop()

    # Descriptions will be exchanged until the two peers agree on a configuration
    # Create an SDP answer and call setLocalDescription() to set that as the configuration at our end of the call
    # before forwarding that answer to the caller.

    await pc.setRemoteDescription(rtc_offer)
    rtc_answer = await pc.createAnswer()
    await pc.setLocalDescription(rtc_answer)

    return web.Response(
        content_type='application/json',
        text=json.dumps({
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type,
        })
    )


if __name__ == '__main__':

    if env.vosk_cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(env.vosk_cert_file)
    else:
        ssl_context = None

    app = web.Application()
    app.router.add_post('/offer', offer)

    app.router.add_get('/', index)
    app.router.add_static('/static/', path=ROOT / 'static', name='static')

    web.run_app(app, port=env.port, ssl_context=ssl_context)
