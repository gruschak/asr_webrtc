import os
import dataclasses
from typing import Optional


@dataclasses.dataclass
class ASREnv:
    """
    Переменные окружения ASR-сервера
    """
    ip_addr: str            # VOSK-server IP address
    port: int               # VOSK-server IP port
    model_path: str         # VOSK model directory name
    vosk_cert_file: str     # VOSK certificate file in PEM format
    samplerate: float       # VOSK sample rate
    max_alternatives: int
    max_workers: Optional[int]


def get_env() -> ASREnv:

    try:
        max_workers_number = int(os.environ.get("VOSK_MAX_WORKERS"))
    except (TypeError, ValueError):
        max_workers_number = None
    else:
        if max_workers_number < 1:
            max_workers_number = None

    env = ASREnv(
        ip_addr=os.environ.get("VOSK_SERVER_IP_ADDR", "0.0.0.0"),
        port=int(os.environ.get("VOSK_SERVER_PORT", 2700)),
        model_path=os.environ.get("VOSK_MODEL_PATH", "../../model"),
        vosk_cert_file=os.environ.get("VOSK_CERT_FILE"),
        samplerate=float(os.environ.get("VOSK_SAMPLERATE", 41000)),
        max_alternatives=int(os.environ.get("VOSK_ALTERNATIVES", 0)),
        max_workers=max_workers_number,
    )
    return env
