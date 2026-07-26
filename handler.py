# handler.py — place at repo root
import runpod
import base64
import asyncio
from starlette.testclient import TestClient
import subprocess, time, requests

vllm_process = subprocess.Popen([
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "kenpath/svara-tts-v1",
    "--port", "8000",
])

# wait for vLLM to be ready before accepting jobs
for _ in range(60):
    try:
        if requests.get("http://localhost:8000/health").status_code == 200:
            break
    except requests.exceptions.ConnectionError:
        pass
    time.sleep(2)

# ... rest of handler.py as before ...
# Import the existing FastAPI app — reuses all the model-loading,
# vLLM calls, and SNAC decoding logic already built into api/server.py
from api.server import app

# TestClient triggers FastAPI's startup events (model load) once,
# at import time — this happens when the worker cold-starts, not per-request.
client = TestClient(app)

def handler(event):
    inp = event["input"]
    text = inp["text"]
    voice_id = inp.get("voice_id", "hi_male")
    voice_clone_id = inp.get("voice_clone_id")
    voice_clone_tokens = inp.get("voice_clone_tokens")

    payload = {
        "text": text,
        "stream": False,  # serverless returns one complete result, not a stream
    }
    if voice_clone_id:
        payload["voice_clone_id"] = voice_clone_id
    elif voice_clone_tokens:
        payload["voice_clone_tokens"] = voice_clone_tokens
    else:
        payload["voice_id"] = voice_id

    response = client.post("/v1/text-to-speech", json=payload)

    if response.status_code != 200:
        return {"error": response.text, "status_code": response.status_code}

    audio_b64 = base64.b64encode(response.content).decode("utf-8")
    return {
        "audio_base64": audio_b64,
        "sample_rate_hz": 24000,
        "format": "pcm_s16le",
    }

runpod.serverless.start({"handler": handler})
