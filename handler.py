import os
import tempfile
import base64

import runpod
import requests
from pydub import AudioSegment

import nemo.collections.asr as nemo_asr

MODEL_PATH = os.environ.get("PARAKEET_NEMO_PATH", "/runpod-volume/models/parakeet-tdt-1.1b.nemo")
MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Make sure your network volume is mounted at /workspace and the .nemo file exists."
            )
        MODEL = nemo_asr.models.EncDecRNNTBPEModel.restore_from(MODEL_PATH)
        MODEL.eval()
    return MODEL

def download_or_decode_audio(job_input: dict) -> str:
    """Returns a local filepath to the input audio (mp3/wav/etc)."""
    tmpdir = tempfile.mkdtemp()
    audio_url = job_input.get("audio_url")
    audio_b64 = job_input.get("audio_b64")

    if audio_b64:
        raw = base64.b64decode(audio_b64)
        path = os.path.join(tmpdir, "input.mp3")
        with open(path, "wb") as f:
            f.write(raw)
        return path

    if audio_url:
        path = os.path.join(tmpdir, "input.mp3")
        r = requests.get(audio_url, timeout=180)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return path

    raise ValueError("Provide either 'audio_url' or 'audio_b64'.")

def to_16k_mono_wav(input_path: str) -> str:
    tmpdir = tempfile.mkdtemp()
    wav_path = os.path.join(tmpdir, "audio_16k.wav")

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(wav_path, format="wav")

    return wav_path

def handler(job):
    inp = job.get("input", {}) or {}

    # Load model from network volume (first request only)
    model = get_model()

    # Get audio
    raw_path = download_or_decode_audio(inp)
    wav_path = to_16k_mono_wav(raw_path)

    # Transcribe
    # NeMo expects list of paths
    out = model.transcribe([wav_path])

    text = out[0].text if out and out[0] else ""

    return {
        "text": text,
        "model_path": MODEL_PATH,
        "model": "parakeet-tdt-1.1b",
    }

runpod.serverless.start({"handler": handler})
