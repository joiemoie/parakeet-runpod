import os
import math
import copy
import tempfile
import base64
from typing import Dict, Any, List, Tuple

import runpod
import requests
from pydub import AudioSegment

import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf, open_dict # Required for the config fix

# ====== Config ======
HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v3")

MODEL_PATH = os.environ.get(
    "PARAKEET_NEMO_PATH",
    HF_MODEL_NAME,
)

# 10 Minutes (Safe for TDT with Batch Size 2)
CHUNK_MS_DEFAULT = int(os.environ.get("CHUNK_MS", str(10 * 60 * 1000))) 

# Increased Overlap to 2.5s to improve boundary accuracy
OVERLAP_MS_DEFAULT = int(os.environ.get("OVERLAP_MS", "2500")) 

# TDT is optimized for Greedy decoding (1). 
BEAM_SIZE_DEFAULT = int(os.environ.get("BEAM_SIZE", "1")) 

MAX_SUFFIX_PREFIX_WORDS = int(os.environ.get("MAX_SUFFIX_PREFIX_WORDS", "50"))
TIME_GUARD_S = float(os.environ.get("TIME_GUARD_S", "0.08"))

# Switch to TDT for Max Accuracy
DEFAULT_DECODING_STRATEGY = os.environ.get("DECODING_STRATEGY", "tdt")

MODEL = None


def _format_hhmmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def unlock_and_force_tdt_config(model):
    """
    CRITICAL FIX: This unlocks the internal OmegaConf of the model 
    to force TDT timestamps to work. This prevents the 'Empty Text' bug.
    """
    try:
        # 1. Get the decoding config
        if hasattr(model, "cfg") and "decoding" in model.cfg:
            cfg = model.cfg.decoding
        elif hasattr(model, "_cfg") and "decoding" in model._cfg:
            cfg = model._cfg.decoding
        else:
            print("Warning: Could not find model config to unlock.")
            return

        # 2. Use open_dict context manager to allow writing to the config
        # This overrides the "read-only" lock causing issues in some containers
        with open_dict(cfg):
            cfg.strategy = "greedy"
            cfg.preserve_alignments = True
            cfg.compute_timestamps = True
            
            # Ensure beam size matches greedy
            if "beam" in cfg:
                cfg.beam.beam_size = 1

        # 3. Apply the modified config back to the model
        model.change_decoding_strategy(cfg)
        print("Success: TDT Config unlocked and timestamps enabled.")

    except Exception as e:
        print(f"Error forcing TDT config: {e}")


def get_model(strategy: str, beam_size: int, enable_timestamps: bool):
    global MODEL
    if MODEL is None:
        path = MODEL_PATH

        # Check runpod-volume/models cache if path is not found directly
        if not os.path.exists(path):
            name = path.split("/")[-1]
            if not name.endswith(".nemo"):
                name += ".nemo"
            cached_path = os.path.join("runpod-volume/models", name)
            if os.path.exists(cached_path):
                print(f"Found cached model at {cached_path}")
                path = cached_path

        print(f"Loading model from {path}...")
        
        if os.path.exists(path):
            MODEL = nemo_asr.models.ASRModel.restore_from(path)
        else:
            # Fallback to downloading/loading from HF Hub
            MODEL = nemo_asr.models.ASRModel.from_pretrained(model_name=path)

        MODEL.eval()

        # Apply the fix immediately after loading
        unlock_and_force_tdt_config(MODEL)

    return MODEL


def download_or_decode_audio(job_input: dict) -> str:
    tmpdir = tempfile.mkdtemp()
    audio_url = job_input.get("audio_url")
    audio_b64 = job_input.get("audio_b64")

    if audio_b64:
        raw = base64.b64decode(audio_b64)
        path = os.path.join(tmpdir, "input.mp3")
        with open(path, "wb") as f: f.write(raw)
        return path

    if audio_url:
        path = os.path.join(tmpdir, "input.mp3")
        r = requests.get(audio_url, timeout=180)
        r.raise_for_status()
        with open(path, "wb") as f: f.write(r.content)
        return path

    raise ValueError("Provide either 'audio_url' or 'audio_b64'.")


def to_16k_mono_wav(input_path: str) -> str:
    tmpdir = tempfile.mkdtemp()
    wav_path = os.path.join(tmpdir, "audio_16k.wav")
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(wav_path, format="wav")
    return wav_path


def split_wav_into_chunks(wav_path: str, chunk_ms: int, overlap_ms: int) -> List[Tuple[str, float, float]]:
    audio = AudioSegment.from_wav(wav_path)
    total_ms = len(audio)
    tmpdir = tempfile.mkdtemp()
    chunks = []

    step_ms = max(1, chunk_ms - overlap_ms)
    num_chunks = max(1, math.ceil(max(0, total_ms - overlap_ms) / step_ms))

    for i in range(num_chunks):
        start_ms = i * step_ms
        end_ms = min(total_ms, start_ms + chunk_ms)
        if start_ms >= total_ms: break
        if end_ms <= start_ms: continue

        seg = audio[start_ms:end_ms]
        chunk_path = os.path.join(tmpdir, f"chunk_{i:05d}.wav")
        seg.export(chunk_path, format="wav")
        chunks.append((chunk_path, start_ms / 1000.0, end_ms / 1000.0))
        if end_ms >= total_ms: break

    return chunks


def _extract_text_and_word_timestamps(hypothesis: Any) -> Tuple[str, List[Dict[str, Any]]]:
    if hypothesis is None: return "", []
    if isinstance(hypothesis, str): return hypothesis.strip(), []

    text = (getattr(hypothesis, "text", "") or "").strip()
    ts = getattr(hypothesis, "timestamp", None) or {}
    
    if isinstance(ts, list): word_items = [] 
    else: word_items = ts.get("word") or []

    words_out = []
    for item in word_items:
        w = (item.get("word") or "").strip()
        if not w: continue
        words_out.append({
            "word": w,
            "start": float(item.get("start", 0.0)),
            "end": float(item.get("end", 0.0))
        })
    return text, words_out


def transcribe_chunks_tdt_safe(model, chunk_infos, want_timestamps):
    """
    Optimized for TDT Accuracy with a safety fallback.
    """
    paths = [c[0] for c in chunk_infos]
    
    # Batch size 2 is safe for TDT memory usage
    BATCH_SIZE = 2

    all_hyps = []
    timestamps_worked = False
    
    # 1. Attempt High-Accuracy TDT with Timestamps
    if want_timestamps:
        try:
            # We already unlocked the config in get_model, so this SHOULD work
            all_hyps = model.transcribe(paths, batch_size=BATCH_SIZE, timestamps=True)
            
            # Validation: Did TDT fail silently? (Audio exists but 0 text returned)
            total_text_len = sum(len(h.text) if hasattr(h, 'text') else 0 for h in all_hyps)
            if total_text_len == 0 and len(paths) > 0:
                print("TDT Timestamps returned empty text. Falling back to text-only mode.")
                timestamps_worked = False
            else:
                timestamps_worked = True
                
        except Exception as e:
            print(f"TDT Timestamp inference failed ({e}). Falling back.")
            timestamps_worked = False

    # 2. Fallback: High-Accuracy TDT (Text Only)
    # If timestamps fail, we still want the high accuracy text, just without word times.
    if not timestamps_worked:
        all_hyps = model.transcribe(paths, batch_size=BATCH_SIZE)

    all_words = []
    chunk_results = []

    for i, hyp in enumerate(all_hyps):
        chunk_start_s = chunk_infos[i][1]
        text, words = _extract_text_and_word_timestamps(hyp)
        
        used_timestamps = (want_timestamps and timestamps_worked and len(words) > 0)

        if used_timestamps:
            shifted_words = []
            for w in words:
                shifted_words.append({
                    "word": w["word"],
                    "start": w["start"] + chunk_start_s,
                    "end": w["end"] + chunk_start_s
                })
            
            # Dedup Suffix/Prefix
            if all_words and shifted_words:
                prev_tokens = [x["word"] for x in all_words]
                cur_tokens = [x["word"] for x in shifted_words]
                k_max = min(MAX_SUFFIX_PREFIX_WORDS, len(prev_tokens), len(cur_tokens))
                cut_idx = 0
                for k in range(k_max, 0, -1):
                    if prev_tokens[-k:] == cur_tokens[:k]:
                        cut_idx = k
                        break
                shifted_words = shifted_words[cut_idx:]

            # Dedup Time Guard
            if all_words and shifted_words:
                last_end = all_words[-1]["end"]
                shifted_words = [w for w in shifted_words if w["start"] > (last_end - TIME_GUARD_S)]

            all_words.extend(shifted_words)

        chunk_results.append({
            "chunk_index": i,
            "start": _format_hhmmss(chunk_infos[i][1]),
            "end": _format_hhmmss(chunk_infos[i][2]),
            "text": text
        })

    if all_words:
        final_text = " ".join(w["word"] for w in all_words)
    else:
        final_text = "\n".join(c["text"] for c in chunk_results).strip()

    return {
        "text": final_text,
        "words": all_words,
        "chunks": chunk_results,
        "mode": "TDT"
    }


def handler(job):
    inp = job.get("input", {}) or {}
    
    chunk_minutes = int(inp.get("chunk_minutes", 10))
    chunk_ms = chunk_minutes * 60 * 1000
    overlap_ms = int(inp.get("overlap_ms", OVERLAP_MS_DEFAULT))
    beam_size = int(inp.get("beam_size", BEAM_SIZE_DEFAULT))
    want_timestamps = bool(inp.get("timestamps", True))
    
    # Default to TDT for better accuracy
    strategy = inp.get("decoding_strategy", DEFAULT_DECODING_STRATEGY) 

    model = get_model(strategy, beam_size, want_timestamps)

    raw_path = download_or_decode_audio(inp)
    wav_path = to_16k_mono_wav(raw_path)
    chunk_infos = split_wav_into_chunks(wav_path, chunk_ms, overlap_ms)

    out = transcribe_chunks_tdt_safe(model, chunk_infos, want_timestamps)

    return {
        "decoding_strategy": "TDT (High Accuracy)",
        "text": out["text"],
        "words": out["words"],
        "chunks": out["chunks"]
    }

runpod.serverless.start({"handler": handler})
