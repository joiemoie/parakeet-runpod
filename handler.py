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

# ====== Config ======
MODEL_PATH = os.environ.get(
    "PARAKEET_NEMO_PATH",
    "/runpod-volume/models/parakeet-tdt_ctc-1.1b.nemo",
)

# 10 Minutes (CTC handles long context better than TDT)
CHUNK_MS_DEFAULT = int(os.environ.get("CHUNK_MS", str(10 * 60 * 1000))) 
OVERLAP_MS_DEFAULT = int(os.environ.get("OVERLAP_MS", "2000")) 
BEAM_SIZE_DEFAULT = int(os.environ.get("BEAM_SIZE", "1")) 
MAX_SUFFIX_PREFIX_WORDS = int(os.environ.get("MAX_SUFFIX_PREFIX_WORDS", "50"))
TIME_GUARD_S = float(os.environ.get("TIME_GUARD_S", "0.08"))

# FORCE CTC (Fixes the "Empty Text" bug with timestamps)
DEFAULT_DECODING_STRATEGY = os.environ.get("DECODING_STRATEGY", "ctc")

MODEL = None


def _format_hhmmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _set_nested(obj: Any, path: List[str], value: Any) -> bool:
    cur = obj
    for key in path[:-1]:
        if cur is None or not hasattr(cur, key): return False
        cur = getattr(cur, key)
    last = path[-1]
    if cur is None or not hasattr(cur, last): return False
    try:
        setattr(cur, last, value)
        return True
    except Exception:
        return False


def configure_decoding(model, strategy: str, beam_size: int, enable_timestamps: bool) -> Dict[str, Any]:
    info = {
        "strategy_requested": strategy,
        "config_source": "default",
        "params_set": [],
        "error": None
    }
    decoding_cfg = None

    # --- 1. Find the correct config ---
    # For Parakeet Hybrid, the CTC config is in model.cfg.aux_ctc
    if strategy.lower() == "ctc":
        if hasattr(model, "cfg") and hasattr(model.cfg, "aux_ctc") and hasattr(model.cfg.aux_ctc, "decoding"):
            decoding_cfg = copy.deepcopy(model.cfg.aux_ctc.decoding)
            info["config_source"] = "aux_ctc"
        else:
            # Fallback
            if hasattr(model, "cfg") and hasattr(model.cfg, "decoding"):
                decoding_cfg = copy.deepcopy(model.cfg.decoding)
            if _set_nested(decoding_cfg, ["strategy"], "ctc"):
                info["params_set"].append("strategy=ctc")
    else:
        # TDT
        if hasattr(model, "cfg") and hasattr(model.cfg, "decoding"):
            decoding_cfg = copy.deepcopy(model.cfg.decoding)
        elif hasattr(model, "_cfg") and hasattr(model._cfg, "decoding"):
            decoding_cfg = copy.deepcopy(model._cfg.decoding)

    if decoding_cfg is None:
        info["error"] = "Could not find decoding config"
        return info

    # --- 2. Set Beam Size ---
    beam_paths = [
        ["beam", "beam_size"], 
        ["ctc", "beam", "beam_size"], 
        ["ctc", "beam_size"]
    ]
    for path in beam_paths:
        if _set_nested(decoding_cfg, path, int(beam_size)):
            info["params_set"].append(f"{'.'.join(path)}={beam_size}")
            break

    # --- 3. Set Timestamps ---
    if enable_timestamps:
        ts_paths = [
            (["preserve_alignments"], True),
            (["compute_timestamps"], True),
            (["ctc", "preserve_alignments"], True), 
            (["ctc", "compute_timestamps"], True),
        ]
        for path, val in ts_paths:
            if _set_nested(decoding_cfg, path, val):
                info["params_set"].append(".".join(path))

    # --- 4. Apply ---
    try:
        model.change_decoding_strategy(decoding_cfg)
        info["success"] = True
    except Exception as e:
        info["success"] = False
        info["error"] = str(e)
        print(f"ERROR applying decoding strategy: {e}")

    return info


def get_model(strategy: str, beam_size: int, enable_timestamps: bool):
    global MODEL
    if MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        print(f"Loading model from {MODEL_PATH}...")
        MODEL = nemo_asr.models.ASRModel.restore_from(MODEL_PATH)
        MODEL.eval()

    decoding_info = configure_decoding(MODEL, strategy, beam_size, enable_timestamps)
    return MODEL, decoding_info


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
    """Converts to 16kHz mono WAV (No Normalization)."""
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


def transcribe_chunks_batched(model, chunk_infos, want_timestamps):
    paths = [c[0] for c in chunk_infos]
    # Batch size 2 is safe for 10-minute chunks. 
    # If using 5-min chunks, you can bump to 4.
    BATCH_SIZE = 2 

    all_hyps = []
    timestamps_worked = True
    
    try:
        # Batched Inference
        if want_timestamps:
            try:
                all_hyps = model.transcribe(paths, batch_size=BATCH_SIZE, timestamps=True)
            except Exception as e:
                print(f"Batch timestamps failed: {e}. Fallback to text-only.")
                timestamps_worked = False
                all_hyps = model.transcribe(paths, batch_size=BATCH_SIZE)
        else:
            all_hyps = model.transcribe(paths, batch_size=BATCH_SIZE)
            
    except Exception as e:
        print(f"Batch inference failed: {e}. Fallback to sequential.")
        all_hyps = []
        for p in paths:
             all_hyps.extend(model.transcribe([p]))

    all_words = []
    chunk_results = []

    for i, hyp in enumerate(all_hyps):
        chunk_start_s = chunk_infos[i][1]
        text, words = _extract_text_and_word_timestamps(hyp)
        
        # Cleanup extra spaces common in CTC
        text = " ".join(text.split())

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
        final_text = "\n".join(c["text"] for c in chunk_results)

    return {
        "text": final_text,
        "words": all_words,
        "chunks": chunk_results,
        "timestamps_worked": timestamps_worked
    }


def handler(job):
    inp = job.get("input", {}) or {}
    
    chunk_minutes = int(inp.get("chunk_minutes", 10))
    chunk_ms = chunk_minutes * 60 * 1000
    overlap_ms = int(inp.get("overlap_ms", OVERLAP_MS_DEFAULT))
    beam_size = int(inp.get("beam_size", BEAM_SIZE_DEFAULT))
    want_timestamps = bool(inp.get("timestamps", True))
    strategy = inp.get("decoding_strategy", DEFAULT_DECODING_STRATEGY) 

    model, decoding_info = get_model(strategy, beam_size, want_timestamps)

    raw_path = download_or_decode_audio(inp)
    wav_path = to_16k_mono_wav(raw_path)
    chunk_infos = split_wav_into_chunks(wav_path, chunk_ms, overlap_ms)

    out = transcribe_chunks_batched(model, chunk_infos, want_timestamps)

    return {
        "decoding": decoding_info,
        "text": out["text"],
        "words": out["words"],
        "chunks": out["chunks"]
    }

runpod.serverless.start({"handler": handler})
