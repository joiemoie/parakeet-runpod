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

# ====== Config (CDCR-friendly defaults) ======
MODEL_PATH = os.environ.get(
    "PARAKEET_NEMO_PATH",
    "/runpod-volume/models/parakeet-tdt_ctc-1.1b.nemo",
)

# 10-minute chunks (in ms)
CHUNK_MS_DEFAULT = int(os.environ.get("CHUNK_MS", str(10 * 60 * 1000)))  # 600_000

# Overlap reduces boundary word cuts, but causes duplicates -> we dedup later
OVERLAP_MS_DEFAULT = int(os.environ.get("OVERLAP_MS", "750"))  # 0.75s

# Beam size for better recall
BEAM_SIZE_DEFAULT = int(os.environ.get("BEAM_SIZE", "8"))

# Dedup behavior
MAX_SUFFIX_PREFIX_WORDS = int(os.environ.get("MAX_SUFFIX_PREFIX_WORDS", "40"))
# Time guard (seconds). Any new word whose start < last_end - guard is dropped.
TIME_GUARD_S = float(os.environ.get("TIME_GUARD_S", "0.08"))

MODEL = None


def _format_hhmmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _set_nested(obj: Any, path: List[str], value: Any) -> bool:
    """
    Best-effort nested setattr for OmegaConf-like objects.
    Returns True if applied, False otherwise.
    """
    cur = obj
    for key in path[:-1]:
        if cur is None or not hasattr(cur, key):
            return False
        cur = getattr(cur, key)
    last = path[-1]
    if cur is None or not hasattr(cur, last):
        return False
    try:
        setattr(cur, last, value)
        return True
    except Exception:
        return False


def configure_ctc_decoding(model, beam_size: int, enable_timestamps: bool) -> Dict[str, Any]:
    """
    Force CTC decoding on Parakeet-TDT-CTC checkpoints (some NeMo versions treat these as hybrid RNNT+CTC).
    Also enables timestamp computation flags when available.

    Returns a dict describing what settings were successfully applied.
    """
    info: Dict[str, Any] = {
        "strategy_set": None,
        "beam_set": None,
        "timestamps_flags_set": [],
    }

    # Some NeMo models have model.cfg.decoding; some have model._cfg.decoding.
    decoding_cfg = None
    if hasattr(model, "cfg") and hasattr(model.cfg, "decoding"):
        decoding_cfg = copy.deepcopy(model.cfg.decoding)
    elif hasattr(model, "_cfg") and hasattr(model._cfg, "decoding"):
        decoding_cfg = copy.deepcopy(model._cfg.decoding)

    if decoding_cfg is None:
        return info

    # ---- Force decoding strategy to CTC ----
    # Different NeMo versions use different keys/strategies:
    # - decoding_cfg.strategy = "ctc"
    # - decoding_cfg.decoding_type = "ctc"
    # We'll try both.
    if _set_nested(decoding_cfg, ["strategy"], "ctc"):
        info["strategy_set"] = "strategy=ctc"
    elif _set_nested(decoding_cfg, ["decoding_type"], "ctc"):
        info["strategy_set"] = "decoding_type=ctc"

    # ---- Beam for CTC ----
    # Different shapes exist across versions:
    # - decoding_cfg.ctc.beam_size
    # - decoding_cfg.ctc.beam.beam_size
    # We'll try both.
    if _set_nested(decoding_cfg, ["ctc", "beam_size"], int(beam_size)):
        info["beam_set"] = "ctc.beam_size"
    elif _set_nested(decoding_cfg, ["ctc", "beam", "beam_size"], int(beam_size)):
        info["beam_set"] = "ctc.beam.beam_size"

    # ---- Timestamp flags (best-effort) ----
    # These flags exist in some NeMo builds; safe to no-op if absent.
    if enable_timestamps:
        for path, val in [
            (["preserve_alignments"], True),
            (["compute_timestamps"], True),
            (["ctc", "preserve_alignments"], True),
            (["ctc", "compute_timestamps"], True),
        ]:
            if _set_nested(decoding_cfg, path, val):
                info["timestamps_flags_set"].append(".".join(path))

    # Apply
    try:
        model.change_decoding_strategy(decoding_cfg)
    except Exception:
        # If change_decoding_strategy fails, we just return what we *attempted*.
        info["change_decoding_strategy_failed"] = True

    return info


def get_model(beam_size: int = BEAM_SIZE_DEFAULT, enable_timestamps: bool = True):
    """
    Loads Parakeet model once and forces CTC decoding + beam.
    (Even though the checkpoint name is *_ctc, some NeMo versions default to the RNNT/TDT decoder unless forced.)
    """
    global MODEL
    if MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Make sure your network volume is mounted and the .nemo file exists."
            )

        # Use the generic ASRModel loader to avoid mismatched class assumptions.
        MODEL = nemo_asr.models.ASRModel.restore_from(MODEL_PATH)
        MODEL.eval()

    # Force CTC decode settings each request (so beam/timestamp toggles can change)
    decoding_info = configure_ctc_decoding(MODEL, beam_size=beam_size, enable_timestamps=enable_timestamps)
    return MODEL, decoding_info


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
    """Converts to 16kHz mono 16-bit PCM WAV."""
    tmpdir = tempfile.mkdtemp()
    wav_path = os.path.join(tmpdir, "audio_16k.wav")

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(wav_path, format="wav")

    return wav_path


def split_wav_into_chunks(
    wav_path: str,
    chunk_ms: int,
    overlap_ms: int,
) -> List[Tuple[str, float, float]]:
    """
    Returns: (chunk_path, start_s, end_s) in ORIGINAL timeline.
    end is exclusive.
    """
    audio = AudioSegment.from_wav(wav_path)
    total_ms = len(audio)
    if total_ms <= 0:
        raise ValueError("Audio appears to be empty after conversion.")

    tmpdir = tempfile.mkdtemp()
    chunks: List[Tuple[str, float, float]] = []

    step_ms = max(1, chunk_ms - overlap_ms)
    num_chunks = max(1, math.ceil(max(0, total_ms - overlap_ms) / step_ms))

    for i in range(num_chunks):
        start_ms = i * step_ms
        end_ms = min(total_ms, start_ms + chunk_ms)
        if start_ms >= total_ms:
            break
        if end_ms <= start_ms:
            continue

        seg = audio[start_ms:end_ms]
        chunk_path = os.path.join(tmpdir, f"chunk_{i:05d}.wav")
        seg.export(chunk_path, format="wav")
        chunks.append((chunk_path, start_ms / 1000.0, end_ms / 1000.0))

        if end_ms >= total_ms:
            break

    return chunks


def _extract_text_and_word_timestamps(hypothesis: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Best-effort extraction of:
      - text: hypothesis.text
      - word timestamps: hypothesis.timestamp['word'] if present

    Many NeMo builds won't return word times for CTC directly; we keep it robust.
    """
    if hypothesis is None:
        return "", []

    if isinstance(hypothesis, str):
        return hypothesis.strip(), []

    text = (getattr(hypothesis, "text", "") or "").strip()

    ts = getattr(hypothesis, "timestamp", None) or {}
    word_items = ts.get("word") or []

    words_out: List[Dict[str, Any]] = []
    for item in word_items:
        w = (item.get("word") or "").strip()
        if not w:
            continue
        start_s = item.get("start")
        end_s = item.get("end")
        words_out.append(
            {
                "word": w,
                "start": (None if start_s is None else float(start_s)),
                "end": (None if end_s is None else float(end_s)),
            }
        )

    return text, words_out


def _shift_words(words: List[Dict[str, Any]], shift_s: float) -> List[Dict[str, Any]]:
    out = []
    for w in words:
        ws = w.get("start")
        we = w.get("end")
        out.append(
            {
                "word": w["word"],
                "start": (None if ws is None else float(ws) + shift_s),
                "end": (None if we is None else float(we) + shift_s),
            }
        )
    return out


def _suffix_prefix_dedup(prev_words: List[Dict[str, Any]], cur_words: List[Dict[str, Any]], max_k: int) -> List[Dict[str, Any]]:
    if not prev_words or not cur_words:
        return cur_words

    prev_tokens = [w["word"] for w in prev_words]
    cur_tokens = [w["word"] for w in cur_words]

    k_max = min(max_k, len(prev_tokens), len(cur_tokens))
    for k in range(k_max, 0, -1):
        if prev_tokens[-k:] == cur_tokens[:k]:
            return cur_words[k:]
    return cur_words


def _time_guard_dedup(all_words: List[Dict[str, Any]], new_words: List[Dict[str, Any]], guard_s: float) -> List[Dict[str, Any]]:
    if not all_words or not new_words:
        return new_words

    last_end = None
    for w in reversed(all_words):
        if w.get("end") is not None:
            last_end = float(w["end"])
            break
    if last_end is None:
        return new_words

    kept = []
    for w in new_words:
        s = w.get("start")
        e = w.get("end")
        if s is None or e is None:
            kept.append(w)
            continue
        if float(s) < (last_end - guard_s):
            continue
        kept.append(w)
    return kept


def transcribe_chunks_with_optional_timestamps(
    model,
    chunk_infos: List[Tuple[str, float, float]],
    want_timestamps: bool,
) -> Dict[str, Any]:
    """
    Transcribe each chunk; try timestamps if requested.
    If NeMo crashes on timestamps (common with hybrid checkpoints), automatically fall back to no timestamps.
    """
    all_words: List[Dict[str, Any]] = []
    chunk_results: List[Dict[str, Any]] = []
    timestamps_worked = True

    for i, (chunk_path, chunk_start_s, chunk_end_s) in enumerate(chunk_infos):
        hyps = None
        used_timestamps = False

        if want_timestamps:
            try:
                hyps = model.transcribe([chunk_path], timestamps=True)
                used_timestamps = True
            except TypeError:
                # Classic failure: hybrid RNNT timestamp path sees None durations/timestamps
                timestamps_worked = False
                hyps = model.transcribe([chunk_path])
            except Exception:
                # Any other timestamp-related failure -> fall back
                timestamps_worked = False
                hyps = model.transcribe([chunk_path])

        else:
            hyps = model.transcribe([chunk_path])

        hyp0 = hyps[0] if hyps else None
        text, words = _extract_text_and_word_timestamps(hyp0)

        if used_timestamps and words:
            words = _shift_words(words, chunk_start_s)
            if all_words and words:
                words = _suffix_prefix_dedup(all_words, words, MAX_SUFFIX_PREFIX_WORDS)
            if all_words and words:
                words = _time_guard_dedup(all_words, words, TIME_GUARD_S)
            all_words.extend(words)

        chunk_results.append(
            {
                "chunk_index": i,
                "start": _format_hhmmss(chunk_start_s),
                "end": _format_hhmmss(chunk_end_s),
                "text": text,
                "words": words if (used_timestamps and words) else [],
            }
        )

    if all_words:
        final_text = " ".join(w["word"] for w in all_words).strip()
    else:
        final_text = "\n\n".join(
            f"[{c['start']} - {c['end']}] {c['text']}".strip() for c in chunk_results
        ).strip()

    return {
        "text": final_text,
        "words": all_words,
        "chunks": chunk_results,
        "timestamps_worked": timestamps_worked and bool(all_words),
    }


def handler(job):
    inp = job.get("input", {}) or {}

    # Per-request overrides
    chunk_minutes = int(inp.get("chunk_minutes", 10))
    overlap_ms = int(inp.get("overlap_ms", OVERLAP_MS_DEFAULT))
    beam_size = int(inp.get("beam_size", BEAM_SIZE_DEFAULT))
    want_timestamps = bool(inp.get("timestamps", True))

    chunk_ms = chunk_minutes * 60 * 1000

    # Load model & force CTC decode
    model, decoding_info = get_model(beam_size=beam_size, enable_timestamps=want_timestamps)

    # Audio -> 16k mono WAV
    raw_path = download_or_decode_audio(inp)
    wav_path = to_16k_mono_wav(raw_path)

    # Chunk
    chunk_infos = split_wav_into_chunks(wav_path, chunk_ms=chunk_ms, overlap_ms=overlap_ms)

    # Transcribe (timestamps best-effort; falls back if NeMo breaks)
    out = transcribe_chunks_with_optional_timestamps(model, chunk_infos, want_timestamps=want_timestamps)

    return {
        "model_path": MODEL_PATH,
        "chunk_minutes": chunk_minutes,
        "overlap_ms": overlap_ms,
        "beam_size": beam_size,
        "timestamps_requested": want_timestamps,
        "timestamps_worked": out["timestamps_worked"],
        "decoding_applied": decoding_info,
        "dedup": {
            "max_suffix_prefix_words": MAX_SUFFIX_PREFIX_WORDS,
            "time_guard_s": TIME_GUARD_S,
        },
        "text": out["text"],
        "words": out["words"],   # global word timeline if available
        "chunks": out["chunks"],
    }


runpod.serverless.start({"handler": handler})

