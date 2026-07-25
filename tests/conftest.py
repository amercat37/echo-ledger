"""Shared pytest fixtures.

All tests run against synthetic data in a tmp dir — they never touch the real
speakers.json, transcripts, or audio, and they never invoke WhisperX (the one
transcription test monkeypatches it). Voiceprints are tiny orthogonal vectors so
matching is deterministic: [1,0,0,0]=Allen, [0,1,0,0]=Sharon.
"""
import json
import sys
from pathlib import Path

import pytest

# Make src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def seg(spk, start, end, text):
    """A transcript segment (with word-level timing) for a speaker."""
    words = [{"speaker": spk, "start": start, "end": end, "word": w}
             for w in text.split()]
    return {"speaker": spk, "start": start, "end": end, "text": text, "words": words}


@pytest.fixture
def dirs(tmp_path):
    d = {k: tmp_path / k for k in ("input", "output", "done", "failed")}
    for p in d.values():
        p.mkdir()
    return d


@pytest.fixture
def config(tmp_path):
    return {
        "hf_token": "test-token", "model": "m", "device": "cpu",
        "compute_type": "int8", "language": "en", "threshold": 0.5,
        "speakers_file": str(tmp_path / "speakers.json"),
        "retention_days": 30, "log_level": "INFO",
    }


@pytest.fixture
def profiles():
    """Allen and Sharon, each with one orthogonal voiceprint."""
    return {"Allen": [[1.0, 0.0, 0.0, 0.0]], "Sharon": [[0.0, 1.0, 0.0, 0.0]]}


@pytest.fixture
def data():
    """A 2-speaker transcript. SPEAKER_00 == Allen's vector, SPEAKER_01 == Sharon's.
    SPEAKER_00's first solid (>=3s) segment is a clean opening; SPEAKER_01's longest
    is a 14s crosstalk-style ramble (to exercise sample_segment_for's preference)."""
    return {
        "segments": [
            seg("SPEAKER_00", 0.0, 1.0, "Hi"),
            seg("SPEAKER_00", 1.0, 5.0, "This is a clean opening over three seconds long"),
            seg("SPEAKER_01", 5.0, 6.0, "Yes"),
            seg("SPEAKER_01", 6.0, 20.0, "a long rambling crosstalk stretch that just keeps going"),
        ],
        "speaker_embeddings": {
            "SPEAKER_00": [1.0, 0.0, 0.0, 0.0],
            "SPEAKER_01": [0.0, 1.0, 0.0, 0.0],
        },
        "source_audio": "meeting.mp3",
    }


@pytest.fixture
def write_transcript(dirs):
    """Helper: persist a transcript JSON to output/<stem>.json."""
    def _write(stem, payload):
        (dirs["output"] / f"{stem}.json").write_text(json.dumps(payload))
        return dirs["output"] / f"{stem}.json"
    return _write
