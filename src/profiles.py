#!/usr/bin/env python3
"""Speaker profile store + cosine matching (Phase 2).

A profile library is a flat JSON file: {name: [embedding, embedding, ...]}, one
or more 256-dim voiceprints per known person. Matching a new speaker's voiceprint
is a nearest-neighbour lookup by cosine similarity against every stored sample;
the best score decides the name if it clears the threshold.

Validated on real data (2026-07-23): same person across files ~0.84, different
people <=0.39 — so a threshold around 0.5 separates cleanly.
"""
import json
import math
import os
import tempfile
from pathlib import Path


def load_profiles(path):
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def save_profiles(path, profiles):
    """Write the profile store atomically (temp file in the same dir + rename),
    so a crash mid-write can't corrupt the biometric store."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".speakers-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(profiles, indent=2))
        os.replace(tmp, p)  # atomic on POSIX
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def rename_person(profiles, old, new):
    """Rename an enrolled person. If `new` already exists, their voiceprints are
    merged. Returns True if anything changed."""
    old, new = old.strip(), new.strip()
    if not new or old not in profiles or old == new:
        return False
    profiles.setdefault(new, [])
    profiles[new].extend(profiles.pop(old))
    return True


def delete_person(profiles, name):
    """Remove an enrolled person (and all their voiceprints). Returns True if
    they existed."""
    return profiles.pop(name, None) is not None


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def best_match(embedding, profiles, threshold):
    """Return (name, score) of the closest profile, or (None, best_score) if the
    closest is below `threshold`. `best_score` is always the raw top similarity so
    callers can print/calibrate it."""
    best_name, best_score = None, -1.0
    for name, vecs in profiles.items():
        for v in vecs:
            s = cosine(embedding, v)
            if s > best_score:
                best_score, best_name = s, name
    if best_score >= threshold:
        return best_name, best_score
    return None, best_score


def add_sample(profiles, name, embedding):
    """Append a voiceprint to a person's profile (creating it if new)."""
    profiles.setdefault(name, []).append(embedding)
