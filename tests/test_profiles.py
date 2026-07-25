"""profiles.py — voiceprint store, cosine matching, and profile mutations."""
import json

import profiles as prof


def test_cosine_identical_and_orthogonal():
    assert prof.cosine([1, 0, 0], [1, 0, 0]) == 1.0
    assert prof.cosine([1, 0, 0], [0, 1, 0]) == 0.0
    assert prof.cosine([0, 0, 0], [1, 1, 1]) == 0.0  # zero vector is safe


def test_best_match_above_and_below_threshold(profiles):
    name, score = prof.best_match([1.0, 0.0, 0.0, 0.0], profiles, 0.5)
    assert name == "Allen" and score == 1.0
    # orthogonal to everyone -> no match, but the raw best score is returned
    name, score = prof.best_match([0.0, 0.0, 1.0, 0.0], profiles, 0.5)
    assert name is None and score < 0.5


def test_add_sample_appends(profiles):
    prof.add_sample(profiles, "Allen", [0.9, 0.1, 0.0, 0.0])
    assert len(profiles["Allen"]) == 2
    prof.add_sample(profiles, "New", [0.0, 0.0, 1.0, 0.0])
    assert profiles["New"] == [[0.0, 0.0, 1.0, 0.0]]


def test_rename_person_merges_when_target_exists(profiles):
    assert prof.rename_person(profiles, "Sharon", "Allen") is True
    assert "Sharon" not in profiles
    assert len(profiles["Allen"]) == 2  # merged Sharon's print into Allen


def test_rename_person_noops_on_missing_or_blank(profiles):
    assert prof.rename_person(profiles, "Nobody", "X") is False
    assert prof.rename_person(profiles, "Allen", "  ") is False
    assert "Allen" in profiles


def test_delete_person(profiles):
    assert prof.delete_person(profiles, "Allen") is True
    assert "Allen" not in profiles
    assert prof.delete_person(profiles, "Allen") is False


def test_save_is_atomic_and_round_trips(tmp_path, profiles):
    path = tmp_path / "sp.json"
    prof.save_profiles(str(path), profiles)
    assert json.loads(path.read_text()) == profiles
    # no leftover temp files from the atomic write
    assert not list(tmp_path.glob(".speakers-*.tmp"))


def test_load_missing_returns_empty(tmp_path):
    assert prof.load_profiles(str(tmp_path / "nope.json")) == {}
