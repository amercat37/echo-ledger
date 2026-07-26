"""refine_speakers.py — diarization cleanup + markdown rendering."""
import refine_speakers as r


def test_ts_formats():
    assert r.ts(5) == "0:05"
    assert r.ts(65) == "1:05"
    assert r.ts(3661) == "1:01:01"


def test_atomic_write_text(tmp_path):
    p = tmp_path / "sub" / "out.md"   # parent doesn't exist yet
    r.atomic_write_text(p, "hello\n")
    assert p.read_text() == "hello\n"
    assert not list((tmp_path / "sub").glob(".tmp-*"))  # no residue


def test_fill_word_gaps_borrows_from_nearest():
    segs = [{"words": [
        {"start": 0.0, "end": 1.0, "speaker": "A"},
        {"start": 1.0, "end": 2.0},                       # missing -> nearest is A
        {"start": 9.0, "end": 10.0, "speaker": "B"},
        {"start": 8.5, "end": 9.0},                       # missing -> nearest is B
    ]}]
    filled = r.fill_word_gaps(segs)
    speakers = [w["speaker"] for w in segs[0]["words"]]
    assert filled == 2
    assert speakers == ["A", "A", "B", "B"]


def test_revote_segments_by_duration_majority():
    seg = {"speaker": "A", "words": [
        {"start": 0.0, "end": 0.2, "speaker": "A"},   # 0.2s of A
        {"start": 0.2, "end": 3.2, "speaker": "B"},   # 3.0s of B  -> B wins
    ]}
    r.revote_segments([seg])
    assert seg["speaker"] == "B"


def test_refine_data_fills_all_gaps():
    data = {"segments": [{"speaker": None, "words": [
        {"start": 0.0, "end": 1.0, "speaker": "A"},
        {"start": 1.0, "end": 2.0},
    ]}]}
    stats = r.refine_data(data)
    assert stats["words_missing_after"] == 0
    assert stats["segments_missing_after"] == 0


def test_friendly_and_display_labels():
    segs = [{"speaker": "SPEAKER_00"}, {"speaker": "SPEAKER_01"}, {"speaker": "SPEAKER_00"}]
    assert r.friendly_labels(segs) == {"SPEAKER_00": "Speaker 1", "SPEAKER_01": "Speaker 2"}
    # a matched speaker shows their name; the rest are numbered among the un-named
    disp = r.display_labels(segs, {"SPEAKER_01": "Allen"})
    assert disp == {"SPEAKER_00": "Speaker 1", "SPEAKER_01": "Allen"}


def test_write_markdown_merges_consecutive_turns(tmp_path):
    segs = [
        {"speaker": "SPEAKER_00", "start": 0, "text": "Hello."},
        {"speaker": "SPEAKER_00", "start": 2, "text": "How are you?"},
        {"speaker": "SPEAKER_01", "start": 5, "text": "Good."},
    ]
    out = tmp_path / "t.md"
    r.write_markdown(segs, out, {"SPEAKER_00": "Allen", "SPEAKER_01": "Sharon"})
    text = out.read_text()
    assert "[0:00] Allen: Hello. How are you?" in text   # merged into one turn
    assert text.count("Allen:") == 1                      # not repeated per segment
    assert "[0:05] Sharon: Good." in text
