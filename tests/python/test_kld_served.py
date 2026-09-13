"""tools/kld_served.py, device-free: the reference capture reader reconstructs
llama.cpp's uint16 log-prob rows to the quantisation step, the dump reader
stitches a window from prefill records by `past` and refuses a sliced record,
and the per-row KL reads 0 for an identical distribution and > 0 for a moved
one.

Red first: the reconstruction cell was written against `min_log_prob + q`
(no scale) and read a KL of 3.1 nats against the row's own log-softmax; the
stitching cell against a dump whose second record was sliced to one row
(rows != n) and the reader accepted it.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import kld_served as ks  # noqa: E402


def llama_row(logits):
    """llama.cpp perplexity.cpp `log_softmax(int, const float*, uint16_t*, int)`,
    transcribed: the writer of every recorded row."""
    logits = np.asarray(logits, dtype=np.float32)
    max_logit = float(logits.max())
    min_logit = max(float(logits.min()), max_logit - 16)
    log_sum_exp = float(np.log(np.exp(logits - max_logit).sum()))
    min_log_prob = min_logit - max_logit - log_sum_exp
    scale = (max_logit - min_logit) / 65535.0
    row = np.zeros(2 * ((len(logits) + 1) // 2) + 4, dtype=np.uint16)
    row[:4] = np.array([scale, min_log_prob], dtype=np.float32).view(np.uint16)
    if scale:
        q = np.where(logits > min_logit, np.rint((logits - min_logit) / scale), 0)
        row[4:4 + len(logits)] = q.astype(np.uint16)
    return row


def write_capture(path, n_ctx, n_vocab, windows_logits, tokens):
    n_chunk = len(windows_logits)
    first = n_ctx // 2
    with open(path, "wb") as f:
        f.write(b"_logits_")
        f.write(struct.pack("<iii", n_ctx, n_vocab, n_chunk))
        np.asarray(tokens, dtype=np.int32).reshape(n_chunk, n_ctx).tofile(f)
        for w in range(n_chunk):
            for p in range(first, n_ctx - 1):
                llama_row(windows_logits[w][p]).tofile(f)


def write_dump(path, records):
    """records: [(lane, past, n, rows_array[rows, vocab])]"""
    with open(path, "wb") as f:
        for lane, past, n, arr in records:
            arr = np.asarray(arr, dtype=np.float32)
            f.write(ks.DUMP_MAGIC)
            f.write(struct.pack("<5Q", lane, past, n, arr.shape[0], arr.shape[1]))
            arr.tofile(f)


def test_the_reference_row_reconstructs_to_the_quantisation_step():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=257).astype(np.float32) * 4
    row = llama_row(logits)
    ref_lp = ks.reference_log_probs(row, 257)
    direct = ks.log_softmax(logits.astype(np.float64))
    # the quantisation step is (max - min) / 65535 <= 16 / 65535 nats
    kept = logits > max(logits.min(), logits.max() - 16)
    assert np.abs(ref_lp[kept] - direct[kept]).max() < 16 / 65535 + 1e-6
    # the quantisation floor: measured 2.9e-6 nats for a 257-wide row whose
    # tail sits below max - 16 (the writer's clamp); 1e-4 is two decades
    # above it and three below anything the gate would read
    assert ks.kl_ref_vs_served(ref_lp, logits) < 1e-4


def test_kl_is_zero_for_the_same_row_and_positive_for_a_moved_one():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=64).astype(np.float32)
    row = llama_row(logits)
    ref_lp = ks.reference_log_probs(row, 64)
    assert ks.kl_ref_vs_served(ref_lp, logits) < 1e-4
    moved = logits.copy()
    moved[0] += 5.0
    assert ks.kl_ref_vs_served(ref_lp, moved) > 0.01


def test_windows_are_stitched_by_past_and_a_sliced_record_is_refused(tmp_path):
    n_ctx, vocab = 12, 9
    rng = np.random.default_rng(3)
    win = rng.normal(size=(n_ctx, vocab)).astype(np.float32)
    dump = tmp_path / "dump.bin"
    write_dump(dump, [(0, 0, 8, win[:8]), (0, 8, 4, win[8:]), (0, 12, 1, win[11:12]),
                      (0, 0, 12, win)])
    recs = ks.read_dump(dump)
    wins = ks.dump_windows(recs)
    assert [len(w) for w in wins] == [3, 1]
    np.testing.assert_array_equal(ks.window_rows(dump, wins[0], n_ctx, vocab), win)
    np.testing.assert_array_equal(ks.window_rows(dump, wins[1], n_ctx, vocab), win)
    sliced = tmp_path / "sliced.bin"
    write_dump(sliced, [(0, 0, 8, win[:8]), (0, 8, 4, win[11:12])])   # rows 1 for n 4
    with pytest.raises(ValueError, match="sliced"):
        ks.window_rows(sliced, ks.dump_windows(ks.read_dump(sliced))[0], n_ctx, vocab)


def test_compare_reads_zero_for_a_dump_equal_to_the_capture(tmp_path, capsys):
    n_ctx, vocab = 10, 7
    rng = np.random.default_rng(4)
    w0 = rng.normal(size=(n_ctx, vocab)).astype(np.float32)
    w1 = rng.normal(size=(n_ctx, vocab)).astype(np.float32)
    cap = tmp_path / "ref.dat"
    write_capture(cap, n_ctx, vocab, [w0, w1], np.arange(2 * n_ctx))
    n_ctx_r, n_vocab_r, n_chunk, tokens, rows = ks.read_capture(cap)
    assert (n_ctx_r, n_vocab_r, n_chunk) == (n_ctx, vocab, 2)
    assert tokens.tolist() == [list(range(10)), list(range(10, 20))]
    dump = tmp_path / "dump.bin"
    # two load-time probe windows first (128 and 4 tokens, not the capture's
    # n_ctx), then the two replays -- the compare must skip the probes
    write_dump(dump, [(0, 0, 4, w0[:4]), (0, 0, 3, w1[:3]),
                      (0, 0, n_ctx, w0), (0, n_ctx, 1, w0[-1:]),
                      (0, 0, n_ctx, w1), (0, n_ctx, 1, w1[-1:])])
    out = tmp_path / "rep.json"
    assert ks.main(["--ref", str(cap), "--compare", "--dump", str(dump),
                    "--out", str(out)]) == 0
    import json
    rep = json.loads(out.read_text())
    assert rep["mean_kl_below"] < 1e-6 and rep["mean_kl_above"] is None
    assert rep["windows"][0]["argmax_agreement"] == 1.0
    # and a moved dump reads positive
    write_dump(dump, [(0, 0, n_ctx, w0 + rng.normal(size=w0.shape) * 3),
                      (0, 0, n_ctx, w1)])
    assert ks.main(["--ref", str(cap), "--compare", "--dump", str(dump),
                    "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["mean_kl_below"] > 0.1


def test_the_floor_is_reported_from_the_last_two_replays(tmp_path):
    """REVIEW ba2d5de F1: two replays of the same windows give KL(A||B); zero
    for identical replays, positive for a moved second one. Written with the
    change, not before it: against the previous compare (no "floor" key in
    the report) the first assertion fails by construction, and that is the
    only red this cell has seen."""
    import json
    n_ctx, vocab = 10, 7
    rng = np.random.default_rng(5)
    w0 = rng.normal(size=(n_ctx, vocab)).astype(np.float32)
    w1 = rng.normal(size=(n_ctx, vocab)).astype(np.float32)
    cap = tmp_path / "ref.dat"
    write_capture(cap, n_ctx, vocab, [w0, w1], np.arange(2 * n_ctx))
    dump = tmp_path / "dump.bin"
    out = tmp_path / "rep.json"
    # replay A == replay B: floor 0
    write_dump(dump, [(0, 0, n_ctx, w0), (0, 0, n_ctx, w1), (0, 0, n_ctx, w0), (0, 0, n_ctx, w1)])
    assert ks.main(["--ref", str(cap), "--compare", "--dump", str(dump), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["floor"] is not None
    assert rep["floor"]["mean_kl_a_b"] < 1e-9 and rep["floor"]["max_kl_a_b"] < 1e-9
    assert all(f["argmax_agreement"] == 1.0 for f in rep["floor"]["windows"])
    # replay B moved: floor positive, and the means are against B (the last)
    w0b = w0 + rng.normal(size=w0.shape).astype(np.float32) * 2
    write_dump(dump, [(0, 0, n_ctx, w0), (0, 0, n_ctx, w1), (0, 0, n_ctx, w0b), (0, 0, n_ctx, w1)])
    assert ks.main(["--ref", str(cap), "--compare", "--dump", str(dump), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["floor"]["windows"][0]["mean_kl_a_b"] > 0.05
    assert rep["floor"]["windows"][1]["mean_kl_a_b"] < 1e-9
    assert rep["windows"][0]["mean_kl_below"] > 0.05      # ref vs B, B moved
    # one replay only: no floor, said so
    write_dump(dump, [(0, 0, n_ctx, w0), (0, 0, n_ctx, w1)])
    assert ks.main(["--ref", str(cap), "--compare", "--dump", str(dump), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["floor"] is None
