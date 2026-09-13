#!/usr/bin/env python3
"""Unit tests for tools/kld_capture.py. Python 3 stdlib only, no venv, no
model, no card.

The cell that matters is the boundary refusal: a capture that cannot serve the
gate must be refused BY NAME, and the arithmetic of "cannot" has an off-by-one
in it that is worth pinning from both sides.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kld_capture as kc  # noqa: E402


class Args:
    """A stand-in for argparse's namespace."""

    def __init__(self, **kw):
        self.reference_bin = "/nonexistent/llama-perplexity"
        self.reference_checkout = None
        self.model = "/nonexistent/model.gguf"
        self.corpus = "/nonexistent/corpus.txt"
        self.out = "/nonexistent/out.dat"
        self.n_ctx = 4096
        self.chunks = 2
        self.batch = 512
        self.threads = 8
        self.hash_model = False
        self.__dict__.update(kw)


class BoundaryRefusal(unittest.TestCase):
    # THE TWO MEASURED POINTS, and the formula is pinned to them rather than to
    # anybody's reasoning about index arithmetic. The first draft of the tool
    # used `T - 1 - boundary` and a test asserted it; both were wrong by one
    # against these rows, and both looked right.
    MEASURED_ROWS = {2052: 1, 2080: 29}

    def test_the_formula_reproduces_both_measured_rows(self):
        for T, affected in self.MEASURED_ROWS.items():
            self.assertEqual(kc.rows_past_boundary(T), affected,
                             f"T={T}: the recorded measurement is "
                             f"{affected}/{T} rows")

    def test_the_smallest_window_the_price_touches_is_the_boundary_plus_one(self):
        self.assertEqual(kc.check_context_crosses_boundary(
            kc.QSA_BOUNDARY_TOKENS + 1), 1)
        with self.assertRaises(kc.CaptureRefusal):
            kc.check_context_crosses_boundary(kc.QSA_BOUNDARY_TOKENS)

    def test_a_window_below_the_boundary_is_refused(self):
        for n_ctx in (1, 512, 2048, kc.QSA_BOUNDARY_TOKENS):
            with self.assertRaises(kc.CaptureRefusal):
                kc.check_context_crosses_boundary(n_ctx)

    def test_the_refusal_names_the_boundary_and_a_usable_value(self):
        try:
            kc.check_context_crosses_boundary(2048)
        except kc.CaptureRefusal as exc:
            msg = str(exc)
        self.assertIn(str(kc.QSA_BOUNDARY_TOKENS), msg)
        self.assertIn(str(kc.QSA_BOUNDARY_TOKENS + 1), msg)
        self.assertIn("2.385560e-02", msg)

    def test_the_touched_count_at_the_capture_width(self):
        """4096 - 2051 = 2045 rows carry the price and 2051 sit below it.
        Derived from the same formula the measured rows pin."""
        self.assertEqual(kc.check_context_crosses_boundary(4096),
                         4096 - kc.QSA_BOUNDARY_TOKENS)
        self.assertEqual(kc.rows_past_boundary(4096), 2045)

    def test_the_boundary_is_not_the_budget(self):
        """2048 is the QSA budget; 2051 is the measured boundary. A tool that
        confused them would take captures that look right and are not."""
        self.assertEqual(kc.QSA_BOUNDARY_TOKENS, 2051)
        with self.assertRaises(kc.CaptureRefusal):
            kc.check_context_crosses_boundary(2048)


class Command(unittest.TestCase):
    def test_the_capture_flag_is_the_upstream_one(self):
        cmd = kc.build_command(Args())
        self.assertIn("--kl-divergence-base", cmd)
        self.assertEqual(cmd[cmd.index("--kl-divergence-base") + 1],
                         "/nonexistent/out.dat")

    def test_the_comparison_flag_is_never_issued(self):
        """The brief's line: build the capture, not the comparison. A bare
        `--kl-divergence` here would silently turn a capture run into a
        comparison against a file that does not exist."""
        self.assertNotIn("--kl-divergence", kc.build_command(Args()))

    def test_every_knob_reaches_the_command(self):
        cmd = kc.build_command(Args(n_ctx=2053, chunks=3, batch=256, threads=4))
        for flag, want in (("-c", "2053"), ("--chunks", "3"),
                           ("-b", "256"), ("-t", "4")):
            self.assertEqual(cmd[cmd.index(flag) + 1], want, flag)


class Manifest(unittest.TestCase):
    def test_the_manifest_hashes_the_capture_not_only_its_inputs(self):
        """A manifest that describes inputs and not the output cannot tell a
        complete capture from a truncated one."""
        with tempfile.TemporaryDirectory() as d:
            for name, body in (("bin", b"\x7fELF"), ("corpus", b"hello"),
                               ("out", b"logits")):
                with open(os.path.join(d, name), "wb") as fh:
                    fh.write(body)
            args = Args(reference_bin=os.path.join(d, "bin"),
                        corpus=os.path.join(d, "corpus"),
                        out=os.path.join(d, "out"))
            man = kc.manifest(args, 2045, 0.0, 1.5, 0, "tail")
            self.assertEqual(man["capture"]["bytes"], len(b"logits"))
            self.assertEqual(len(man["capture"]["sha256"]), 64)
            self.assertEqual(len(man["corpus"]["sha256"]), 64)
            self.assertEqual(len(man["reference"]["binary_sha256"]), 64)
            self.assertIsNone(man["model"]["sha256"])   # --hash-model off
            self.assertEqual(man["qsa_boundary_tokens"], 2051)
            self.assertEqual(man["rows_past_boundary_per_window"], 2045)
            json.dumps(man)                              # must be serialisable

    def test_a_missing_capture_file_is_recorded_as_absent_not_as_zero_hash(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("bin", "corpus"):
                with open(os.path.join(d, name), "wb") as fh:
                    fh.write(b"x")
            args = Args(reference_bin=os.path.join(d, "bin"),
                        corpus=os.path.join(d, "corpus"),
                        out=os.path.join(d, "never-written"))
            man = kc.manifest(args, 1, 0.0, 0.1, 1, "boom")
            self.assertEqual(man["capture"]["bytes"], 0)
            self.assertIsNone(man["capture"]["sha256"])


class Cli(unittest.TestCase):
    def test_dry_run_refuses_a_context_that_cannot_cross_the_boundary(self):
        with self.assertRaises(kc.CaptureRefusal):
            kc.main(["--reference-bin", "b", "--model", "m", "--corpus", "c",
                     "--out", "o", "--n-ctx", "2048", "--dry-run"])

    def test_dry_run_accepts_a_context_that_can(self):
        self.assertEqual(
            kc.main(["--reference-bin", "b", "--model", "m", "--corpus", "c",
                     "--out", "o", "--n-ctx", "4096", "--dry-run"]), 0)

    def test_one_chunk_is_refused_because_it_captures_nothing(self):
        with self.assertRaises(kc.CaptureRefusal):
            kc.main(["--reference-bin", "b", "--model", "m", "--corpus", "c",
                     "--out", "o", "--n-ctx", "4096", "--chunks", "1",
                     "--dry-run"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
