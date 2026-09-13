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


class ThePriceFormula(unittest.TestCase):
    """How many rows of a T-token window the QSA price TOUCHES. Pinned to the
    two measured points, because reasoning about index arithmetic produced the
    wrong answer here once (`T - 1 - boundary`, which gives 0 and 28)."""

    MEASURED_ROWS = {2052: 1, 2080: 29}

    def test_the_formula_reproduces_both_measured_rows(self):
        for T, affected in self.MEASURED_ROWS.items():
            self.assertEqual(kc.rows_past_boundary(T), affected,
                             f"T={T}: the recorded measurement is "
                             f"{affected}/{T} rows")

    def test_the_boundary_is_not_the_budget(self):
        """2048 is the QSA budget; 2051 is the measured boundary."""
        self.assertEqual(kc.QSA_BOUNDARY_TOKENS, 2051)
        self.assertEqual(kc.rows_past_boundary(2048), 0)


class WhatACaptureContains(unittest.TestCase):
    """`llama-perplexity` records only the SECOND HALF of each window. These
    figures come from a real capture, not from the source alone."""

    # MEASURED: a `-c 4096 --chunks 2` capture of the shipped artifact.
    # File layout is "_logits_" + n_ctx + n_vocab + n_chunk + token ids, then
    # one row of 2*((n_vocab+1)//2)+4 uint16 per recorded token.
    CAPTURE_BYTES = 2_033_309_700
    CAPTURE_N_CTX = 4096
    CAPTURE_N_CHUNK = 2
    CAPTURE_N_VOCAB = 248_320

    def test_the_recorded_row_count_matches_the_measured_file_exactly(self):
        nv = 2 * ((self.CAPTURE_N_VOCAB + 1) // 2) + 4
        header = 8 + 4 + 4 + 4 + self.CAPTURE_N_CHUNK * self.CAPTURE_N_CTX * 4
        body = self.CAPTURE_BYTES - header
        self.assertEqual(body % (nv * 2), 0,
                         "the file does not divide into whole rows; the "
                         "layout assumed here is wrong")
        rows_in_file = body // (nv * 2)
        per_window = kc.recorded_rows(self.CAPTURE_N_CTX)[2]
        self.assertEqual(rows_in_file, self.CAPTURE_N_CHUNK * per_window,
                         f"the file holds {rows_in_file} rows; the model says "
                         f"{self.CAPTURE_N_CHUNK} x {per_window}")

    def test_the_recorded_range_is_the_second_half(self):
        first, last, count = kc.recorded_rows(4096)
        self.assertEqual((first, last, count), (2048, 4094, 2047))

    def test_the_capture_width_that_looks_fine_and_is_not(self):
        """THE DEFECT THIS MODEL EXISTS FOR. A 4096 window obviously crosses
        the boundary, so a check on the WINDOW passes it -- and the capture
        holds three usable rows below the boundary against 2044 above."""
        below, at_or_above = kc.rows_by_side(4096)
        self.assertEqual((below, at_or_above), (3, 2044))
        with self.assertRaises(kc.CaptureRefusal):
            kc.check_capture_serves_both_sides(4096, 256)

    def test_a_window_that_records_nothing_above_the_boundary(self):
        below, at_or_above = kc.rows_by_side(2048)
        self.assertEqual(at_or_above, 0)
        with self.assertRaises(kc.CaptureRefusal):
            kc.check_capture_serves_both_sides(2048, 1)

    def test_the_balanced_width_really_maximises_the_scarcer_side(self):
        best_n, best_rows = kc.balanced_n_ctx()
        self.assertEqual(min(kc.rows_by_side(best_n)), best_rows)
        for n in range(kc.QSA_BOUNDARY_TOKENS + 2, 2 * kc.QSA_BOUNDARY_TOKENS):
            self.assertLessEqual(min(kc.rows_by_side(n)), best_rows,
                                 f"n_ctx={n} beats the balanced width")

    def test_a_balanced_width_is_accepted_and_reports_both_sides(self):
        best_n, _ = kc.balanced_n_ctx()
        below, at_or_above = kc.check_capture_serves_both_sides(best_n, 256)
        self.assertGreaterEqual(below, 256)
        self.assertGreaterEqual(at_or_above, 256)

    def test_the_refusal_names_the_boundary_and_a_usable_width(self):
        with self.assertRaises(kc.CaptureRefusal) as caught:
            kc.check_capture_serves_both_sides(4096, 256)
        msg = str(caught.exception)
        self.assertIn(str(kc.QSA_BOUNDARY_TOKENS), msg)
        self.assertIn(str(kc.balanced_n_ctx()[0]), msg)
        self.assertIn("2.385560e-02", msg)
        self.assertIn("SECOND HALF", msg)


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
        cmd = kc.build_command(Args(n_ctx=2736, chunks=3, batch=256, threads=4))
        for flag, want in (("-c", "2736"), ("--chunks", "3"),
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
            man = kc.manifest(args, (3, 2044), 0.0, 1.5, 0, "tail")
            self.assertEqual(man["capture"]["bytes"], len(b"logits"))
            self.assertEqual(len(man["capture"]["sha256"]), 64)
            self.assertEqual(len(man["corpus"]["sha256"]), 64)
            self.assertEqual(len(man["reference"]["binary_sha256"]), 64)
            self.assertIsNone(man["model"]["sha256"])   # --hash-model off
            self.assertEqual(man["qsa_boundary_tokens"], 2051)
            self.assertEqual(man["recorded_rows_below_boundary"], 3)
            self.assertEqual(man["recorded_rows_at_or_above_boundary"], 2044)
            self.assertEqual(man["recorded_rows_per_window"], 2047)
            json.dumps(man)                              # must be serialisable

    def test_a_missing_capture_file_is_recorded_as_absent_not_as_zero_hash(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("bin", "corpus"):
                with open(os.path.join(d, name), "wb") as fh:
                    fh.write(b"x")
            args = Args(reference_bin=os.path.join(d, "bin"),
                        corpus=os.path.join(d, "corpus"),
                        out=os.path.join(d, "never-written"))
            man = kc.manifest(args, (1, 1), 0.0, 0.1, 1, "boom")
            self.assertEqual(man["capture"]["bytes"], 0)
            self.assertIsNone(man["capture"]["sha256"])


class Cli(unittest.TestCase):
    BASE = ["--reference-bin", "b", "--model", "m", "--corpus", "c", "--out", "o"]

    def test_dry_run_refuses_a_width_whose_rows_land_on_one_side(self):
        with self.assertRaises(kc.CaptureRefusal):
            kc.main(self.BASE + ["--n-ctx", "2048", "--dry-run"])

    def test_dry_run_refuses_the_width_that_merely_crosses_the_boundary(self):
        """4096 is the width the first version of this tool accepted."""
        with self.assertRaises(kc.CaptureRefusal):
            kc.main(self.BASE + ["--n-ctx", "4096", "--dry-run"])

    def test_dry_run_accepts_the_balanced_width(self):
        best_n, _ = kc.balanced_n_ctx()
        self.assertEqual(
            kc.main(self.BASE + ["--n-ctx", str(best_n), "--dry-run"]), 0)

    def test_the_minimum_each_side_is_the_callers(self):
        """4096 is refused at the default and admitted at 3, because 3 is what
        it has below the boundary. The knob is not a way to pretend."""
        with self.assertRaises(kc.CaptureRefusal):
            kc.main(self.BASE + ["--n-ctx", "4096", "--dry-run"])
        self.assertEqual(
            kc.main(self.BASE + ["--n-ctx", "4096", "--min-rows-each-side",
                                 "3", "--dry-run"]), 0)

    def test_one_chunk_is_allowed_over_a_long_enough_corpus(self):
        """It was refused, on a rationale that belonged to the corpus."""
        best_n, _ = kc.balanced_n_ctx()
        with tempfile.TemporaryDirectory() as d:
            corpus = os.path.join(d, "corpus")
            with open(corpus, "wb") as fh:
                fh.write(b"x" * (8 * best_n))
            self.assertEqual(
                kc.main(["--reference-bin", "b", "--model", "m", "--corpus",
                         corpus, "--out", "o", "--n-ctx", str(best_n),
                         "--chunks", "1", "--dry-run"]), 0)

    def test_zero_chunks_is_refused(self):
        best_n, _ = kc.balanced_n_ctx()
        with self.assertRaises(kc.CaptureRefusal):
            kc.main(self.BASE + ["--n-ctx", str(best_n), "--chunks", "0",
                                 "--dry-run"])


class CorpusLength(unittest.TestCase):
    def test_a_corpus_that_cannot_reach_two_windows_is_refused(self):
        """A token is at least one byte, so this bound cannot be wrong."""
        with tempfile.TemporaryDirectory() as d:
            corpus = os.path.join(d, "corpus")
            with open(corpus, "wb") as fh:
                fh.write(b"x" * 100)
            with self.assertRaises(kc.CaptureRefusal) as caught:
                kc.check_corpus_long_enough(corpus, 2736)
            self.assertIn("5,472", str(caught.exception))

    def test_a_corpus_that_could_reach_it_is_not_refused(self):
        with tempfile.TemporaryDirectory() as d:
            corpus = os.path.join(d, "corpus")
            with open(corpus, "wb") as fh:
                fh.write(b"x" * 200_000)
            self.assertEqual(kc.check_corpus_long_enough(corpus, 2736), 200_000)

    def test_a_missing_corpus_is_the_runners_error_not_this_check(self):
        self.assertIsNone(
            kc.check_corpus_long_enough("/nonexistent/corpus", 2736))


if __name__ == "__main__":
    unittest.main(verbosity=2)
