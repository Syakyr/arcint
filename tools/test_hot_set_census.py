#!/usr/bin/env python3
"""Unit ladder for tools/hot_set_census.py (stdlib only, device-free).

Red-first: each cell fails against a plausible wrong implementation before it
passes against the one here.

  * parse_trace REFUSES a malformed row (fewer than three fields) where the
    replay loader silently drops it -- a census that drops a row undercounts,
    so the refusal is the point.
  * the canonical summary is a pure function of the trace's own layer-index
    space, sorted, and its `# total,` equals the routed-access count.
  * selection is deterministic, budgeted (exactly `slots_per_layer` per layer
    when the layer has that many distinct experts) and tie-breaks on the
    ASCENDING expert id -- a wrong tie-break is invisible until it moves a
    served answer, so it is pinned here instead.
  * rounds_to_plateau returns the first stable prefix and REFUSES to claim a
    plateau for a ranking that never stabilises (clause V3's failing shape).
  * patch 0013's four-column CSV parses, and a three-column row, a
    wrong `# total,`, and a perturbed join count are all refused or caught
    (the schema blocker the campaign review caught is pinned as a cell).

The fixture is the committed WP7 routing sample
`tools/testdata/qwen4exp_moe_trace_sample.txt` (400 tokens x 48 layers,
top-10), the same trace tools/test_expert_lru_replay.py uses. The full-trace
WP6b reproduction stays a separate check.
"""
import contextlib
import inspect
import io
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hot_set_census as hc  # noqa: E402

FIXTURE = os.path.join(HERE, "testdata", "qwen4exp_moe_trace_sample.txt")


def write_trace(path, rows, header=("# arcint routing trace v1",)):
    lines = list(header)
    for tok, lay, experts in rows:
        lines.append(" ".join([str(tok), str(lay)] + [str(e) for e in experts]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


class TestParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = hc.parse_trace(FIXTURE)

    def test_fixture_shape(self):
        self.assertEqual(len(self.rows), 19200)                    # 400 x 48
        self.assertEqual(len({r[0] for r in self.rows}), 400)
        self.assertEqual(len({r[1] for r in self.rows}), 48)
        self.assertTrue(all(len(r[2]) == 10 for r in self.rows))   # top-10

    def test_comment_and_blank_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            write_trace(p, [(0, 0, [1, 2])], header=("# a comment", "", "# k=v"))
            rows = hc.parse_trace(p)
            self.assertEqual(rows, [(0, 0, [1, 2])])

    def test_malformed_row_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            with open(p, "w") as f:
                f.write("0 0 1 2\n0 1\n")                          # 2-field row
            with self.assertRaises(ValueError):
                hc.parse_trace(p)

    def test_negative_index_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            with open(p, "w") as f:
                f.write("0 0 -1 2\n")
            with self.assertRaises(ValueError):
                hc.parse_trace(p)

    def test_non_integer_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            with open(p, "w") as f:
                f.write("0 0 1 x\n")
            with self.assertRaises(ValueError):
                hc.parse_trace(p)

    def test_multi_pair_provenance_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            write_trace(p, [(0, 0, [1, 2])],
                        header=("# arcint routing trace v1",
                                "# card=none device=cpu depth=48 kv=u8"))
            prov = hc.read_provenance(p)
            self.assertEqual(prov.get("card"), "none")
            self.assertEqual(prov.get("device"), "cpu")
            self.assertEqual(prov.get("depth"), "48")
            self.assertEqual(prov.get("kv"), "u8")

    def test_non_strict_drops_malformed(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            with open(p, "w") as f:
                f.write("0 0 1 2\n0 1\n1 0 3\n")
            rows = hc.parse_trace(p, strict=False)
            self.assertEqual(rows, [(0, 0, [1, 2]), (1, 0, [3])])


class TestSummary(unittest.TestCase):
    def test_summary_is_sorted_and_total_matches(self):
        rows = hc.parse_trace(FIXTURE)
        summ = hc.canonical_summary(rows)
        self.assertEqual(summ, sorted(summ))
        self.assertEqual(sum(c for _l, _e, c in summ),
                         sum(len(r[2]) for r in rows))

    def test_summary_is_pure_of_row_order(self):
        import random
        rows = hc.parse_trace(FIXTURE)
        shuffled = list(rows)
        random.Random(7).shuffle(shuffled)
        self.assertEqual(hc.canonical_summary(rows), hc.canonical_summary(shuffled))

    def test_summary_text_has_four_column_free_header_and_total(self):
        rows = hc.parse_trace(FIXTURE)
        text = hc.write_summary(rows, None, {"card": "none", "device": "cpu"})
        self.assertIn("layer,expert,count", text)
        self.assertIn("# total,", text)
        self.assertIn("# card=none", text)
        self.assertNotIn("weight_offset", text)


class TestSelection(unittest.TestCase):
    def test_budget_is_exact_per_layer(self):
        rows = hc.parse_trace(FIXTURE)
        hot = hc.select_hot_set(rows, 10)
        self.assertEqual(sorted(hot), list(range(48)))
        for lay, es in hot.items():
            self.assertEqual(len(es), 10)
            self.assertEqual(len(set(es)), 10)                     # distinct

    def test_zero_budget_is_empty(self):
        rows = hc.parse_trace(FIXTURE)
        hot = hc.select_hot_set(rows, 0)
        self.assertTrue(all(v == [] for v in hot.values()))

    def test_tie_break_is_ascending_expert_id(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            # expert 7 and expert 3 both routed once -> 3 must win
            write_trace(p, [(0, 0, [7]), (1, 0, [3])])
            hot = hc.select_hot_set(hc.parse_trace(p), 1)
            self.assertEqual(hot[0], [3])

    def test_clearly_hot_expert_beats_barely_routed(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            rows = [(t, 0, [5] if t < 9 else [9]) for t in range(10)]
            write_trace(p, rows)
            hot = hc.select_hot_set(hc.parse_trace(p), 1)
            self.assertEqual(hot[0], [5])

    def test_selection_is_deterministic(self):
        rows = hc.parse_trace(FIXTURE)
        self.assertEqual(hc.select_hot_set(rows, 17), hc.select_hot_set(rows, 17))


class TestCoverage(unittest.TestCase):
    def test_hot_set_cover_exceeds_random_baseline(self):
        rows = hc.parse_trace(FIXTURE)
        for s in (1, 10, 50):
            hot = hc.select_hot_set(rows, s)
            cov = hc.hot_coverage(rows, hot)
            # a random set of s of 512 experts expects s/512 coverage
            self.assertGreater(cov, s / 512.0)

    def test_full_budget_covers_everything(self):
        rows = hc.parse_trace(FIXTURE)
        hot = hc.select_hot_set(rows, 512)
        self.assertAlmostEqual(hc.hot_coverage(rows, hot), 1.0)


class TestPlateau(unittest.TestCase):
    def test_stable_ranking_plateaus_early(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            rows = [(t, 0, [5]) for t in range(16)]                 # always expert 5
            write_trace(p, rows)
            rep = hc.rounds_to_plateau(hc.parse_trace(p), 1)
            self.assertTrue(rep["plateau"])
            self.assertEqual(rep["rounds"], 1)

    def test_never_stabilising_ranking_has_no_plateau(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            # each power-of-two prefix has a NEW argmax, so no two consecutive
            # prefixes are unchanged -> no plateau (clause V3's failing shape)
            rows = [(0, 0, [6]), (1, 0, [5])]
            rows += [(t, 0, [4]) for t in range(2, 4)]
            rows += [(t, 0, [3]) for t in range(4, 8)]
            rows += [(t, 0, [2]) for t in range(8, 16)]
            rows += [(t, 0, [1]) for t in range(16, 32)]
            write_trace(p, rows)
            rep = hc.rounds_to_plateau(hc.parse_trace(p), 1)
            self.assertFalse(rep["plateau"])
            self.assertIsNone(rep["rounds"])
            self.assertEqual([s["changed"] for s in rep["series"][1:]],
                             [True] * (len(rep["series"]) - 1))

    def test_stable_window_below_two_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            write_trace(p, [(t, 0, [5]) for t in range(4)])
            with self.assertRaises(ValueError):
                hc.rounds_to_plateau(hc.parse_trace(p), 1, stable_window=1)

    def test_series_is_monotone_in_prefix_length(self):
        rows = hc.parse_trace(FIXTURE)
        rep = hc.rounds_to_plateau(rows, 10)
        lens = rep["prefix_tokens"]
        self.assertEqual(lens, sorted(lens))
        self.assertEqual(lens, sorted(set(lens)))
        self.assertEqual(lens[-1], 400)


class TestPluginCsv(unittest.TestCase):
    def _write(self, path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_four_column_csv_parses_with_total(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write(os.path.join(d, "h.csv"),
                            "layer,weight_offset,expert,count\n"
                            "0,100,3,2\n0,100,7,1\n1,200,2,4\n# total,7\n")
            rows = hc.read_plugin_csv(p)
            self.assertEqual(rows, [(0, 100, 3, 2), (0, 100, 7, 1), (1, 200, 2, 4)])

    def test_three_column_row_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write(os.path.join(d, "h.csv"),
                            "layer,expert,count\n0,3,2\n")
            with self.assertRaises(ValueError):
                hc.read_plugin_csv(p)

    def test_wrong_total_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write(os.path.join(d, "h.csv"),
                            "layer,weight_offset,expert,count\n0,100,3,2\n# total,99\n")
            with self.assertRaises(ValueError):
                hc.read_plugin_csv(p)

    def test_join_matches_when_counts_agree(self):
        rows = hc.parse_trace(FIXTURE)
        # build a plugin CSV in weight-offset space from the same trace
        layer_keys = {lay: 1000 + lay for lay in range(48)}
        lines = ["layer,weight_offset,expert,count"]
        total = 0
        for lay, e, c in hc.canonical_summary(rows):
            off = layer_keys[lay]
            lines.append(f"{lay},{off},{e},{c}")
            total += c
        lines.append(f"# total,{total}")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "h.csv")
            with open(p, "w") as f:
                f.write("\n".join(lines) + "\n")
            plugin = hc.read_plugin_csv(p)
        _p, _t, mism = hc.join_plugin_to_trace(rows, plugin, layer_keys)
        self.assertEqual(mism, [])

    def test_join_mismatch_fires_on_perturbed_count(self):
        rows = hc.parse_trace(FIXTURE)
        layer_keys = {lay: 1000 + lay for lay in range(48)}
        plugin = [(lay, layer_keys[lay], e, c + 1)
                  for lay, e, c in hc.canonical_summary(rows)]   # every count +1
        _p, _t, mism = hc.join_plugin_to_trace(rows, plugin, layer_keys)
        self.assertTrue(mism)
        self.assertTrue(all(a - b == 1 for _k, a, b in mism))

    def test_seed_text_carries_layer_key_map(self):
        hot = {0: [3, 7], 1: [2]}
        text = hc.seed_text(hot, {0: 1000, 1: 1001})
        self.assertIn("0 3", text)
        self.assertIn("1 2", text)
        self.assertIn("layer_key_by_index", text)


class TestRouterTraceWriter(unittest.TestCase):
    def test_format_v1_token_major_and_ascending(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.trace")
            by_layer = {0: [[9, 3], [4, 1]],
                        1: [[2, 7], [5, 6]]}
            hc.write_router_trace(p, by_layer, 2, {"card": "none", "device": "cpu"})
            rows = hc.parse_trace(p)
            self.assertEqual(rows, [(0, 0, [3, 9]), (0, 1, [2, 7]),
                                    (1, 0, [1, 4]), (1, 1, [5, 6])])
            prov = hc.read_provenance(p)
            self.assertEqual(prov.get("card"), "none")
            self.assertEqual(prov.get("device"), "cpu")

    def test_missing_layer_row_for_a_short_array_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.trace")
            # layer 1 has only one token; the token-2 row must be absent, not
            # an IndexError, and the trace must still parse
            hc.write_router_trace(p, {0: [[1]] * 5, 1: [[2]]}, 5)
            rows = hc.parse_trace(p)
            self.assertEqual(len(rows), 6)          # 5 x layer0 + 1 x layer1


class TestCallTrace(unittest.TestCase):
    def test_split_two_tokens_into_chunks(self):
        self.assertEqual(hc.split_topk_chunks([1, 2, 3, 4], 2), [[1, 2], [3, 4]])
        self.assertEqual(hc.split_topk_chunks([1, 2, 3, 4, 5, 6], 3),
                         [[1, 2, 3], [4, 5, 6]])

    def test_split_missized_is_refused_not_truncated(self):
        with self.assertRaises(ValueError):
            hc.split_topk_chunks([1, 2, 3], 2)
        with self.assertRaises(ValueError):
            hc.split_topk_chunks([], 2)

    def test_split_zero_top_k_is_refused(self):
        with self.assertRaises(ValueError):
            hc.split_topk_chunks([1, 2], 0)

    def test_layer_key_map_is_ascending_export_order(self):
        self.assertEqual(hc.layer_key_index_map([300, 100, 200]),
                         {100: 0, 200: 1, 300: 2})

    def test_explicit_map_non_injective_is_refused(self):
        with self.assertRaises(ValueError):
            hc.layer_key_index_map([100, 200], {0: 100, 1: 200, 2: 200})

    def test_explicit_map_missing_key_is_refused(self):
        with self.assertRaises(ValueError):
            hc.layer_key_index_map([100, 200, 300], {0: 100, 1: 200})

    def test_decode_calls_reconstruct_token_major_v1(self):
        call_rows = [(0, 100, 2, [3, 4]), (1, 200, 2, [5, 6]),
                     (2, 100, 2, [7, 8]), (3, 200, 2, [9, 10])]
        rows, rep = hc.call_trace_to_v1(call_rows)
        self.assertEqual(rows, [(0, 0, [3, 4]), (0, 1, [5, 6]),
                                (1, 0, [7, 8]), (1, 1, [9, 10])])
        self.assertEqual(rep["tokens"], 2)
        self.assertEqual(rep["layer_keys"], 2)
        self.assertNotIn("prefill", rep)

    def test_batched_call_is_refused(self):
        with self.assertRaises(ValueError):
            hc.call_trace_to_v1([(0, 100, 2, [1, 2, 3, 4])])

    def test_skip_batched_keeps_decode_rows_and_counts_the_skip(self):
        # a served trace opens with a batched prefill call: skipped, counted,
        # and the decode rows still convert frame-for-frame. As modelled here
        # the prefill is ONE call, so the decode stream resumes mid-sequence:
        # token 0 = seq 1 (layer 200) + seq 2 (layer 100), token 1 = seq 3
        # (layer 200) ONLY -- token 1 is the partial one. A real depth-48
        # prefill emits one batched call per layer; skipped, they leave the
        # decode stream at layer 0.
        call_rows = [(0, 100, 2, [1, 2, 3, 4]),          # prefill, 2 tokens
                     (1, 200, 2, [5, 6]),
                     (2, 100, 2, [7, 8]),
                     (3, 200, 2, [9, 10])]
        rows, rep = hc.call_trace_to_v1(call_rows, skip_batched=True)
        self.assertEqual(rows, [(0, 0, [7, 8]), (0, 1, [5, 6]), (1, 1, [9, 10])])
        self.assertEqual(rep["batched_calls_skipped"], 1)
        self.assertEqual(rep["batched_tokens_skipped"], 2)

    def test_all_batched_is_refused_even_with_skip_batched(self):
        # the skip must not turn a prefill-only trace into an empty census.
        with self.assertRaises(ValueError):
            hc.call_trace_to_v1([(0, 100, 2, [1, 2, 3, 4])], skip_batched=True)

    def test_skip_batched_default_is_off(self):
        # the name says the DEFAULT is off, so pin the default itself: the
        # empty-list count would be zero whichever way it were set.
        sig = inspect.signature(hc.call_trace_to_v1)
        self.assertIs(sig.parameters["skip_batched"].default, False)

    def test_empty_call_trace_reports_zero_tokens(self):
        rows, rep = hc.call_trace_to_v1([])
        self.assertEqual(rows, [])
        self.assertEqual(rep["tokens"], 0)

    def test_call_trace_parse_refuses_negative(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.trace")
            with open(p, "w") as f:
                f.write("0 -5 2 3 4\n")
            with self.assertRaises(ValueError):
                hc.parse_call_trace(p)

    def test_call_trace_parse_refuses_missing_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.trace")
            with open(p, "w") as f:
                f.write("0 100 2\n")          # top_k declared, no ids
            with self.assertRaises(ValueError):
                hc.parse_call_trace(p)

    def test_from_call_trace_cli_writes_format_v1(self):
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            out = os.path.join(d, "v1.trace")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 3 4\n1 200 2 5 6\n2 100 2 7 8\n3 200 2 9 10\n")
            with open(prov, "w") as f:
                f.write("# artifact=d48n card=A770 kv=u8\n")
            rc = hc.main(["from-call-trace", "--call-trace", calls,
                          "--provenance", prov, "--out", out])
            self.assertEqual(rc, 0)
            rows = hc.parse_trace(out)
            self.assertEqual(rows, [(0, 0, [3, 4]), (0, 1, [5, 6]),
                                    (1, 0, [7, 8]), (1, 1, [9, 10])])
            prov_out = hc.read_provenance(out)
            self.assertEqual(prov_out["artifact"], "d48n")
            self.assertEqual(prov_out["card"], "A770")

    def test_from_call_trace_cli_refuses_missing_provenance(self):
        # an empty provenance file carries neither artifact= nor card=.
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 3 4\n")
            with open(prov, "w") as f:
                f.write("")
            with self.assertRaises(ValueError):
                hc.main(["from-call-trace", "--call-trace", calls,
                         "--provenance", prov])

    def test_from_call_trace_cli_refuses_incomplete_provenance(self):
        # artifact without card is still unattributable: refused, not written.
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 3 4\n")
            with open(prov, "w") as f:
                f.write("# artifact=d48n\n")
            with self.assertRaises(ValueError):
                hc.main(["from-call-trace", "--call-trace", calls,
                         "--provenance", prov])

    def test_from_call_trace_cli_refuses_empty_provenance_values(self):
        # an empty value is not an attribution: refused, not written.
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 3 4\n")
            with open(prov, "w") as f:
                f.write("# artifact_sha256= card=\n")
            with self.assertRaises(ValueError):
                hc.main(["from-call-trace", "--call-trace", calls,
                         "--provenance", prov])

    def test_from_call_trace_cli_accepts_the_artifact_sha256_spelling(self):
        # §2's canonical key is artifact_sha256=; both spellings are accepted
        # and the emitted header carries the one that was injected.
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            out = os.path.join(d, "v1.trace")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 3 4\n1 200 2 5 6\n")
            with open(prov, "w") as f:
                f.write("# artifact_sha256=deadbeef card=8086:56A0\n")
            rc = hc.main(["from-call-trace", "--call-trace", calls,
                          "--provenance", prov, "--out", out])
            self.assertEqual(rc, 0)
            prov_out = hc.read_provenance(out)
            self.assertEqual(prov_out["artifact_sha256"], "deadbeef")
            self.assertEqual(prov_out["token_labels"], "reconstructed")

    def test_from_call_trace_cli_stdout_is_a_valid_v1_trace(self):
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 3 4\n1 200 2 5 6\n")
            with open(prov, "w") as f:
                f.write("# artifact=d48n card=A770\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = hc.main(["from-call-trace", "--call-trace", calls,
                              "--provenance", prov])
            self.assertEqual(rc, 0)
            out = os.path.join(d, "stdout.trace")
            with open(out, "w") as f:
                f.write(buf.getvalue())
            self.assertEqual(hc.read_provenance(out)["card"], "A770")
            self.assertEqual(len(hc.parse_trace(out)), 2)

    def test_from_call_trace_cli_skip_batched_reports_the_skip(self):
        with tempfile.TemporaryDirectory() as d:
            calls = os.path.join(d, "c.trace")
            out = os.path.join(d, "v1.trace")
            prov = os.path.join(d, "prov.txt")
            with open(calls, "w") as f:
                f.write("0 100 2 1 2 3 4\n1 200 2 5 6\n2 100 2 7 8\n3 200 2 9 10\n")
            with open(prov, "w") as f:
                f.write("# artifact=d48n card=A770\n")
            rc = hc.main(["from-call-trace", "--call-trace", calls,
                          "--provenance", prov, "--skip-batched", "--out", out])
            self.assertEqual(rc, 0)
            prov_out = hc.read_provenance(out)
            self.assertEqual(prov_out["batched_calls_skipped"], "1")
            self.assertEqual(prov_out["batched_tokens_skipped"], "2")
            self.assertEqual(len(hc.parse_trace(out)), 3)


class TestCli(unittest.TestCase):
    def test_summary_subcommand_writes_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "s.csv")
            rc = hc.main(["summary", "--trace", FIXTURE, "--out", out])
            self.assertEqual(rc, 0)
            text = open(out).read()
            self.assertIn("layer,expert,count", text)
            self.assertIn("# total,", text)

    def test_select_subcommand_is_parseable(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "seed.txt")
            rc = hc.main(["select", "--trace", FIXTURE,
                          "--slots-per-layer", "4", "--out", out])
            self.assertEqual(rc, 0)
            data = [l for l in open(out) if l.strip() and not l.startswith("#")]
            self.assertEqual(len(data), 48 * 4)          # 48 layers x 4 slots


if __name__ == "__main__":
    unittest.main()
