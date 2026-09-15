"""tools/probe_segment_residency.py, device-free (B.3).

The probe's job on a card is to say which POOL each of a segmented
artifact's three big allocations lands in. Everything it decides BEFORE it
touches a device -- which directories form the chain, which ports are the
table and which are expert slots, and what one buffer set costs -- is
arithmetic over an xml, and that is what these cells hold.

Red first: every cell here fails against the parent commit, where
`tools/probe_segment_residency.py` does not exist (ModuleNotFoundError /
"can't open file"). The REFUSAL cells are red a second way -- each one was
re-run with its own guard deleted from the probe, and the named refusal
stops firing:

  - discover_segments' contiguity check deleted -> a chain with segment0 +
    segment2 loads as a 2-segment chain and the cell goes green-wrong.
  - plan()'s "only the first segment may declare table ports" check
    deleted -> a chain whose segment1 carries a table returns a plan.
  - the MAX in plan()'s buffer_set_bytes replaced by sum() -> the buffer
    set of a 2-segment chain doubles and the cell reads the sum.

The synthetic artifacts here are tiny (tens of bytes per port) and carry
the SERVED path's port NAMES only; nothing here is the served path and no
cell compiles or runs anything.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("openvino")

import openvino as ov                                       # noqa: E402
from openvino import opset13 as opset                       # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE = REPO_ROOT / "tools" / "probe_segment_residency.py"
sys.path.insert(0, str(REPO_ROOT / "tools"))

import probe_segment_residency as psr                       # noqa: E402

# the real artifact's shapes, divided down: the NAMES are what the probe
# keys on, the sizes only have to be arithmetic a cell can check by hand.
GATE = [2, 4, 2, 4]          # 64 B at u8
UP = [2, 4, 2, 4]            # 64 B
DOWN = [2, 8, 1, 4]          # 64 B
EXPERT_BYTES_PER_LAYER = 3 * 64
TABLE_ROW_BYTES = 90


def seg_model(layers, table_rows=(), hidden=8):
    """One segment's port surface: inputs_embeds, the expert slots of its
    own layer range, and (PLE segment only) the ngram_table chunks."""
    params = []
    emb = opset.parameter([1, -1, hidden], ov.Type.f32)
    emb.set_friendly_name("inputs_embeds")
    emb.output(0).set_names({"inputs_embeds"})
    params.append(emb)
    for i, rows in enumerate(table_rows):
        t = opset.parameter([rows, TABLE_ROW_BYTES], ov.Type.u8)
        t.set_friendly_name(f"ngram_table.{i}")
        t.output(0).set_names({f"ngram_table.{i}"})
        params.append(t)
    for layer in layers:
        for kind, shape in (("gate", GATE), ("up", UP), ("down", DOWN)):
            e = opset.parameter(shape, ov.Type.u8)
            name = f"layer{layer}/moe/experts_{kind}/weight_u8"
            e.set_friendly_name(name)
            e.output(0).set_names({name})
            params.append(e)
    res = opset.result(opset.add(emb, emb))
    res.set_friendly_name("hidden_out")
    return ov.Model([res], params, "segment")


def write_artifact(root, segments):
    """segments = [(layers, table_rows), ...] in segment order."""
    root = Path(root)
    for k, (layers, table_rows) in enumerate(segments):
        d = root / f"segment{k}"
        d.mkdir(parents=True, exist_ok=True)
        ov.save_model(seg_model(layers, table_rows),
                      str(d / "openvino_language_model.xml"), compress_to_fp16=False)
    return root


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """A 2-segment chain shaped like the real one: the table on segment 0
    only, four layers of expert slots per segment."""
    root = tmp_path_factory.mktemp("seg-chain")
    return write_artifact(root, [([0, 1, 2, 3], (5, 3)), ([4, 5, 6, 7], ())])


def censuses_of(root):
    core = ov.Core()
    out = []
    for d in psr.discover_segments(root):
        out.append(psr.port_census(core.read_model(str(d / "openvino_language_model.xml"))))
    return out


# ---- the chain's identity ---------------------------------------------------

def test_segments_are_discovered_in_segment_order(chain):
    got = [d.name for d in psr.discover_segments(chain)]
    assert got == ["segment0", "segment1"]


def test_a_gap_in_the_segment_indices_is_refused(tmp_path):
    """REFUSAL: a chain is its order. segment0 + segment2 is not a
    2-segment chain, it is a chain with segment1 missing, and loading it
    as the former would serve 8 layers under a 12-layer pin."""
    write_artifact(tmp_path, [([0], ()), ([1], ())])
    (tmp_path / "segment1").rename(tmp_path / "segment2")
    with pytest.raises(ValueError, match="not.*contiguous"):
        psr.discover_segments(tmp_path)


def test_a_directory_with_no_segments_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a segmented artifact"):
        psr.discover_segments(tmp_path)


# ---- the port census --------------------------------------------------------

def test_the_census_separates_table_expert_and_boundary_ports(chain):
    c = censuses_of(chain)
    assert [n for n, _, _ in c[0]["table"]] == ["ngram_table.0", "ngram_table.1"]
    assert len(c[0]["experts"]) == 12                     # 4 layers x 3 kinds
    assert [n for n, _, _ in c[0]["other"]] == ["inputs_embeds"]
    assert c[1]["table"] == []                            # only the PLE segment
    assert len(c[1]["experts"]) == 12


def test_the_table_ports_are_censused_in_chunk_order_not_string_order(tmp_path):
    """ngram_table.10 sorts before ngram_table.2 as a string; the table is
    a contiguous row range split into chunks, so the ORDER is the chunk
    index and a string sort would bind rows to the wrong offsets."""
    write_artifact(tmp_path, [([0], tuple(1 for _ in range(11)))])
    c = censuses_of(tmp_path)[0]
    assert [n for n, _, _ in c["table"]] == [f"ngram_table.{i}" for i in range(11)]


# ---- the plan ---------------------------------------------------------------

def test_the_plan_prices_the_table_off_the_ports(chain):
    p = psr.plan(censuses_of(chain))
    assert p["table_ports"] == 2
    assert p["table_bytes"] == (5 + 3) * TABLE_ROW_BYTES


def test_one_buffer_set_is_the_MAX_over_segments_not_the_sum(tmp_path):
    """The design's buffer set is ONE set, refilled per segment. A chain
    whose second segment is larger needs the larger set once -- not both.
    With sum() here the number doubles and a 14.06 GiB set reads 28."""
    write_artifact(tmp_path, [([0], ()), ([1, 2, 3], ())])
    p = psr.plan(censuses_of(tmp_path))
    assert p["expert_bytes_per_segment"] == [EXPERT_BYTES_PER_LAYER,
                                             3 * EXPERT_BYTES_PER_LAYER]
    assert p["buffer_set_bytes"] == 3 * EXPERT_BYTES_PER_LAYER
    assert p["blob_bytes"] == 4 * EXPERT_BYTES_PER_LAYER    # the per-forward read


def test_a_table_on_a_later_segment_is_refused(tmp_path):
    """REFUSAL: the table is the PLE's, and the PLE rides in segment 0. A
    second segment declaring ngram_table ports would need a second 26.8
    GiB binding, which the host budget has no room for -- the probe must
    say so rather than price one table and bind two."""
    write_artifact(tmp_path, [([0], (2,)), ([1], (2,))])
    with pytest.raises(ValueError, match="only the PLE-carrying segment"):
        psr.plan(censuses_of(tmp_path))


# ---- the tool itself, device-free -------------------------------------------

def run_probe(*args):
    p = subprocess.run([sys.executable, str(PROBE), *args],
                       capture_output=True, text=True, timeout=600)
    return p.returncode, p.stdout + p.stderr


def test_dry_run_prints_the_plan_and_allocates_nothing(chain, tmp_path):
    js = tmp_path / "dry.json"
    rc, out = run_probe("--artifact", str(chain), "--dry-run", "--json", str(js))
    assert rc == 0, out
    assert "CHAIN 2 segment(s)" in out
    assert "DRY RUN, nothing allocated" in out
    assert "RESID [mem]" not in out          # no sample without a device
    assert not js.exists()                   # --json is the card leg's record


def test_dry_run_names_the_two_pools_it_will_never_sum(chain):
    """The sentence this whole probe exists to stop being implied."""
    rc, out = run_probe("--artifact", str(chain), "--dry-run")
    assert rc == 0, out
    assert "THE TWO POOLS, never summed" in out
    assert "invisible to the cgroup" in out


def test_a_partial_chain_says_it_is_not_the_model(chain):
    rc, out = run_probe("--artifact", str(chain), "--dry-run", "--segments", "0")
    assert rc == 0, out
    assert "PARTIAL CHAIN" in out and "NOT the model" in out
    assert "CHAIN 1 segment(s)" in out


def test_a_segment_index_the_artifact_does_not_have_is_refused(chain):
    rc, out = run_probe("--artifact", str(chain), "--dry-run", "--segments", "0,7")
    assert rc == 2, out
    assert "REFUSED segment(s) [7] do not exist" in out


def test_a_gap_refusal_exits_2_from_the_tool(tmp_path):
    write_artifact(tmp_path, [([0], ()), ([1], ())])
    (tmp_path / "segment1").rename(tmp_path / "segment2")
    rc, out = run_probe("--artifact", str(tmp_path), "--dry-run")
    assert rc == 2, out
    assert "REFUSED" in out and "contiguous" in out
