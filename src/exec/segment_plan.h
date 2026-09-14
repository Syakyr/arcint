#pragma once

// THE SEGMENTED RUNTIME, DEVICE-FREE HALF (window-051 §2). A depth cut of
// the model into K serving-shape IRs, each its own compiled model, each
// carrying at least one full-attention layer (SDPAToPagedAttention needs
// one v13::ScaledDotProductAttention or it refuses the segment). Segment k
// takes `hidden_out` from segment k-1 as `inputs_embeds` (segment 0 takes
// the H-wide embedding, every other segment the hc*H-wide hyper-connection
// state); the last segment carries the final mixer and `logits`. Expert
// bodies are u8 PORTS, not Constants (tools/q4e/serving_shape.py
// ExpertPortSink): one buffer set of `slots_per_segment` slots is refilled
// from `expert_bodies.u8` before each segment runs, so the same buffer set
// serves every segment's bodies in turn -- legal ONLY because a kind's body
// shape (gate / up / down) is identical across every layer and every
// segment (checked below, its own refusal: shape drift is not caught by
// any of the counting checks).
//
// This header parses and validates `serving-shape.json` (the schema
// tools/export_serving_artifact.py writes at tools/export_serving_artifact.py
// lines 326-512, `tools/q4e/serving_shape.py build_serving_shape_ir`'s
// report going into `segments[]` and `expert_bodies`), and derives the
// runtime bookkeeping (slot assignment, refill order, the chain hash, and
// the one-buffer-set race check) a driver needs. It is pure and
// device-free -- no OpenVINO include -- exactly as `exec/ngram_ports.h` is
// the device-free half of the n-gram port contract.

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "util/log.h"
#include "util/sha256.h"

namespace lgc::segplan {

inline constexpr const char* kHiddenOut  = "hidden_out";
inline constexpr const char* kBodySuffix = "/weight_u8";
inline constexpr const char* kTablePrefix = "ngram_table.";
// Expert body kinds, in the order the exporter emits them a layer
// (tools/q4e/serving_shape.py emit_moe_tiled: gate, up, down) -- the same
// order `kind_index` and `slot_shape` index by.
inline constexpr std::array<const char*, 3> kKinds = {"gate", "up", "down"};

inline int kind_index_of(const std::string& kind) {
    for (size_t i = 0; i < kKinds.size(); ++i) {
        if (kind == kKinds[i]) return static_cast<int>(i);
    }
    return -1;
}

// One compiled segment, as the manifest's `segments[]` row describes it.
struct SegmentSpec {
    int index = 0;
    int lo = 0, hi = 0;
    bool first = false;
    bool last  = false;
    bool has_ple = false;
    int in_width    = 0;  // inputs_embeds_width
    int attn_layers = 0;
    std::vector<std::string> outputs;         // output port names, in order
    bool declares_ngram_ports = false;         // an `ngram_table.K` input
};

// One expert body, as the manifest's `expert_bodies.entries[]` row
// describes it. `layer` is the GLOBAL layer index (the body name
// `layer{i}/moe/experts_{kind}/weight_u8` carries it), not relative to the
// segment.
struct BodySpec {
    std::string name;
    int segment = 0;
    int layer   = 0;
    int kind_index = 0;  // index into kKinds
    std::vector<int64_t> shape;
    uint64_t offset = 0;
    uint64_t bytes  = 0;
};

struct RefillOp {
    size_t   slot = 0;
    uint64_t offset = 0;
    uint64_t bytes  = 0;
};

struct Plan {
    std::vector<SegmentSpec> segments;
    // Sorted by (segment, layer, kind_index) -- which is also the
    // exporter's write order (segment 0..K-1, layer lo..hi-1, gate/up/down),
    // so `offset` is ascending in this same order too.
    std::vector<BodySpec> bodies;
    size_t slots_per_segment = 0;              // 3 * max(hi - lo) over segments
    std::array<std::vector<int64_t>, 3> slot_shape;  // per kind, gate/up/down

    // Bytes of one slot's buffer (the kind's body shape; u8-packed, one
    // byte an element).
    size_t buffer_bytes(size_t slot) const {
        const std::vector<int64_t>& shp = slot_shape[slot % 3];
        size_t n = 1;
        for (int64_t d : shp) n *= static_cast<size_t>(d);
        return n;
    }

    // Bytes of the whole buffer set (every slot, once).
    size_t buffer_set_bytes() const {
        size_t total = 0;
        for (size_t s = 0; s < slots_per_segment; ++s) total += buffer_bytes(s);
        return total;
    }
};

namespace detail {

inline std::vector<std::string> output_names(const nlohmann::json& outputs) {
    std::vector<std::string> names;
    for (const auto& o : outputs) {
        if (!o.is_array() || o.empty() || !o.front().is_string()) {
            throw std::runtime_error("segment plan: an 'outputs' row is not [name, dims, type]");
        }
        names.push_back(o.front().get<std::string>());
    }
    return names;
}

}  // namespace detail

// Parses and cross-checks `manifest` (the exporter's serving-shape.json,
// parsed top-level object) against the geometry the loader read from
// config.json. Throws std::runtime_error, naming the offending segment or
// body, on any of: a segment with no full-attention layer; a chain that
// does not tile [0, n_layer) in index order without gap or overlap;
// segment 0 not `first` or the last segment not `last`; an `in_width` that
// disagrees with `hidden_size` (segment 0) or `hc_count * hidden_size`
// (every other segment); a non-last segment whose outputs are not exactly
// `["hidden_out"]` or a last segment whose outputs are not exactly
// `["logits"]`; `has_ple` set on more than one segment; an `ngram_table.`
// input declared outside segment 0; a body count that does not match
// `3 * (hi - lo)` for its segment, a missing kind, or a layer outside
// `[lo, hi)`; `bytes` disagreeing with `product(shape)`; offsets that are
// not contiguous and ascending in write order, or whose sum disagrees with
// `expert_bodies.bytes`; and shape drift -- the same kind (gate / up /
// down) taking a different shape in two segments, which would make the one
// shared buffer set wrong for one of them.
inline Plan plan_segments(const nlohmann::json& manifest, int n_layer, int hidden_size,
                          int hc_count) {
    if (!manifest.contains("segments") || !manifest.at("segments").is_array() ||
        manifest.at("segments").empty()) {
        throw std::runtime_error("segment plan: manifest has no non-empty 'segments' array");
    }

    std::vector<SegmentSpec> segs;
    for (const auto& sj : manifest.at("segments")) {
        SegmentSpec s;
        s.index = sj.at("index").get<int>();
        const auto& layers = sj.at("layers");
        if (!layers.is_array() || layers.size() != 2) {
            throw std::runtime_error(log::format(
                "segment %d: 'layers' must be [lo, hi]", s.index));
        }
        s.lo = layers.at(0).get<int>();
        s.hi = layers.at(1).get<int>();
        s.first = sj.at("first").get<bool>();
        s.last  = sj.at("last").get<bool>();
        s.has_ple = sj.value("has_ple", false);
        s.in_width = sj.at("inputs_embeds_width").get<int>();
        s.attn_layers = sj.at("attn_layers").get<int>();
        s.outputs = detail::output_names(sj.at("outputs"));
        for (const auto& in : sj.at("inputs")) {
            if (!in.is_array() || in.empty() || !in.front().is_string()) continue;
            const std::string name = in.front().get<std::string>();
            if (name.rfind(kTablePrefix, 0) == 0) s.declares_ngram_ports = true;
        }
        segs.push_back(std::move(s));
    }
    std::sort(segs.begin(), segs.end(),
              [](const SegmentSpec& a, const SegmentSpec& b) { return a.index < b.index; });

    // A segment with no full-attention layer (index % 4 == 3): the pass
    // that turns ScaledDotProductAttention into PagedAttention needs one.
    for (const SegmentSpec& s : segs) {
        bool has_attn = false;
        for (int i = s.lo; i < s.hi; ++i) {
            if (i % 4 == 3) { has_attn = true; break; }
        }
        if (!has_attn) {
            throw std::runtime_error(log::format(
                "segment %d (layers %d..%d) holds no full-attention layer (index %% 4 == 3); "
                "SDPAToPagedAttention needs at least one",
                s.index, s.lo, s.hi - 1));
        }
    }

    // The chain: [0, n_layer) tiled in index order, no gap, no overlap, no
    // reorder, and it must reach n_layer (not stop short).
    {
        int expect = 0;
        for (size_t k = 0; k < segs.size(); ++k) {
            const SegmentSpec& s = segs[k];
            if (s.index != static_cast<int>(k)) {
                throw std::runtime_error(log::format(
                    "segment chain out of order: position %zu holds segment index %d",
                    k, s.index));
            }
            if (s.lo > expect) {
                throw std::runtime_error(log::format(
                    "segment chain has a gap: segment %d starts at layer %d, the chain "
                    "reached %d",
                    s.index, s.lo, expect));
            }
            if (s.lo < expect) {
                throw std::runtime_error(log::format(
                    "segment chain overlaps: segment %d starts at layer %d, the chain "
                    "already reached %d",
                    s.index, s.lo, expect));
            }
            expect = s.hi;
        }
        if (expect != n_layer) {
            throw std::runtime_error(log::format(
                "segment chain is short of n_layer: it covers layers 0..%d, n_layer is %d",
                expect - 1, n_layer));
        }
    }

    if (!segs.front().first) {
        throw std::runtime_error(log::format(
            "segment 0 has first=false; the chain's first segment must take inputs_embeds "
            "at H width"));
    }
    if (!segs.back().last) {
        throw std::runtime_error(log::format(
            "segment %d (the last) has last=false; the chain's last segment must carry the "
            "final mixer and logits",
            segs.back().index));
    }

    // in_width: H on segment 0, hc_count * H elsewhere.
    for (const SegmentSpec& s : segs) {
        const int expected = (s.index == 0) ? hidden_size : hc_count * hidden_size;
        if (s.in_width != expected) {
            throw std::runtime_error(log::format(
                "segment %d inputs_embeds_width is %d, expected %d (%s)",
                s.index, s.in_width, expected,
                s.index == 0 ? "hidden_size" : "hc_count * hidden_size"));
        }
    }

    // outputs: exactly ["hidden_out"] on a non-last segment, ["logits"] on
    // the last.
    for (const SegmentSpec& s : segs) {
        const std::vector<std::string> want =
            s.last ? std::vector<std::string>{"logits"} : std::vector<std::string>{kHiddenOut};
        if (s.outputs != want) {
            std::string got;
            for (size_t i = 0; i < s.outputs.size(); ++i) got += (i ? "," : "") + s.outputs[i];
            throw std::runtime_error(log::format(
                "segment %d outputs are [%s], expected [%s]",
                s.index, got.c_str(), s.last ? "logits" : kHiddenOut));
        }
    }

    // has_ple on at most one segment.
    {
        int count = 0;
        for (const SegmentSpec& s : segs) {
            if (s.has_ple) ++count;
        }
        if (count > 1) {
            throw std::runtime_error(log::format(
                "has_ple is set on %d segments, expected at most one", count));
        }
    }

    // ngram_table. inputs only in segment 0.
    for (const SegmentSpec& s : segs) {
        if (s.index != 0 && s.declares_ngram_ports) {
            throw std::runtime_error(log::format(
                "segment %d declares an 'ngram_table.' input; the n-gram table lives in "
                "segment 0 only",
                s.index));
        }
    }

    Plan plan;
    plan.segments = segs;

    // Expert bodies.
    if (!manifest.contains("expert_bodies") || manifest.at("expert_bodies").is_null()) {
        throw std::runtime_error("segment plan: manifest has no 'expert_bodies'");
    }
    const nlohmann::json& eb = manifest.at("expert_bodies");
    for (const auto& e : eb.at("entries")) {
        BodySpec b;
        b.name    = e.at("name").get<std::string>();
        b.segment = e.at("segment").get<int>();
        b.layer   = e.at("layer").get<int>();
        const std::string kind = e.at("kind").get<std::string>();
        b.kind_index = kind_index_of(kind);
        if (b.kind_index < 0) {
            throw std::runtime_error(log::format(
                "body '%s' has kind '%s', not one of gate/up/down", b.name.c_str(),
                kind.c_str()));
        }
        for (const auto& d : e.at("shape")) b.shape.push_back(d.get<int64_t>());
        b.offset = e.at("offset").get<uint64_t>();
        b.bytes  = e.at("bytes").get<uint64_t>();
        plan.bodies.push_back(std::move(b));
    }
    std::sort(plan.bodies.begin(), plan.bodies.end(),
              [](const BodySpec& a, const BodySpec& b) {
                  if (a.segment != b.segment) return a.segment < b.segment;
                  if (a.layer != b.layer) return a.layer < b.layer;
                  return a.kind_index < b.kind_index;
              });

    // Per-segment body count, kind coverage, and layer range.
    for (const SegmentSpec& s : segs) {
        std::vector<std::array<bool, 3>> seen(static_cast<size_t>(s.hi - s.lo),
                                               std::array<bool, 3>{false, false, false});
        int count = 0;
        for (const BodySpec& b : plan.bodies) {
            if (b.segment != s.index) continue;
            ++count;
            if (b.layer < s.lo || b.layer >= s.hi) {
                throw std::runtime_error(log::format(
                    "body '%s' has layer %d, outside segment %d's range [%d, %d)",
                    b.name.c_str(), b.layer, s.index, s.lo, s.hi));
            }
            seen[static_cast<size_t>(b.layer - s.lo)][static_cast<size_t>(b.kind_index)] = true;
        }
        const int expected_count = 3 * (s.hi - s.lo);
        if (count != expected_count) {
            throw std::runtime_error(log::format(
                "segment %d has %d expert bodies, expected 3 * (hi - lo) = %d", s.index, count,
                expected_count));
        }
        for (int li = 0; li < s.hi - s.lo; ++li) {
            for (int k = 0; k < 3; ++k) {
                if (!seen[static_cast<size_t>(li)][static_cast<size_t>(k)]) {
                    throw std::runtime_error(log::format(
                        "segment %d layer %d is missing its '%s' expert body", s.index,
                        s.lo + li, kKinds[k]));
                }
            }
        }
    }

    // bytes == product(shape); offsets contiguous and ascending in write
    // order (segment, layer, kind_index -- the same order the exporter
    // wrote them); sum(bytes) == expert_bodies.bytes.
    {
        uint64_t expect_offset = 0;
        for (const BodySpec& b : plan.bodies) {
            uint64_t want_bytes = 1;
            for (int64_t d : b.shape) want_bytes *= static_cast<uint64_t>(d);
            if (b.bytes != want_bytes) {
                throw std::runtime_error(log::format(
                    "body '%s' is %llu bytes, product(shape) is %llu", b.name.c_str(),
                    static_cast<unsigned long long>(b.bytes),
                    static_cast<unsigned long long>(want_bytes)));
            }
            if (b.offset != expect_offset) {
                throw std::runtime_error(log::format(
                    "body '%s' offset %llu is not contiguous/ascending (expected %llu)",
                    b.name.c_str(), static_cast<unsigned long long>(b.offset),
                    static_cast<unsigned long long>(expect_offset)));
            }
            expect_offset += b.bytes;
        }
        const uint64_t total = eb.at("bytes").get<uint64_t>();
        if (expect_offset != total) {
            throw std::runtime_error(log::format(
                "sum(bytes) %llu disagrees with expert_bodies.bytes %llu",
                static_cast<unsigned long long>(expect_offset),
                static_cast<unsigned long long>(total)));
        }
    }

    // Shape drift: the buffer set has one shape a kind, shared by every
    // layer of every segment. A kind whose body shape differs between two
    // segments (or two layers) would make that one shared buffer wrong for
    // whichever body does not match the first one seen.
    {
        std::array<bool, 3> have{false, false, false};
        for (const BodySpec& b : plan.bodies) {
            const size_t ki = static_cast<size_t>(b.kind_index);
            if (!have[ki]) {
                plan.slot_shape[ki] = b.shape;
                have[ki] = true;
            } else if (b.shape != plan.slot_shape[ki]) {
                throw std::runtime_error(log::format(
                    "shape drift: body '%s' (kind '%s') has a different shape than an earlier "
                    "'%s' body; the one buffer set needs a single shape a kind",
                    b.name.c_str(), kKinds[ki], kKinds[ki]));
            }
        }
    }

    size_t max_span = 0;
    for (const SegmentSpec& s : segs) {
        max_span = std::max(max_span, static_cast<size_t>(s.hi - s.lo));
    }
    plan.slots_per_segment = 3 * max_span;

    return plan;
}

// (layer - lo) * 3 + kind_index -- the slot a body sits in within the one
// buffer set, relative to its own segment's layer range.
inline size_t body_slot(const Plan& plan, const BodySpec& body) {
    for (const SegmentSpec& s : plan.segments) {
        if (s.index == body.segment) {
            return static_cast<size_t>((body.layer - s.lo) * 3 + body.kind_index);
        }
    }
    throw std::runtime_error(log::format(
        "body_slot: body '%s' references unknown segment %d", body.name.c_str(),
        body.segment));
}

// slot -> port name, for every body of `segment`, in slot order. A short
// last segment (hi - lo < max span) only ever populates slots
// [0, 3 * (hi - lo)) -- a strict prefix of [0, slots_per_segment).
inline std::vector<std::pair<size_t, std::string>> bind_list(const Plan& plan, int segment) {
    std::vector<std::pair<size_t, std::string>> out;
    for (const BodySpec& b : plan.bodies) {
        if (b.segment == segment) out.emplace_back(body_slot(plan, b), b.name);
    }
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return out;
}

// The refill for `segment`, in ascending offset order (the order a
// sequential read of `expert_bodies.u8` produces them).
inline std::vector<RefillOp> refill_ops(const Plan& plan, int segment) {
    std::vector<RefillOp> ops;
    for (const BodySpec& b : plan.bodies) {
        if (b.segment == segment) ops.push_back(RefillOp{body_slot(plan, b), b.offset, b.bytes});
    }
    std::sort(ops.begin(), ops.end(),
              [](const RefillOp& a, const RefillOp& b) { return a.offset < b.offset; });
    return ops;
}

// sha256(lowercase hex sha256 of segment 0's xml || ... || segment K-1's),
// hash_prefix'd -- order-sensitive, and even at K == 1 this is NOT the same
// as `hash_prefix(sha256_file(xml))`: it hashes the 64-character hex
// STRING, not the file's bytes.
inline std::string chain_arch_hash(const std::vector<std::string>& segment_xml_sha256_hex) {
    std::string concat;
    concat.reserve(segment_xml_sha256_hex.size() * 64);
    for (const std::string& h : segment_xml_sha256_hex) {
        std::string lower = h;
        for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        concat += lower;
    }
    return hash_prefix(sha256_hex(concat));
}

// The serial-cut race bookkeeping: which segment's bodies the one buffer
// set currently holds, and a hard check before a forward reads them.
struct BufferSetState {
    int resident = -1;

    void refill(int k) { resident = k; }

    void check_before_infer(int k) {
        if (resident != k) {
            throw std::runtime_error(log::format(
                "buffer set holds segment %d, inference for segment %d was about to read it",
                resident, k));
        }
    }
};

}  // namespace lgc::segplan
