// THE SEGMENTED RUNTIME, DEVICE-FREE HALF (window-051 §2, src/exec/segment_plan.h).
// Fixtures restate the real geometry, in-memory, in the exporter's own
// serving-shape.json schema (tools/export_serving_artifact.py:326-512,
// tools/q4e/serving_shape.py build_serving_shape_ir's report): 48 layers,
// 4 segments of 12, H = 2560, hc_count = 4 (hc * H = 10240, window-051 §2's
// hidden-state width), gate/up expert bodies [512, 640, 20, 64], down
// [512, 2560, 5, 64], every body 419,430,400 bytes (512*640*20*64 ==
// 512*2560*5*64) -- so a 12-layer segment's buffer set is
// 12 * 3 * 419,430,400 = 15,099,494,400 bytes.
#include "core/artifact.h"
#include "exec/segment_plan.h"
#include "harness.h"
#include "util/sha256.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace lgc;
using json = nlohmann::json;

namespace {

constexpr int      kNLayer      = 48;
constexpr int       kSegLayers   = 12;
constexpr int       kHidden      = 2560;
constexpr int       kHcCount     = 4;
constexpr uint64_t  kBodyBytes   = 419430400ULL;
constexpr uint64_t  kSegBufBytes = 12ULL * 3ULL * kBodyBytes;  // 15,099,494,400

std::vector<int64_t> kind_shape(const std::string& kind) {
    return kind == "down" ? std::vector<int64_t>{512, 2560, 5, 64}
                          : std::vector<int64_t>{512, 640, 20, 64};
}

// One segment row, layers [lo, hi). `attn_layers` and outputs/inputs follow
// the real geometry: has_ple only on the segment holding global layer 1
// (always segment 0 at this cut); the ngram ports live only there too.
json segment_row(int index, int lo, int hi, int n_layer) {
    const bool first = (lo == 0);
    const bool last  = (hi == n_layer);
    const bool has_ple = (lo <= 1 && 1 < hi);
    const int in_width = first ? kHidden : kHcCount * kHidden;
    int attn_layers = 0;
    for (int i = lo; i < hi; ++i) {
        if (i % 4 == 3) ++attn_layers;
    }
    json inputs = json::array();
    inputs.push_back({"inputs_embeds", {1, -1, in_width}, "f32"});
    inputs.push_back({"position_ids", {1, -1}, "i64"});
    if (has_ple) {
        inputs.push_back({"ngram_chunk_ids", {1, -1, 16}, "i32"});
        inputs.push_back({"ngram_local_ids", {1, -1, 16}, "i64"});
        inputs.push_back({"ngram_table.0", {47718400, 90}, "u8"});
    }
    inputs.push_back({"conv_mask", {1, -1}, "f32"});
    inputs.push_back({"attention_mask", {1, -1}, "i64"});
    inputs.push_back({"beam_idx", {-1}, "i32"});

    json outputs = json::array();
    if (last) {
        outputs.push_back({"logits", {1, -1, 151936}, "f32"});
    } else {
        outputs.push_back({"hidden_out", {1, -1, kHcCount * kHidden}, "f32"});
    }

    return json{
        {"index", index},          {"dir", "segment" + std::to_string(index)},
        {"layers", {lo, hi}},      {"first", first},
        {"last", last},            {"inputs_embeds_width", in_width},
        {"has_ple", has_ple},      {"nodes", 1},
        {"gdn_layers", (hi - lo) - attn_layers},
        {"attn_layers", attn_layers},
        {"inputs", inputs},        {"outputs", outputs},
        {"lm_bin_bytes", 1024},
    };
}

// The full 4x12 manifest over 48 layers, expert bodies filled for every
// segment. `mutate_segments`/`mutate_bodies` let a case perturb one field
// after the fact without re-deriving the whole fixture.
json full_manifest(int n_layer = kNLayer, int seg_layers = kSegLayers) {
    json segments = json::array();
    json entries  = json::array();
    uint64_t offset = 0;
    for (int lo = 0, k = 0; lo < n_layer; lo += seg_layers, ++k) {
        const int hi = std::min(lo + seg_layers, n_layer);
        segments.push_back(segment_row(k, lo, hi, n_layer));
        for (int i = lo; i < hi; ++i) {
            for (const std::string& kind : {std::string("gate"), std::string("up"),
                                            std::string("down")}) {
                json e = {
                    {"name", "layer" + std::to_string(i) + "/moe/experts_" + kind + "/weight_u8"},
                    {"segment", k},   {"layer", i}, {"kind", kind},
                    {"shape", kind_shape(kind)},
                    {"offset", offset}, {"bytes", kBodyBytes},
                };
                entries.push_back(e);
                offset += kBodyBytes;
            }
        }
    }
    return json{
        {"segment_layers", seg_layers},
        {"segments", segments},
        {"expert_bodies", {{"file", "expert_bodies.u8"}, {"bytes", offset}, {"entries", entries}}},
    };
}

std::string refusal(const json& manifest, int n_layer = kNLayer, int hidden = kHidden,
                    int hc = kHcCount) {
    try {
        (void)segplan::plan_segments(manifest, n_layer, hidden, hc);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return "";
}

}  // namespace

// -------------------------------------------------------------- (a), (b)

TEST(segments_full_fixture_offsets_are_contiguous_and_the_12layer_buffer_set_matches) {
    const json manifest = full_manifest();
    const segplan::Plan plan = segplan::plan_segments(manifest, kNLayer, kHidden, kHcCount);
    CHECK_EQ(plan.bodies.size(), static_cast<size_t>(kNLayer * 3));
    uint64_t expect = 0;
    for (const auto& b : plan.bodies) {
        CHECK_EQ(b.offset, expect);
        expect += b.bytes;
    }
    CHECK_EQ(expect, manifest.at("expert_bodies").at("bytes").get<uint64_t>());
    CHECK_EQ(plan.slots_per_segment, static_cast<size_t>(36));
    CHECK_EQ(plan.buffer_set_bytes(), kSegBufBytes);
    CHECK_EQ(plan.buffer_set_bytes(), static_cast<size_t>(15099494400ULL));
}

TEST(segments_refill_ops_are_one_contiguous_run_a_segment_disjoint_across_segments) {
    const json manifest = full_manifest();
    const segplan::Plan plan = segplan::plan_segments(manifest, kNLayer, kHidden, kHcCount);
    for (int k = 0; k < 4; ++k) {
        const auto ops = segplan::refill_ops(plan, k);
        CHECK_EQ(ops.size(), static_cast<size_t>(36));
        uint64_t expect_offset = static_cast<uint64_t>(k) * kSegBufBytes;
        for (const auto& op : ops) {
            CHECK_EQ(op.offset, expect_offset);
            expect_offset += op.bytes;
        }
        CHECK_EQ(expect_offset, static_cast<uint64_t>(k + 1) * kSegBufBytes);
    }
}

// -------------------------------------------------------------------- (c)

TEST(segments_bind_list_is_a_bijection_and_reuses_slots_across_segments) {
    const json manifest = full_manifest();
    const segplan::Plan plan = segplan::plan_segments(manifest, kNLayer, kHidden, kHcCount);

    const auto b0 = segplan::bind_list(plan, 0);
    const auto b1 = segplan::bind_list(plan, 1);
    CHECK_EQ(b0.size(), static_cast<size_t>(36));
    CHECK_EQ(b1.size(), static_cast<size_t>(36));

    std::vector<bool> seen(36, false);
    for (const auto& [slot, name] : b0) {
        CHECK(slot < 36);
        CHECK(!seen[slot]);
        seen[slot] = true;
    }
    for (bool s : seen) CHECK(s);

    // slot 0 is (layer - lo) * 3 + kind_index == 0: layer0/.../experts_gate
    // in segment 0, layer12/.../experts_gate in segment 1 -- the SAME slot.
    CHECK_EQ(b0.front().first, static_cast<size_t>(0));
    CHECK_EQ(b0.front().second, std::string("layer0/moe/experts_gate/weight_u8"));
    CHECK_EQ(b1.front().first, static_cast<size_t>(0));
    CHECK_EQ(b1.front().second, std::string("layer12/moe/experts_gate/weight_u8"));

    // A short last segment binds a strict prefix of [0, slots_per_segment).
    json short_manifest = full_manifest(/*n_layer=*/44, /*seg_layers=*/12);
    const segplan::Plan sp = segplan::plan_segments(short_manifest, 44, kHidden, kHcCount);
    CHECK_EQ(sp.slots_per_segment, static_cast<size_t>(36));  // still 3*max(hi-lo) == 12*3
    const auto last = segplan::bind_list(sp, static_cast<int>(sp.segments.size()) - 1);
    CHECK_EQ(last.size(), static_cast<size_t>(24));  // 8-layer tail: 3*8
    for (const auto& [slot, name] : last) CHECK(slot < 24);
}

// -------------------------------------------------------------------- (d)

TEST(segments_refuse_shape_drift_of_a_kind_across_segments) {
    json manifest = full_manifest();
    // Corrupt one 'gate' body's shape in segment 1 (layer 12) so it
    // disagrees with segment 0's gate shape, while keeping bytes/product
    // internally consistent so no other check fires first.
    for (auto& e : manifest["expert_bodies"]["entries"]) {
        if (e["name"] == "layer12/moe/experts_gate/weight_u8") {
            e["shape"] = {512, 320, 40, 64};  // same product, different shape
        }
    }
    const std::string why = refusal(manifest);
    CHECK(why.find("shape drift") != std::string::npos);
    CHECK(why.find("layer12/moe/experts_gate/weight_u8") != std::string::npos);
}

// -------------------------------------------------------------------- (e)

TEST(segments_refuse_a_segment_without_a_full_attention_layer) {
    // A 2-layer cut over 4 layers puts layers {0,1} and {2,3} in separate
    // segments -- segment 0 holds no index %4==3 layer.
    json segments = json::array();
    segments.push_back(segment_row(0, 0, 2, 4));
    segments.push_back(segment_row(1, 2, 4, 4));
    json manifest = full_manifest(4, 2);
    // full_manifest already builds the right layer ranges for seg_layers=2;
    // reuse it directly.
    const std::string why = refusal(manifest, 4);
    CHECK(why.find("segment 0") != std::string::npos);
    CHECK(why.find("full-attention") != std::string::npos);
}

// -------------------------------------------------------------------- (f)

TEST(segments_refuse_a_gap_in_the_chain) {
    json manifest = full_manifest();
    manifest["segments"][2]["layers"] = {25, 36};  // was [24, 36): gap at 24
    for (auto& s : manifest["segments"]) {
        if (s["index"] == 2) s["inputs_embeds_width"] = kHcCount * kHidden;
    }
    // The bodies still claim layer 24 in segment 2; irrelevant, the chain
    // check runs before body checks.
    const std::string why = refusal(manifest);
    CHECK(why.find("gap") != std::string::npos);
}

TEST(segments_refuse_an_overlap_in_the_chain) {
    json manifest = full_manifest();
    manifest["segments"][2]["layers"] = {23, 36};  // was [24, 36): overlaps segment 1
    const std::string why = refusal(manifest);
    CHECK(why.find("overlap") != std::string::npos);
}

TEST(segments_refuse_an_out_of_order_chain) {
    // The chain is sorted by 'index' before tiling is checked, so a plain
    // reordering of array position never reaches the "out of order" branch
    // (sorting undoes it) -- what does is the index FIELD itself skipping a
    // value: {0, 1, 1, 3} has no segment claiming index 2, so after sorting
    // position 2 holds a segment whose own index says 1.
    json manifest = full_manifest();
    manifest["segments"][2]["index"] = 1;  // was 2; now duplicates segment 1's index
    const std::string why = refusal(manifest);
    CHECK(why.find("out of order") != std::string::npos);
}

TEST(segments_refuse_a_chain_short_of_n_layer) {
    json manifest = full_manifest();
    manifest["segments"][3]["layers"] = {36, 44};  // was [36, 48): 4 short
    manifest["segments"][3]["last"] = false;
    // last must be true somewhere; make none true to isolate the "short of
    // n_layer" refusal rather than the "back().last" one -- but the chain
    // check runs first, so this is refused as short regardless.
    const std::string why = refusal(manifest);
    CHECK(why.find("short of n_layer") != std::string::npos);
}

// -------------------------------------------------------------------- (g)

TEST(segments_refuse_a_width_mismatch) {
    json manifest = full_manifest();
    manifest["segments"][1]["inputs_embeds_width"] = kHidden;  // should be hc*H
    const std::string why = refusal(manifest);
    CHECK(why.find("segment 1") != std::string::npos);
    CHECK(why.find("inputs_embeds_width") != std::string::npos);
}

TEST(segments_refuse_segment0_width_not_hidden_size) {
    json manifest = full_manifest();
    manifest["segments"][0]["inputs_embeds_width"] = kHcCount * kHidden;
    const std::string why = refusal(manifest);
    CHECK(why.find("segment 0") != std::string::npos);
}

// -------------------------------------------------------------------- (h)

TEST(segments_refuse_ple_declared_on_more_than_one_segment) {
    json manifest = full_manifest();
    manifest["segments"][2]["has_ple"] = true;
    const std::string why = refusal(manifest);
    CHECK(why.find("has_ple") != std::string::npos);
}

TEST(segments_refuse_ngram_table_port_outside_segment_0) {
    json manifest = full_manifest();
    manifest["segments"][1]["inputs"].push_back(
        json{"ngram_table.0", {47718400, 90}, "u8"});
    const std::string why = refusal(manifest);
    CHECK(why.find("ngram_table") != std::string::npos);
    CHECK(why.find("segment 1") != std::string::npos);
}

TEST(segments_refuse_wrong_outputs_on_a_non_last_segment) {
    json manifest = full_manifest();
    manifest["segments"][0]["outputs"] = json::array({{"logits", {1, -1, 151936}, "f32"}});
    const std::string why = refusal(manifest);
    CHECK(why.find("segment 0 outputs") != std::string::npos);
}

TEST(segments_refuse_wrong_outputs_on_the_last_segment) {
    json manifest = full_manifest();
    manifest["segments"][3]["outputs"] =
        json::array({{"hidden_out", {1, -1, kHcCount * kHidden}, "f32"}});
    const std::string why = refusal(manifest);
    CHECK(why.find("segment 3 outputs") != std::string::npos);
}

TEST(segments_refuse_a_body_count_that_disagrees_with_3_times_span) {
    json manifest = full_manifest();
    manifest["expert_bodies"]["entries"].erase(0);  // drop layer0's gate body
    // Fix the arithmetic offset field of nothing else; the count check
    // fires before contiguity is even re-derived, since it counts first.
    const std::string why = refusal(manifest);
    CHECK(why.find("expert bodies, expected 3 * (hi - lo)") != std::string::npos);
}

TEST(segments_refuse_a_missing_kind) {
    json manifest = full_manifest();
    // Replace layer0's down body with a second gate body of the same
    // shape/bytes so the COUNT still matches 3*(hi-lo) but 'down' is absent.
    for (auto& e : manifest["expert_bodies"]["entries"]) {
        if (e["name"] == "layer0/moe/experts_down/weight_u8") {
            e["name"] = "layer0/moe/experts_gate/weight_u8_dup";
            e["kind"] = "gate";
        }
    }
    const std::string why = refusal(manifest);
    CHECK(why.find("missing its 'down' expert body") != std::string::npos);
}

TEST(segments_refuse_a_body_layer_outside_its_segment_range) {
    json manifest = full_manifest();
    for (auto& e : manifest["expert_bodies"]["entries"]) {
        if (e["name"] == "layer0/moe/experts_gate/weight_u8") {
            e["layer"] = 12;  // belongs to segment 1, not segment 0
        }
    }
    const std::string why = refusal(manifest);
    CHECK(why.find("outside segment 0's range") != std::string::npos);
}

TEST(segments_refuse_bytes_disagreeing_with_shape_product) {
    json manifest = full_manifest();
    manifest["expert_bodies"]["entries"][0]["bytes"] = kBodyBytes + 1;
    const std::string why = refusal(manifest);
    CHECK(why.find("product(shape)") != std::string::npos);
}

// ----------------------------------------------------------------------- (i)

TEST(chain_arch_hash_is_order_sensitive_and_k1_differs_from_plain_sha256) {
    const std::string a = std::string(64, 'a');
    const std::string b = std::string(64, 'b');
    const std::string ab = segplan::chain_arch_hash({a, b});
    const std::string ba = segplan::chain_arch_hash({b, a});
    CHECK(ab != ba);

    const std::string k1 = segplan::chain_arch_hash({a});
    // The formula: hash_prefix(sha256_hex(the 64-char hex STRING)), never
    // the identity and never sha256 of the underlying file's bytes (which
    // this test never has -- the point is the string is hashed, not passed
    // through).
    CHECK(k1 != a);
    const std::string expect = hash_prefix(sha256_hex(a));
    CHECK_EQ(k1, expect);
}

// ----------------------------------------------------------------------- (j)

namespace {

class TempSegmentedArtifactDir {
public:
    TempSegmentedArtifactDir() {
        const char* tmpdir = std::getenv("TMPDIR");
        std::string tmpl_s =
            std::string(tmpdir && *tmpdir ? tmpdir : "/tmp") + "/arcint-test-segments-XXXXXX";
        std::vector<char> tmpl(tmpl_s.begin(), tmpl_s.end());
        tmpl.push_back('\0');
        const char* base = ::mkdtemp(tmpl.data());
        if (base == nullptr) throw std::runtime_error("mkdtemp failed");
        parent_ = base;
        dir_    = parent_ + "/qwen36-coder-b5-ov";
        if (::mkdir(dir_.c_str(), 0700) != 0) throw std::runtime_error("mkdir failed");
        if (::mkdir((dir_ + "/segment0").c_str(), 0700) != 0)
            throw std::runtime_error("mkdir segment0 failed");
        if (::mkdir((dir_ + "/segment1").c_str(), 0700) != 0)
            throw std::runtime_error("mkdir segment1 failed");

        write("segment0/openvino_language_model.xml", "<xml0/>");
        write("segment0/openvino_language_model.bin", "weights0");
        write("segment1/openvino_language_model.xml", "<xml1/>");
        write("segment1/openvino_language_model.bin", "weightsAB");  // 9 bytes
        write("openvino_text_embeddings_model.xml", "<xml/>");
        write("openvino_tokenizer.xml", "<xml/>");
        write("openvino_detokenizer.xml", "<xml/>");
        write("chat_template.jinja", "{{ messages }}");
        write("tokenizer.json", "{}");
        write("expert_bodies.u8", "12345678");

        config_ = R"({"num_hidden_layers":8,"hidden_size":2560,)"
                  R"("layer_types":["linear_attention","linear_attention","linear_attention",)"
                  R"("qwen_sparse_attention","linear_attention","linear_attention",)"
                  R"("linear_attention","qwen_sparse_attention"]})";
        write("config.json", config_);

        json seg0 = json{{"index", 0}, {"dir", "segment0"}, {"layers", {0, 4}},
                         {"first", true}, {"last", false}, {"inputs_embeds_width", 2560},
                         {"has_ple", false}, {"attn_layers", 1},
                         {"lm_bin_bytes", 8},
                         {"inputs", json::array()}, {"outputs", json::array()}};
        json seg1 = json{{"index", 1}, {"dir", "segment1"}, {"layers", {4, 8}},
                         {"first", false}, {"last", true}, {"inputs_embeds_width", 10240},
                         {"has_ple", false}, {"attn_layers", 1},
                         {"lm_bin_bytes", 9},
                         {"inputs", json::array()}, {"outputs", json::array()}};
        const std::string xml0_sha = sha256_file(dir_ + "/segment0/openvino_language_model.xml");
        const std::string xml1_sha = sha256_file(dir_ + "/segment1/openvino_language_model.xml");
        json manifest = json{
            {"segment_layers", 4},
            {"segments", {seg0, seg1}},
            {"expert_bodies", nullptr},
            {"sha256", {{"segment0/openvino_language_model.xml", xml0_sha},
                       {"segment1/openvino_language_model.xml", xml1_sha}}},
        };
        write("serving-shape.json", manifest.dump());
    }

    ~TempSegmentedArtifactDir() {
        for (const char* name :
             {"segment0/openvino_language_model.xml", "segment0/openvino_language_model.bin",
              "segment1/openvino_language_model.xml", "segment1/openvino_language_model.bin",
              "openvino_text_embeddings_model.xml", "openvino_tokenizer.xml",
              "openvino_detokenizer.xml", "chat_template.jinja", "tokenizer.json",
              "expert_bodies.u8", "config.json", "serving-shape.json"}) {
            ::unlink((dir_ + "/" + name).c_str());
        }
        ::rmdir((dir_ + "/segment0").c_str());
        ::rmdir((dir_ + "/segment1").c_str());
        ::rmdir(dir_.c_str());
        ::rmdir(parent_.c_str());
    }

    const std::string& dir() const { return dir_; }
    void remove_segment1_bin() { ::unlink((dir_ + "/segment1/openvino_language_model.bin").c_str()); }

    void write(const std::string& name, const std::string& content) {
        std::ofstream out(dir_ + "/" + name, std::ios::binary);
        out << content;
    }

private:
    std::string parent_;
    std::string dir_;
    std::string config_;
};

}  // namespace

TEST(artifact_loads_a_segmented_manifest_with_summed_weights_bytes) {
    TempSegmentedArtifactDir d;
    Artifact a;
    const auto err = load_artifact(d.dir(), a);
    CHECK(!err.has_value());
    CHECK_EQ(a.n_layer, 8);
    CHECK_EQ(a.segments.size(), static_cast<size_t>(2));
    CHECK(a.segmented());
    CHECK_EQ(a.weights_bytes, static_cast<uint64_t>(8 + 9));
}

TEST(artifact_refuses_a_segmented_manifest_missing_segment1_bin) {
    TempSegmentedArtifactDir d;
    d.remove_segment1_bin();
    Artifact a;
    const auto err = load_artifact(d.dir(), a);
    CHECK(err.has_value());
    if (err.has_value()) {
        CHECK(err->find("segment 1") != std::string::npos);
    }
}

// ----------------------------------------------------------------------- (k)

TEST(buffer_set_state_ok_on_matching_segment) {
    segplan::BufferSetState st;
    st.refill(0);
    st.check_before_infer(0);  // must not throw
}

TEST(buffer_set_state_throws_naming_both_indices_on_mismatch) {
    segplan::BufferSetState st;
    st.refill(0);
    std::string why;
    try {
        st.check_before_infer(1);
    } catch (const std::runtime_error& e) {
        why = e.what();
    }
    CHECK(why.find("0") != std::string::npos);
    CHECK(why.find("1") != std::string::npos);
}

TEST(buffer_set_state_ok_after_a_second_refill) {
    segplan::BufferSetState st;
    st.refill(0);
    st.refill(1);
    st.check_before_infer(1);  // must not throw
}
