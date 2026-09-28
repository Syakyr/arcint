#include "core/artifact.h"
#include "harness.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace lgc;

namespace {

// A throwaway artifact directory under an allowlisted alias
// (models/allowlist-raw.json's "qwen36-coder-b5-ov"), holding just enough
// for load_artifact to succeed: the language model, text embeddings,
// tokenizer/detokenizer, config.json and chat_template.jinja. Content is
// throwaway bytes -- load_artifact only checks that these files exist and
// hashes them, it does not parse the IR graph.
class TempArtifactDir {
public:
    // The alias defaults to the coder artifact's real directory name; pass a
    // different one to stand in for a mislabelled directory.
    explicit TempArtifactDir(std::string alias = "qwen36-coder-b5-ov") {
        const char* tmpdir = std::getenv("TMPDIR");
        std::string tmpl_s = std::string(tmpdir && *tmpdir ? tmpdir : "/tmp") + "/arcint-test-artifact-XXXXXX";
        std::vector<char> tmpl(tmpl_s.begin(), tmpl_s.end());
        tmpl.push_back('\0');
        const char* base = ::mkdtemp(tmpl.data());
        if (base == nullptr) throw std::runtime_error("mkdtemp failed");
        parent_ = base;
        dir_    = parent_ + "/" + alias;
        if (::mkdir(dir_.c_str(), 0700) != 0) throw std::runtime_error("mkdir failed");

        write("openvino_language_model.xml", "<xml/>");
        write("openvino_language_model.bin", "weights");
        write("openvino_text_embeddings_model.xml", "<xml/>");
        write("openvino_tokenizer.xml", "<xml/>");
        write("openvino_detokenizer.xml", "<xml/>");
        write("config.json", "{}");
        write("chat_template.jinja", "{{ messages }}");
        write("tokenizer.json", "{}");
    }

    ~TempArtifactDir() {
        for (const char* name :
             {"openvino_language_model.xml", "openvino_language_model.bin",
              "openvino_text_embeddings_model.xml", "openvino_tokenizer.xml",
              "openvino_detokenizer.xml", "config.json", "chat_template.jinja",
              "tokenizer.json", "openvino_vision_embeddings_model.xml",
              "openvino_vision_embeddings_model.bin", "openvino_vision_embeddings_pos_model.xml",
              "openvino_vision_embeddings_pos_model.bin", "openvino_vision_embeddings_merger_model.xml",
              "openvino_vision_embeddings_merger_model.bin"}) {
            ::unlink((dir_ + "/" + name).c_str());
        }
        ::rmdir(dir_.c_str());
        ::rmdir(parent_.c_str());
    }

    const std::string& dir() const { return dir_; }

    void write(const std::string& name, const std::string& content) {
        std::ofstream out(dir_ + "/" + name, std::ios::binary);
        out << content;
    }

    // Writes exactly `bytes` bytes -- the vision-file-size case needs the
    // reported size to match precisely, not just be nonzero.
    void write_sized(const std::string& name, uint64_t bytes) {
        std::ofstream out(dir_ + "/" + name, std::ios::binary);
        const std::string content(bytes, 'v');
        out << content;
    }

private:
    std::string parent_;
    std::string dir_;
};

}  // namespace

// FULL-DEPTH (2026-09-13): the Flash-Next config names its full-attention
// layers "qwen_sparse_attention" (the pin's `layer_types`; the selection
// branch is the indexer, ruled out, and the layer is served dense-causal --
// window-050 §8, 0.0 price to T=2051). The loader counted only the literal
// "full_attention", so a real serving-shape artifact reported 0 attention
// layers and 4 GDN out of 4, and the registry's layer split could never sum.
// Red first: this cell read n_attn_layer 0 before the loader learnt the name.
TEST(artifact_counts_qwen_sparse_attention_as_full_attention) {
    TempArtifactDir d;
    d.write("config.json",
            R"({"num_hidden_layers":4,"layer_types":["linear_attention","linear_attention",)"
            R"("linear_attention","qwen_sparse_attention"]})");
    Artifact   a;
    const auto err = load_artifact(d.dir(), a);
    CHECK(!err.has_value());
    CHECK_EQ(a.n_layer, 4);
    CHECK_EQ(a.n_attn_layer, 1);
    CHECK_EQ(a.n_gdn_layer, 3);
}

// M13 (docs/milestone-0.3.0.md): a *ForConditionalGeneration export carries
// vision-tower/projector IRs the loader never reads (src/core/artifact.cpp
// resolves only the language model and text embeddings). A plain text-only
// export has none of them, and the inventory must say exactly that -- no
// entries, not "we didn't look".
TEST(artifact_reports_no_unloaded_vision_irs_without_them) {
    TempArtifactDir d;
    Artifact         a;
    const auto       err = load_artifact(d.dir(), a);
    CHECK(!err.has_value());
    CHECK(a.unloaded_vision_irs.empty());
}

// The red case this milestone cares about: a vision IR present on disk (here,
// just the merger's .bin -- the loader never asked for its .xml either) is
// inventoried with its exact name and byte size, so backend_ov.cpp can log
// "N files, X MiB on disk" truthfully.
TEST(artifact_reports_a_present_vision_ir_with_its_exact_size) {
    TempArtifactDir d;
    d.write_sized("openvino_vision_embeddings_merger_model.bin", 12345);
    Artifact   a;
    const auto err = load_artifact(d.dir(), a);
    CHECK(!err.has_value());
    CHECK_EQ(a.unloaded_vision_irs.size(), static_cast<size_t>(1));
    if (a.unloaded_vision_irs.size() == 1) {
        CHECK_EQ(a.unloaded_vision_irs[0].name,
                 std::string("openvino_vision_embeddings_merger_model.bin"));
        CHECK_EQ(a.unloaded_vision_irs[0].bytes, static_cast<uint64_t>(12345));
    }
}

// All three named IRs (embeddings, pos, merger), both xml and bin, are
// checked independently -- a checkpoint that ships every half of every
// vision file must report all six, not just the first one found.
TEST(artifact_reports_every_present_vision_ir_file) {
    TempArtifactDir d;
    d.write("openvino_vision_embeddings_model.xml", "<xml/>");
    d.write_sized("openvino_vision_embeddings_model.bin", 100);
    d.write("openvino_vision_embeddings_pos_model.xml", "<xml/>");
    d.write_sized("openvino_vision_embeddings_pos_model.bin", 200);
    d.write("openvino_vision_embeddings_merger_model.xml", "<xml/>");
    d.write_sized("openvino_vision_embeddings_merger_model.bin", 300);

    Artifact   a;
    const auto err = load_artifact(d.dir(), a);
    CHECK(!err.has_value());
    CHECK_EQ(a.unloaded_vision_irs.size(), static_cast<size_t>(6));

    uint64_t total = 0;
    for (const auto& ir : a.unloaded_vision_irs) total += ir.bytes;
    // Three ".xml" files at 6 bytes ("<xml/>") each, plus the three sized
    // ".bin" files.
    CHECK_EQ(total, static_cast<uint64_t>(3 * 6 + 100 + 200 + 300));
}

// ---------------------------------------------------------------------------
// The directory name SELECTS the allowlist entry and the hashes PROVE it
// (DESIGN.md §3.1). When the name selects nothing, the refusal has to say what
// the bytes are: the entry id and the directory alias are disjoint namespaces,
// and a bare "not an allowlisted artifact directory" cannot tell a mislabelled
// artifact from one that has never been pinned.
//
// The fixture's bytes are throwaway, so end to end it lands on the "new
// artifact" branch; the branch that NAMES an entry is covered against the real
// pinned hashes below.

TEST(artifact_probe_arch_hash_matches_what_the_loader_computes) {
    TempArtifactDir d;
    const std::string probed = probe_arch_hash(d.dir());
    CHECK(!probed.empty());
    CHECK_EQ(probed.size(), static_cast<size_t>(16));  // the pinned prefix form

    Artifact a;
    CHECK(!load_artifact(d.dir(), a, /*require_allowlisted=*/false).has_value());
    CHECK_EQ(probed, a.arch_hash);
}

TEST(artifact_probe_arch_hash_is_empty_without_anything_to_hash) {
    char        tmpl[] = "/tmp/arcint-probe-empty-XXXXXX";
    const char* base   = ::mkdtemp(tmpl);
    CHECK(base != nullptr);
    if (base != nullptr) {
        CHECK(probe_arch_hash(base).empty());
        ::rmdir(base);
    }
}

TEST(artifact_unknown_directory_error_names_the_entry_behind_a_pinned_hash) {
    // The coder's pinned lm-xml sha (models/allowlist-raw.json, and the
    // registry entry transcribed from it).
    const std::string msg = unknown_directory_error("qwen36-27b-a3b-coder", "6745cfe3d57e3f0f");
    CHECK(msg.find("not an allowlisted artifact directory") != std::string::npos);
    CHECK(msg.find("qwen3.6-27b-a3b-coder") != std::string::npos);
    CHECK(msg.find("qwen36-coder-b5-ov") != std::string::npos);
    CHECK(msg.find("bytes are allowlisted and the name is not") != std::string::npos);
}

TEST(artifact_unknown_directory_error_names_every_entry_on_a_shared_hash) {
    // The 35B and its MTP variant share one language model (21fe4d57d6d016f5)
    // and differ only in the head files beside it. Naming one would be a guess.
    const std::string msg = unknown_directory_error("qwen36-35b-a3b-copy", "21fe4d57d6d016f5");
    CHECK(msg.find("qwen3.6-35b-a3b (") != std::string::npos);
    CHECK(msg.find("qwen3.6-35b-a3b-mtp (") != std::string::npos);
    CHECK(msg.find("qwen36-35b-a3b-int4-ov") != std::string::npos);
    CHECK(msg.find("qwen36-35b-a3b-mtp-ov") != std::string::npos);
}

TEST(artifact_unknown_directory_error_calls_a_new_artifact_new) {
    const std::string msg = unknown_directory_error("brand-new-export", "0123456789abcdef");
    CHECK(msg.find("matches no allowlisted entry either") != std::string::npos);
    CHECK(msg.find("new artifact, not a mislabelled one") != std::string::npos);
}

TEST(artifact_mislabelled_directory_refusal_carries_its_hash) {
    TempArtifactDir wrong_named("qwen36-27b-a3b-coder");  // real layout, wrong name
    const std::string hash = probe_arch_hash(wrong_named.dir());
    CHECK(!hash.empty());

    Artifact        a;
    const auto      err = load_artifact(wrong_named.dir(), a);
    CHECK(err.has_value());
    if (err.has_value()) {
        CHECK(err->find("not an allowlisted artifact directory") != std::string::npos);
        CHECK(err->find(hash) != std::string::npos);
        // Throwaway fixture bytes: the honest answer is "nothing here is
        // pinned", not a confident name.
        CHECK(err->find("new artifact, not a mislabelled one") != std::string::npos);
    }
}
