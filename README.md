# arcint

A deliberately narrow LLM inference engine for Intel Arc GPUs.

arcint runs a short allowlist of hybrid Qwen models on two Intel cards — the
Arc A770 (16 GiB) and the Arc Pro B60 (24 GB) — and tries to do that better
than the general-purpose engines do. The inspiration is
[NInfer](https://github.com/Neroued/ninfer), a from-scratch engine that
supports two checkpoints on one GPU and beats every generalist on that pair.
arcint translates the idea to Intel: kernel work is delegated to OpenVINO's
compiler stack, which already emits good Xe code, and arcint owns everything
around the compute graph — the serving loop, the scheduler, the KV and
recurrent-state memory, prefix caching, and speculative decoding. Where a
kernel has to change, the change is a small, published patch series against a
pinned OpenVINO build.

Every served model is a hybrid: most layers use linear attention
(GatedDeltaNet), one in four uses full attention. Most of the design follows
from that; see [DESIGN.md](DESIGN.md). [llm.txt](llm.txt) is the
machine-readable summary, [CHANGELOG.md](CHANGELOG.md) the per-release record
and the runtime each release depends on.

## Building

C++20, CMake, no network at build time. `third_party/` holds the two vendored
single headers (cpp-httplib, nlohmann/json) and
[models/allowlist-raw.json](models/allowlist-raw.json) holds the IR metadata
the allowlist is pinned against.

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    ctest --test-dir build --output-on-failure -L unit

    ./build/arcint --stub --port 8090 -v

This builds the **stub**, which serves the HTTP surface without a model and is
what the device-free tests run against. In this configuration `--model` is
refused at startup instead of starting something that cannot run. `-L unit`
selects the device-free gate (`arcint-test`, the HTTP round trip, the
stub-only concurrency stress test, and the acceptance enumeration's own
consistency checks) — the only thing bare `ctest` ever runs. The
card-requiring acceptance cells are a separate, enumerated target
(`tests/acceptance/`, checklist in
[docs/release-checklist.md](docs/release-checklist.md)); see
[DEVELOPMENT.md](DEVELOPMENT.md) for how to build and run them.

A build that serves a model needs OpenVINO:

    cmake -S . -B build-ov -DARCINT_OPENVINO=ON -DCMAKE_BUILD_TYPE=Release \
          -DARCINT_GIT_SHA=$(git rev-parse --short=12 HEAD)
    cmake --build build-ov -j"$(nproc)"
    cmake --install build-ov --prefix ~/.local

`-DARCINT_WERROR=ON` gives the warning-clean build CI should use. Pass
`-DARCINT_GIT_SHA` whenever the build tree has no `.git`; without it
`--version` and `/props` report `unknown`.

**The runtime.** The numbers below depend on a *patched* OpenVINO:
`marfrit-openvino +p20`, the 2026.4.0 nightly at upstream commit `71640275`
with the patch series 0003–0067 applied. `contrib/packaging/` holds the Debian
recipes that build it and arcint itself, and
`contrib/packaging/marfrit-openvino/patches/README.md` says what each patch
does and what it measured. No `.deb` is published anywhere; the directory
contains everything needed to build the same thing, and it is the shortest
path to reproducing a number.

## Measured

The acceptance task is a Lua CSV parser to RFC 4180: ten named cases (CRLF
and bare LF, a missing final terminator, quoted fields, an embedded comma, a
doubled quote, an embedded newline, empty and empty-quoted fields, no
trimming). The candidate code is **executed**, not read; one point per case.
"task" below is the score out of 10. Every row was measured on this project's
own hardware; the card, the configuration and the runtime are named, and the
dated measurement is in DESIGN.md §7.

### The two served configurations

| endpoint | card | model | configuration | prefill | decode | task |
|---|---|---|---|---|---|---|
| coder | A770, 16 GiB | Qwen3.6-27B-A3B-Coder, int4 (MoE, 184 experts) | `u8` KV, prefix cache | ~510 t/s at ~1k tokens; 621 t/s at 98k | 47.6 t/s at ~1k; 40.3 t/s at 71.7k | 10/10 |
| agent | B60, 24 GB | Qwen3.8-27B dense, int4 AWQ, MTP head | `u8:i4` KV, MTP on, 151,552 tokens | 1,436 t/s at 850 tokens | 23.5 t/s | 10/10 (at `u8` KV) |

The coder on the 24 GB card decodes at 66.5 t/s and prefills at 2,821 t/s at
~1k tokens — above the 60 t/s OpenVINO GenAI baseline on the same card and
artifact that justifies the project.

### Qwen3.6-35B-A3B on the 16 GiB card, all-resident

The 35B's int4 IR does not fit the A770 resident. Exported from the
checkpoint's own GGUF with its expert blocks kept in their native formats
(IQ2_S gate/up, IQ3_XXS down) and computed per routed expert on the card
(`--offload-ratio 0 --moe-per-expert-dispatch`), it does, at 13.11 GiB:

| A770, all-resident, `u8` KV | value |
|---|---|
| prefill, 4,096 tokens | ~960 t/s |
| prefill, 32,768 tokens | 779 t/s |
| decode after 4,096 tokens | 28.1 t/s |
| max context per lane | 112,288 tokens |
| task | 10/10 |

### Speculative decoding on the dense Qwen3.8-27B (B60)

| drafter | short prompt | at 77k tokens |
|---|---|---|
| none | 24.0 t/s | 15.3 t/s |
| MTP head (`--mtp on`) | 33.0 t/s, 76.7% accepted | 4.9 t/s |
| DFlash2 int4 (`--dflash`) | **44.8 t/s**, 3.13 tokens per verify cycle | **18.8 t/s** |

The DFlash2 drafter is the public block-diffusion head
[`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
exported with `tools/export_dflash.py`. Drafters engage only under greedy, and
a drafted token is accepted only when it equals what the sampler would have
picked — but a multi-token verify pass and a single-token step are not
bit-identical on this backend, so the answer can differ from plain greedy at a
near-tie. Serve without a drafter where bit-exact reproducibility matters.

### A GGUF through arcint (B60)

The dense Qwen3.8-27B as Unsloth's Q4_K_M, opened with `--gguf` on the dense
IR as the topology template, `u8` KV, one lane, MTP off, against Intel's own
int4 IR of the same model:

| weights | resident | prefill 856 / 71.7k tokens | decode 856 / 71.7k tokens | task |
|---|---|---|---|---|
| **GGUF Q4_K_M, mixed form (the default)** | 16.54 GiB | **1,008 / 464 t/s** | **18.5 / 13.7 t/s** | 10/10 |
| GGUF Q4_K_M, native rows (`--gguf-mode native`) | 15.22 GiB | 662 / 395 t/s | 19.5 / 14.2 t/s | 10/10 |
| Intel int4 IR | 13.06 GiB | 1,609 / 552 t/s | 23.1 / 16.5 t/s | — |
| llama.cpp SYCL, the same file (1k / 10k) | — | 249 / 206 t/s | 14.2 / 12.4 t/s | — |
| llama.cpp Vulkan, the same file (1k / 10k) | — | 126 / 108 t/s | 7.8 / 7.0 t/s | — |

The mixed form repacks the file's Q4_K rows at load into the runtime's own
compressed form — every weight within a measured 1/64 of a quantisation step
of ggml's value, the mins as exact extra columns — and runs Q5_K and Q6_K as
the file's rows through a K-quant kernel carried in the patch series.

## Supported models and formats

**arcint loads an OpenVINO IR directory** — `openvino_language_model.{xml,bin}`,
`openvino_text_embeddings_model.{xml,bin}`, the tokenizer and detokenizer IRs,
`config.json` and the chat template — and only one whose hashes and byte count
match an allowlist entry. GPTQ or NVFP4 safetensors will not load.

| family | models | what serves |
|---|---|---|
| `qwen3_5_moe` | Qwen3.6-27B-A3B-Coder, Qwen3.6-35B-A3B | int4 IRs (the 35B via expert offload on the 16 GiB card); the 35B's native-format artifact all-resident |
| `qwen3_5` dense | Qwen3.8-27B | our AWQ export and Intel's int4 IR (MTP head reconstructed with `tools/export_mtp.py`); its GGUF through `--gguf` |
| `qwen4_exp` | Qwen3.8 Flash-Next (512 experts) | serving-shape artifacts built from the GGUF with `tools/export_serving_artifact.py`; full depth serves on one card with part of the experts on the CPU tier, slowly |

**A GGUF opens on top of an IR directory**: `--gguf FILE --model DIR` takes
the served IR of the same architecture as the topology template and replaces
its projections with the file's own K-quant rows (Q4_K, Q5_K, Q6_K, Q8_0).
`--gguf-mode repack|native|mixed` chooses the form (mixed, the default);
`--gguf-embed file|template` where the token embeddings come from;
`--gguf-check once|always` whether the repack's per-projection deviation check
is cached between loads. Dense models of the allowlisted families.

Where to get the artifacts, how the served ones were exported and calibrated,
the features, the non-goals and deployment notes are in
[FURTHER-READING.md](FURTHER-READING.md).

**This branch carries the current state.** The dated measurement record, the
campaign documents, design notes and acceptance windows behind every number
here live on the development branch, `qfndev`.
