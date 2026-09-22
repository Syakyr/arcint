# Docker toolbox for arcint — investigation

Question: can arcint be packed the way [kyuz0/intel-b70-ai-toolboxes](https://github.com/kyuz0/intel-b70-ai-toolboxes)
packs llama.cpp for the B70 — pre-built, pull-and-run containers for Intel
Arc — and what does that take?

Short answer: **yes, but the split is different.** kyuz0's split is
per-backend (`sycl` / `vulkan` / `openvino`); arcint's necessary split is
per-**layer of divergence**: a heavy, rarely-rebuilt base image carrying the
*patched* OpenVINO runtime, and a thin per-commit image carrying the engine.
The patch series in `contrib/packaging/marfrit-openvino/patches/` is the
whole reason the two cannot be one Dockerfile that rebuilds on every push.

Everything below carries an evidence class per the repo's convention:
`code` = read from source (kyuz0's repo fetched 2026-09-22; this repo's
`contrib/packaging/`, `CMakeLists.txt`, `DEVELOPMENT.md` read here),
`measured-here` = exercised on this host, `unverified` = designed but not
run. **No image was built in this investigation — this host has no Docker
daemon (`measured-here`).** The draft Dockerfiles in this directory are
`unverified` and were written by transposing the verified `.deb` recipes.

---

## 1. What kyuz0 actually does (`code`)

Read from `toolboxes/Dockerfile.{sycl,vulkan,openvino}` and
`.github/workflows/`:

- **Two-stage Fedora builds.** `fedora:43` builder compiles llama.cpp;
  a second `fedora:43` stage installs only the runtime set and copies
  `/usr/local/` from the builder. No one-shot mega-layer.
- **The GPU driver stack lives inside the image.** Runtime stages install
  `intel-compute-runtime`, `intel-level-zero`, `oneapi-level-zero`,
  `ocl-icd` from Fedora repos (the SYCL image adds the oneAPI yum repo for
  the DPC++ compiler and its redistributables). The host supplies only
  `/dev/dri` and kernel driver — no oneAPI install on the host.
- **Device access is a documented run-flag recipe**, not baked in:
  `--device /dev/dri --group-add video --group-add render
  --security-opt seccomp=unconfined` (plus `--shm-size 200g` for vLLM).
  The same flags work under `toolbox`/`distrobox`, which is the headline
  convenience: `toolbox create --image docker.io/kyuz0/... -- <docker flags>`.
- **Models are never baked in.** Users mount/download at run time.
- **Upstream is cloned at build time** (`git clone llama.cpp master`), and
  a **poll workflow** (`poll-llama-cpp.yml`) checks upstream HEAD twice a
  day against a stored SHA artifact and re-triggers the build on change.
- **Publishing discipline** (`build_and_publish.yml`): matrix over backends,
  an **immutable timestamped tag** plus a **moving channel tag**
  (`:sycl_20250101T120000` and `:sycl`), smoke tests between the two
  (`llama version`, `--help` exit-status gates), and a prune workflow for
  old images.
- **Helpers ride along**: `gguf-vram-estimator.py` on `PATH`, banner
  scripts, a TUI launcher for the vLLM image.

## 2. What arcint actually needs (`code`)

From `DEVELOPMENT.md`, `README.md`, `CMakeLists.txt` and
`contrib/packaging/`:

- **A patched OpenVINO, not a release.** The measurement stack is nightly
  `2026.4.0-22849-71640275d29` (commit `71640275`) plus the numbered
  patch series (`0003`–`0032` in `patches/`; deployed level `+p15`,
  build script currently stamps `marfrit-p16` — see §5, open question).
  kyuz0's `Dockerfile.openvino` downloads the **official 2026.2 tgz** —
  that path is unusable here: no MoE slot-pool fixes, no paged-KV
  precision patches, no kquant kernels. `arcint` reads the `+pN` marker
  off the plugin version string at load and needs `+p7` minimum for
  `--gguf`; the dense/MoE production configurations need `+p15`.
- **The OpenVINO build is expensive and constrained**: ~35 min on 8 cores,
  x86_64 glibc only (`build-openvino.sh` header). Cloning upstream with
  submodules is a few GB.
- **The wheel layout is load-bearing.** `OpenVINOTargets-release.cmake`
  resolves every library as `${_IMPORT_PREFIX}/openvino/libs/<soname>`.
  The `.deb` recipe's trick: unpack the upstream nightly wheels for the
  layout, headers, CMake package and `libopenvino_tokenizers.so`, then
  **overwrite** the OpenVINO libraries with the patched build's
  `bin/intel64/Release/*`, strip, and assert the version string carries
  the patch marker. The image reproduces this exactly at
  `/usr/lib/marfrit-openvino` so the *same* CMake flags from
  `contrib/packaging/arcint/build-deb.sh` work unchanged.
- **The tokenizers library comes from the upstream nightly wheel**
  (unpatched, `manylinux_2_28`, ABI-matched to the same commit). Nightly
  indexes **prune old builds** — the deb recipe carries a `MIRROR_DIR`
  escape hatch for exactly that. A published base image *is* the mirror:
  this is arguably the strongest argument for the container route.
- **arcint itself is cheap to build**: C++20, CMake ≥ 3.20, vendored
  `third_party/`, no network at build time. Minutes, not 35.
- **Runtime GPU dependency**: Level Zero / Intel compute-runtime (same set
  kyuz0 installs), plus `ocl-icd` (the deb's `Depends:` line:
  `ocl-icd-libopencl1 | libopencl1`, `Recommends: intel-opencl-icd`).
- **Models are OpenVINO IR directories** (`openvino_language_model.{xml,bin}`,
  tokenizer/detokenizer, `config.json`, chat template) plus optional GGUF
  files on top. Tens of GB — never bake in; mount read-only.
- **Security posture**: no auth, permissive CORS, no TLS — LAN/reverse-proxy
  only (`DEVELOPMENT.md`). The image must not encourage a public bind.

## 3. Gap analysis — why the shapes differ

| kyuz0 | arcint | consequence |
|---|---|---|
| Backend varies per image | One backend (patched OV) | Split by rebuild cost, not backend |
| Official OV tgz, download-only | OV built from source + 30 patches | Base image is heavy; cache/publish it |
| Rebuild on every upstream commit | OV pin changes rarely (a *commit*, and "a patch that doesn't apply is a bug in patches/, not a reason to move the pin") | Tier-1 rebuilds only on `patches/**` or pin change |
| llama.cpp is the product | arcint is the product; OV is its libc | Tier-2 is thin: `cmake && ctest -L unit && install` |
| `llama version` smoke test | Device-free stub exists: `arcint --stub --port 8090` | Better smoke test available without a GPU |
| Users: tinkerers with toolbox | Same audience fits, but serving posture | Keep the toolbox recipe; add compose example |

## 4. Proposed design (drafts in this directory, `unverified`)

**Two images, one repo, three artifacts:**

1. **`arcint-ov:<pin>-<patchlevel>`** — `Dockerfile.openvino-patched`
   (tier 1, ~2 GB runtime). Stages:
   - `ov-src`: shallow-fetch upstream OpenVINO at `71640275`, apply
     `contrib/packaging/marfrit-openvino/patches/*.patch` with the
     same `git apply --check` gate, build Release with the same CMake
     flags as `build-openvino.sh` (including
     `-DCI_BUILD_NUMBER=2026.4.0-22849-71640275d29-<patchlevel>` so the
     version string names the patch level out loud).
   - `wheels`: fetch the two pinned nightly wheels, `sha256sum -c`,
     unpack with `python3 -m zipfile` (same as the deb recipe).
   - `runtime` (`fedora:43`): install
     `intel-compute-runtime intel-level-zero oneapi-level-zero ocl-icd`
     (+ `ca-certificates curl hwdata pciutils`), copy the wheel layout to
     `/usr/lib/marfrit-openvino`, overwrite the libs with the patched
     build, strip, run the deb recipe's existence/marker asserts as build
     gates. Deliberately **no `/etc/ld.so.conf.d` entry** — arcint finds
     these via RPATH, mirroring the deb's reasoning.
   Rebuild trigger: `contrib/packaging/marfrit-openvino/patches/**` or
   the pin changes.

2. **`arcint:<ver>` / `arcint:latest`** — `Dockerfile.arcint` (tier 2,
   minutes). `FROM` the tier-1 image for both builder and final stages
   (it carries the CMake package, headers and tokenizers `.so`):
   `cmake -DARCINT_OPENVINO=ON -DOpenVINO_DIR=/usr/lib/marfrit-openvino/openvino/cmake
   -DARCINT_TOKENIZERS_SO=/usr/lib/marfrit-openvino/openvino_tokenizers/lib/libopenvino_tokenizers.so
   -DCMAKE_INSTALL_RPATH=/usr/lib/marfrit-openvino/openvino/libs` —
   byte-for-byte the deb's flags — then **`ctest -L unit
   --no-tests=error` as a build gate** (the device-free unit ladder runs
   on any runner, no card needed), install, copy into the final stage.
   `ENTRYPOINT ["/usr/bin/arcint"]`, args passed through.

3. **CI** — `workflow-build-toolbox.yml.example`: build tier 1 on path
   filter, tier 2 on push; immutable timestamp tag + channel tag; the
   device-free smoke test (`arcint --stub` + `curl /props`) between build
   and push. GitHub-hosted runners have no Intel GPU, so the
   card-requiring acceptance cells stay out of CI by design (same ladder
   rule as `DEVELOPMENT.md`); a self-hosted Arc runner could later run
   `tests/acceptance/run.py` against the built image.

**Run recipe** (mirrors kyuz0's, in `docker-compose.example.yaml`):

```
docker run --rm --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -v /path/to/models:/models:ro -p 8080:8080 \
  ghcr.io/<owner>/arcint:latest \
  --model /models/ov/qwen36-coder-b5-ov --model-id qwen3.6-27b-a3b-coder \
  --served-model-name qwen3.6-coder --device GPU.0 \
  --host 0.0.0.0 --port 8080 --n-ctx 262144
```

The `contrib/packaging/arcint.service` systemd template needs no change:
RPATH resolves the runtime inside the image exactly as on the packaged
host, so the ExecStart shape carries over verbatim.

**What is NOT proposed:**

- Baking models in (tens of GB per model family; kyuz0 doesn't either).
- One mega-image per card — the card is a run-time flag (`--device GPU.0`),
  not an image axis.
- Rebuilding tier 1 per push — 35+ minutes of CI per commit for a
  runtime that changes only with the patch series.
- Replacing the `.deb` route — the image and the deb consume the same
  recipes; neither forks the other.

## 5. Open questions for the operator

1. **Patch-level drift (`code`, unresolved).** `build-openvino.sh` stamps
   `PATCHLEVEL=marfrit-p16`; `build-deb.sh` ships `PKGVER=…+p15` and
   `README.md` says the deployed level/floor is `+p15` with "patches
   0003–0033", while `patches/` holds `0003`–`0032`. The image drafts
   take `marfrit-p16` as the ARG default because that is what the build
   script says, but the three sources disagree and per this repo's own
   discipline that discrepancy should be resolved before anything is
   published. Which is the deployed level?
2. **Registry**: GHCR (free for public repos, no username juggling) vs
   Docker Hub (matches kyuz0's discoverability, `toolbox` docs use
   `docker.io/`). Drafts assume GHCR; trivially changed.
3. **Base distro**: Fedora 43 chosen for parity with kyuz0's runtime
   driver packages (`intel-compute-runtime` etc. are current there) and
   because the from-source OV build is distro-agnostic. The deb recipe
   insists on Debian trixie *for the .deb* (ABI skew of cross-distro
   installs); a self-contained image doesn't inherit that constraint, but
   if you want the image to be the same glibc as the fleet hosts, a
   `debian:trixie` base is the conservative swap (driver packages then
   come from Intel's OCL/Level Zero apt repos instead of Fedora's).
4. **Nightly wheel availability**: the tokenizers wheel is fetched from
   `storage.openvinotoolkit.org/wheels/nightly/` by exact filename; if
   it has been pruned since the recipes were written, the tier-1 build
   fails at the `wheels` stage. First CI run answers this. If pruned:
   pin a mirror (same insurance `MIRROR_DIR` is).
5. **Image for the B60 vs A770 configurations**: same image, different
   run flags and different model dirs — confirm no per-card build axis is
   wanted (nothing in the build suggests one).
6. **Version-string gate**: tier 1 asserts the built `libopenvino`
   carries the patch marker (copied from the deb recipe). If you want
   tier 2 to *also* assert the base is ≥ the engine's floor at build time
   (rather than at arcint load time), that's a two-line addition —
   included as a comment in the draft.

## 6. Files in this branch

- `docker/INVESTIGATION.md` — this document.
- `docker/Dockerfile.openvino-patched` — tier 1 draft (patched OV base).
- `docker/Dockerfile.arcint` — tier 2 draft (engine image).
- `docker/docker-compose.example.yaml` — run recipe with GPU passthrough.
- `docker/workflow-build-toolbox.yml.example` — CI draft (kept out of
  `.github/workflows/` so it does not fire from this investigation
  branch).

## 7. Verification status

| Claim | Class |
|---|---|
| kyuz0 image/workflow structure | `code` (fetched 2026-09-22) |
| arcint build/deb/OV-recipe facts | `code` (read here) |
| No Docker daemon on this host | `measured-here` |
| Draft Dockerfiles build at all | **unverified** — first CI run is the test |
| GPU passthrough recipe works for arcint on Arc | `code`-transposed from kyuz0; unverified for this engine |
| Stub smoke test is device-free | `code` (README/DEVELOPMENT.md; unit ladder design) |
