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
  build script had moved to `marfrit-p16` — resolved in §5.1: the image builds the deployed +p15 set).
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

1. **Patch-level drift — RESOLVED 2026-09-22 (`code`, measured from
   git).** The deployed `+p15` deb was built from patches **0003–0033**
   (31 patches; measured at commit `b0ccd47`, where `build-deb.sh` was
   stamped `+p15`) — README.md was right; an earlier note here claiming
   `patches/` held only `0003–0032` was a `head -60` truncation
   artifact. `0034`–`0036` were added later (0.4.5+/flash-next work)
   and were never stamped into a deb. **`0037` — the `+p16` patch per
   its own commit `ae620e3` — is TRUNCATED in this repo AND in upstream
   `marfrit/arcint`** (identical bytes): its last hunk declares
   `new=82/old=6`, the file carries `79/4`, corrupt at line 436. The
   project's own `build-openvino.sh` fails on it identically — this is
   a bug in the patch directory, not in the image recipe, and it needs
   re-export by the patch author. The image therefore builds the
   deployed `+p15` set (`PATCH_MAX=0033`) and stamps `marfrit-p15`,
   matching the deb's semantics exactly; bumping later is one ARG.
2. **Registry**: ~~GHCR vs Docker Hub~~ **decided 2026-09-22: GHCR,
   `ghcr.io/syakyr/`** — the workflow is live (see §8). Docker Hub
   remains an option if discoverability like kyuz0's matters later.
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
- `.github/workflows/build-toolbox.yml` — **live** CI (activated
  2026-09-22; see §8). `workflow_dispatch` only: tiers `ov` / `arcint`
  / `all`, moving channel tag (default `investigation`).

## 7. Verification status

| Claim | Class |
|---|---|
| kyuz0 image/workflow structure | `code` (fetched 2026-09-22) |
| arcint build/deb/OV-recipe facts | `code` (read here) |
| No Docker daemon on this host | `measured-here` |
| Draft Dockerfiles build at all | **unverified** — first CI run is the test |
| GPU passthrough recipe works for arcint on Arc | `code`-transposed from kyuz0; unverified for this engine |
| Stub smoke test is device-free | `code` (README/DEVELOPMENT.md; unit ladder design) |
| Workflow YAML fires / tier gating | **unverified** — first dispatch is the test |

## 8. Status log (appended, never rewritten)

- **2026-09-22, investigation pushed.** Design + drafts, nothing built
  (no Docker daemon on the authoring host).
- **2026-09-22, workflow activated under the operator namespace.**
  `.github/workflows/build-toolbox.yml` is live on this branch. First
  run: dispatch with `tiers=all` (tier 2 FROMs the tier-1 image from the
  registry, so tier 1 must exist first). Images:
  - `ghcr.io/syakyr/arcint-ov:2026.4.0-marfrit-p16`
  - `ghcr.io/syakyr/arcint:investigation-<ts>` (immutable) and
    `ghcr.io/syakyr/arcint:investigation` (moving channel).
  `latest` is deliberately untouched while this is an investigation.
- **GHCR first-push gotcha:** packages are **private** by default. After
  the first successful push, flip both packages to public (package page →
  Package settings → Change visibility) so the B60 box can pull without
  credentials — or have it log in with a PAT carrying `read:packages`:
  `echo $PAT | docker login ghcr.io -u syakyr --password-stdin`.
- **Known CI risks for the first run** (`unverified` until dispatched):
  the nightly-wheel fetch (§5.4) and the tier-1 disk footprint — the
  OpenVINO clone+build is tens of GB and GitHub runners are tight even
  after the cleanup step; if tier 1 dies with no space left on device,
  the fix is a self-hosted runner or a pre-baked OV cache, not the
  Dockerfile.

- **2026-09-22, run 3 failed at patch 0037 — root-caused.**
  `corrupt patch at 0037-moe-hybrid-prefill-split.patch:436` (measured
  via CI log). The committed blob is truncated in this repo and upstream
  alike (hunk declares new=82/old=6; file has 79/4). Reconstruction of
  the missing tail was rejected — one added line is unrecoverable and a
  guessed patch is worse than a loud failure. Fix taken: `PATCH_MAX=0033`
  in the tier-1 Dockerfile, level stamped `marfrit-p15` (= the deployed
  set, measured from git at `b0ccd47`); all image/workflow tags moved
  from `2026.4.0-marfrit-p16` to `2026.4.0-marfrit-p15`. **Action for
  the operator: get a complete `0037` re-exported by the patch author,
  then bump `PATCH_MAX` and the level.**

- **2026-09-22, run 4: tier 1 PASSED (`ghcr.io/syakyr/arcint-ov:
  2026.4.0-marfrit-p15` pushed), tier 2 failed at the ctest gate —
  `libtbb.so.12: cannot open shared object file`.** Root cause measured,
  not guessed: pulled the pushed tier-1 image layers via the GHCR registry
  API (no Docker needed) and read the ELF dynamic tags with a minimal
  parser. The wheel's original libs carry `DT_RPATH=$ORIGIN`
  (self-locating); the source-build libs that replace them carry
  `DT_RUNPATH=/opt/ov/temp/Linux_x86_64/tbb/lib` — build-tree paths that
  don't exist in the final image. RUNPATH is not inherited from the
  executable, so `arcint → libopenvino → libtbb` fails at the second
  hop. The deb survives this via load-order on the host; an image must
  not depend on that. Fix: tier 1 now installs
  `/etc/ld.so.conf.d/marfrit-openvino.conf` + `ldconfig` (with a gate
  that the cache picked it up). The deb's "no ld.so.conf entry"
  discipline protects shared hosts; inside a dedicated container nothing
  else links these sonames, so the deviation is safe and every chain
  resolves. Requires a full tier-1 rebuild (~90 min) — layer cache is
  not configured across runs.

## 9. Testing on the Pro B60 box

Once `ghcr.io/syakyr/arcint:investigation` exists:

```bash
# 1. Pull (public, or after docker login ghcr.io — see §8)
docker pull ghcr.io/syakyr/arcint:investigation

# 2. Device-free sanity: the stub serves without a card
docker run --rm ghcr.io/syakyr/arcint:investigation --stub --port 8090 &
curl -s http://127.0.0.1:8090/props
kill %1

# 3. Card test — same device recipe as kyuz0's toolboxes. Models are
#    mounted read-only; the IR directory layout arcint expects is in
#    README.md ("Supported model formats").
docker run --rm -it \
  --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -v "$MODELS_ROOT:/models:ro" \
  -v arcint-cache:/var/cache/arcint \
  -p 8080:8080 \
  ghcr.io/syakyr/arcint:investigation \
  --model /models/ov/<ir-dir> --model-id <allowlist-id> \
  --served-model-name <name> --device GPU.0 \
  --host 0.0.0.0 --port 8080 --n-ctx 262144 \
  --cache-dir /var/cache/arcint --queue-timeout 30

# 4. Inside the container, confirm the runtime says it is patched:
#    /props reports the engine sha; the OV plugin version string should
#    carry marfrit-p15 (visible in arcint's -v load log).
curl -s http://127.0.0.1:8080/v1/models
```

What the B60 run actually proves, in order: Level Zero sees the card
through `/dev/dri` (load log), the patched plugin loads at the expected
level (version string), the IR opens (model load), and the serving
surface answers end-to-end (`/v1/models`, a chat completion). A failure
at step 1 is the host driver/kernel; at step 2 the image's driver stack;
at step 3 the model mount; at step 4 the engine — the classes are
separable before any log diving.

