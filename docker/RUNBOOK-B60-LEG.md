# RUNBOOK — arcint vs llama.cpp on the B60 box (podman + llama-swap)

Self-contained operational doc for the B60 legs. Complements
`HANDOFF-B60.md` (which assumes docker and the packager's model root);
this one is adapted to the actual target: **podman, llama-swap,
`~/Models`, Intel's public `Qwen3.8-27B-int4-ov` IR, and the locally
reconstructed MTP head** built 2026-09-24 (see §3).

Two legs, strictly serial:

- **Leg 1 (serial):** serve with MTP **off**. Proves image + podman +
  device recipe + allowlist + serving chain. Expect ~25 t/s
  (measured-here, B60, this artifact).
- **Leg 2 (MTP):** add the reconstructed head files, serve with MTP
  **on**. Expect 36–38 t/s, draft acceptance 45–75%+.

Do not start Leg 2 until Leg 1 is green. Each leg has STOP gates; a
failed gate means stop and record, not work around.

Evidence classes: `measured-here` = this repo/CI measured it;
`card` = from `docs/mtp-head-card.md` / allowlist; `unverified` =
this leg establishes it.

---

## 0. Known facts to carry in

| Fact | Value | Class |
|---|---|---|
| Engine image | `ghcr.io/syakyr/arcint:investigation` | measured-here |
| Image digest | `sha256:e5d09a34…98b8e34` | measured-here |
| Runtime stamp to see in load log | `marfrit-p15` in OV version string | measured-here |
| Model | `OpenVINO/Qwen3.8-27B-int4-ov` → dir `qwen38-intel-int4-ov` | card |
| `--model-id` (entry id, NOT the dir alias) | `qwen3.8-27b-intel-int4` | code: model_registry.cpp:172 |
| Base IR lm_xml sha (16) | `ab94a08ce150de6a` | card |
| chat_template sha (16) | `c3cf9e34abf4f9e3` | card |
| tokenizer sha (16) | `87a7830d63fcf43b` | card |
| MTP head files | built 2026-09-24 from upstream shard 18/18 + Intel base | measured-here (this session) |
| No-MTP decode | 25.0 t/s | card |
| MTP decode (Intel layer + our lm_head) | 37.7–38.1 t/s, 93.9% code / 76.4% prose | card |
| MTP decode (our layer + our lm_head) | 36.3 t/s, 90.8% acceptance task | card |
| MTP head is a LOAD requirement | allowlist pins presence; `--mtp off` does not waive it (measured-here: first Leg 1 refused without the files) | measured-here |
| B60 PCI id | `8086:e211` (DRM numbering ≠ OpenVINO numbering) | measured-here |

**Do-NOTs (from HANDOFF §6 — these are incidents waiting to happen):**
- Do NOT bump `PATCH_MAX` or add patches 0034–0037. Loud failure is
  correct; a guessed patch is worse. Stop and escalate.
- Do NOT expose port 8080 without auth/TLS/reverse proxy. Engine has
  none, by design. LAN/loopback bind only.
- Do NOT re-run blindly if output is wrong — capture `/props`, load
  log, and the exact request first.
- Do NOT use `pkill -f arcint` — matches its own command chain. Kill
  by pid / `podman stop`.

---

## 1. Host pre-flight (physical host, before any container)

```bash
# 1a. Card identity by PCI id — NEVER by cardN (SOP §2)
for c in /sys/class/drm/card*; do
  echo "$c vendor=$(cat $c/device/vendor) device=$(cat $c/device/device)"
done
ls -l /dev/dri/
```
**GATE:** a card shows `vendor=0x8086 device=0xe211` and has a
`renderD*` node. Note which renderD — that is the B60. No e211 →
stop, wrong box.

```bash
# 1b. Driver
lsmod | grep -E '^(xe|i915)'
```
**GATE:** `xe` present. `i915` only → stop (i915-era paths untested).

```bash
# 1c. Card is free
fuser -v /dev/dri/* 2>&1
pgrep -af 'arcint|llama-server'
```
**GATE:** empty. Anything holding the card → stop; a leg against a
busy card localises nothing.

```bash
# 1d. Headroom (driver/USM memory charges the PHYSICAL host, not the container)
free -h
```
**GATE:** `MemAvailable` comfortably > 8 GiB before the leg.

```bash
# 1e. runtime + groups
podman --version
getent group video render
```

## 2. Sampler (SOP §1 — starts BEFORE the leg, leg aborts without it)

Minimum viable, on the physical host:

```bash
nohup bash -c 'while :; do date -Is; free -h; sleep 2; done' \
  >> ~/b60-leg-sampler.log 2>&1 &
SAMPLER=$!
sleep 5
[ -s ~/b60-leg-sampler.log ] || { echo "sampler not growing — ABORT"; kill $SAMPLER; exit 1; }
echo "sampler pid $SAMPLER"
```

Watch it during model load. If `MemAvailable` approaches 4 GiB,
`podman stop arcint-b60` immediately.

## 3. Artifacts on the box

### 3a. Base IR (download on the B60)

The allowlist keys on the **directory name** — it must be the alias:

```bash
pip install "huggingface_hub[cli]"   # or uv tool install hf
hf download OpenVINO/Qwen3.8-27B-int4-ov \
    --local-dir ~/Models/ov/qwen38-intel-int4-ov

cd ~/Models/ov/qwen38-intel-int4-ov
sha256sum openvino_language_model.xml | cut -c1-16   # MUST be ab94a08ce150de6a
sha256sum chat_template.jinja         | cut -c1-16   # MUST be c3cf9e34abf4f9e3
sha256sum tokenizer.json              | cut -c1-16   # MUST be 87a7830d63fcf43b
```
**GATE:** all three match. Any mismatch → stop; the engine's allowlist
will refuse it anyway, find out now, not at boot.

### 3b. MTP head (transfer from the build box) — REQUIRED FOR BOTH LEGS

Built 2026-09-24 on the build box at `~/mtp-build/out/`
(layer.bin 849,399,048 B / lm_head.bin 1,272,143,360 B; sizes match
`docs/mtp-head-card.md`).

**Corrected 2026-09-24 after the first Leg 1 attempt failed:**
`--mtp off` does NOT waive the head files. The allowlist entry pins
`mtp_head_exported: true` and `validate_artifact`
(`src/core/model_registry.cpp:489`) refuses the artifact at load when
the artifact reports the head absent — before any serving flag is
consulted. Detection rule (`src/core/artifact.cpp:459`):

    has_mtp_head = (openvino_mtp_layer.xml OR openvino_mtp_model.xml)
                   AND openvino_mtp_lm_head.xml present

Intel's IR already ships the layer half (`openvino_mtp_model.xml`).
The only missing files are our `openvino_mtp_lm_head.{xml,bin}`.
Copy the head files BEFORE Leg 1; `--mtp on/off` then selects use,
not loadability.

```bash
# from the build box:
rsync -avP --checksum ~/mtp-build/out/ <user>@<b60>:~/mtp-build-mtphead/
# on the B60, verify BEFORE installing:
cd ~/mtp-build-mtphead && sha256sum -c SHASUMS256.txt
# install beside the base IR (both legs):
mv ~/mtp-build-mtphead/openvino_mtp_*.{xml,bin} ~/Models/ov/qwen38-intel-int4-ov/
```

With all four files present the dir carries TWO layers (our
`openvino_mtp_layer` + Intel's `openvino_mtp_model`). Which one
`--mtp on` uses by default is UNVERIFIED — settle it from `--help`
(§4d) and the load log, and record it. The card's two pairings:
ours = 36.3 t/s @ 90.8%; Intel-layer via `--mtp-layer exported` =
37.7–38.1 t/s @ 93.9% code.

## 4. Pull + container sanity (failure-class separation)

```bash
# 4a. Pull and confirm digest
podman pull ghcr.io/syakyr/arcint:investigation
podman images --digests | grep arcint
```
**GATE:** digest matches `sha256:e5d09a34…98b8e34`. Different →
record it, stop (tag moved; ask before proceeding).

```bash
# 4b. Device-free stub sanity — failure here = broken IMAGE, not host
podman run --rm --entrypoint bash ghcr.io/syakyr/arcint:investigation \
  -c '/usr/bin/arcint --stub --port 8090 & S=$!; sleep 2; curl -fsS http://127.0.0.1:8090/props; kill $S'
```
**GATE:** `/props` JSON answers.

```bash
# 4c. Device recipe passes muster
podman run --rm --device /dev/dri \
  --annotation run.oci.keep_original_groups=1 \
  --security-opt seccomp=unconfined \
  --entrypoint bash ghcr.io/syakyr/arcint:investigation \
  -c 'ls -l /dev/dri/'
```
**GATE:** the render node from §1a is visible inside. If not: try
`--group-add keep-groups` (podman variant), then `--privileged` once
to isolate group-vs-driver. Record which worked.

```bash
# 4d. FLAG GROUND TRUTH — record actual spellings before Leg 1
podman run --rm --entrypoint bash ghcr.io/syakyr/arcint:investigation \
  -c '/usr/bin/arcint --help' | tee ~/arcint-help.txt
```
**Confirm before continuing:** exact spellings of `--mtp` /
`--mtp off` / `--mtp-layer`, `--model-id`, `--prefix-cache-mib`,
`--cache-dir`, `--queue-timeout`. The commands below use the spellings
from HANDOFF/card; if `--help` disagrees, `--help` wins — and note the
discrepancy in the record.

## 5. LEG 1 — serial (MTP off)

```bash
podman run -d --name arcint-b60 \
  --device /dev/dri \
  --annotation run.oci.keep_original_groups=1 \
  --security-opt seccomp=unconfined \
  -v ~/Models:/models:ro \
  -v arcint-cache:/var/cache/arcint \
  -p 127.0.0.1:8080:8080 \
  -e ZES_ENABLE_SYSMAN=1 \
  -e ZE_AFFINITY_MASK=0 \
  ghcr.io/syakyr/arcint:investigation \
  --model /models/qwen38-intel-int4-ov \
  --model-id qwen3.8-27b-intel-int4 \
  --served-model-name qwen3.8-27b \
  --device GPU.0 \
  --host 0.0.0.0 --port 8080 \
  --n-ctx 120000 \
  --prefix-cache-mib 8192 \
  --cache-dir /var/cache/arcint \
  --queue-timeout 30 \
  --mtp off \
  -v
podman logs -f arcint-b60
```

**GATES in the load log, in order:**
1. OpenVINO plugin version string contains **`marfrit-p15`**.
   If the load log doesn't print it, check the lib directly:
   `podman run --rm --entrypoint bash ghcr.io/syakyr/arcint:investigation \
    -c 'strings /usr/lib/libopenvino* /usr/lib/*/libopenvino* 2>/dev/null | grep -m1 marfrit'`.
   Anything else → STOP, wrong base was pulled; record the string.
2. Device named: record the FULL_DEVICE_NAME / mem size lines —
   confirms `GPU.0` is the e211 and not a phantom.
3. Model load completes with no allowlist refusal.
4. Record: load time (cold cache), VRAM used.

**Measured-here (B60 leg 2026-09-24), context ceiling:** with MTP
resident the reservation math gives **max ctx 126672 per lane** on
22.71 GiB usable (weights+graph 13.06 + drafters 3.16 + MTP state
0.97 + activations 0.60 + margin 0.25 + GDN rows 303 MiB + KV
36.2 KiB/token). `--n-ctx 262144` does NOT fit this config; 120000
is the working value. Lever for more: lower `--prefix-cache-mib`.

**Speculation engages ONLY under greedy** (`src/api/handlers.cpp:46`).
A sampled request (temp > 0) shows `draft accept 0.0% (0/0)` BY
DESIGN and gets serial speed. To exercise MTP, send
`"temperature": 0`. All card acceptance numbers are greedy.

**Serve test:**
```bash
curl -s http://localhost:8080/v1/models
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"introduce yourself in one line"}],"max_tokens":64}'
```
**GATE:** sane output. Wrong/garbage output → do NOT re-run; capture
`/props`, full load log, exact request; this is a real defect.

**Record:** decode t/s from the response usage (or timed curl).
Reference: 25.0 t/s no-MTP (card).

## 6. LEG 2 — MTP on

Only after Leg 1 green. Install the head files (§3b), then:

```bash
podman stop arcint-b60 && podman rm arcint-b60
# same podman run as §5, with --mtp off replaced by --mtp on
```

**GATES:**
- **Test with `"temperature": 0`** — drafting is gated to greedy
  (handlers.cpp:46); sampled requests draft zero and look inert.
- With BOTH our `openvino_mtp_layer` and Intel's `openvino_mtp_model`
  in the dir, the engine defaults to **ours** (measured-here:
  `mtp: layer (reconstructed)`). The Intel pairing needs
  `--mtp-layer exported`.
- Decode lines print `draft accept NN% (…)`.
  - **45–75%+ → head is correct** (this is the oracle: a wrong head
    cannot change outputs, only make speculation useless).
  - **~0% under greedy → STOP.** Head/IR pairing or flag wiring is
    wrong. Record the acceptance line and the file shas; escalate.
- Reference points (card): our layer + our lm_head = 36.3 t/s @ 90.8%
  (acceptance task); Intel's `openvino_mtp_model` + our lm_head via
  `--mtp-layer exported` = 37.7–38.1 t/s @ 93.9% code / 76.4% prose.
- Record which layer you served (ours vs Intel's exported one) — the
  numbers are not interchangeable.

## 7. Cold vs warm cache (quantifies the volume)

After Leg 1 or 2, once, deliberately:

```bash
podman stop arcint-b60 && podman rm arcint-b60
podman volume rm arcint-cache
# re-run Leg 1; time to "ready"
```
Record: cold-start vs warm-start time. That is the volume's worth.

## 8. llama-swap entries (after manual legs are green)

Add alongside the existing llama baselines — same proxy/stop/ttl
pattern as the sycl entries:

```yaml
  "qwen3.8-27b-int4-arcint-serial":
    proxy: "http://127.0.0.1:${PORT}"
    checkEndpoint: /v1/models
    cmd: |
      podman run --rm --name ${MODEL_ID}
      --device /dev/dri
      --annotation run.oci.keep_original_groups=1
      --security-opt seccomp=unconfined
      -p 127.0.0.1:${PORT}:${PORT}
      -v /home/amadeus/Models:/models:ro
      -v arcint-cache:/var/cache/arcint
      -e ZES_ENABLE_SYSMAN=1 -e ZE_AFFINITY_MASK=0
      ghcr.io/syakyr/arcint:investigation
      --model /models/qwen38-intel-int4-ov
      --model-id qwen3.8-27b-intel-int4
      --served-model-name qwen3.8-27b
      --device GPU.0 --host 0.0.0.0 --port ${PORT}
      --n-ctx 120000 --prefix-cache-mib 8192
      --cache-dir /var/cache/arcint --queue-timeout 30
      --mtp off
    cmdStop: podman stop ${MODEL_ID}
    ttl: 3600
    healthCheckTimeout: 900   # cold IR compile; tighten after first measure
```
(+ a `-mtp` twin with `--mtp on` once Leg 2 is green.)

**Comparison discipline vs the sycl baselines:**
- Match `--n-ctx` to the baseline being compared (120000 here; the
  arcint ceiling with MTP resident is 126672), not the engine max —
  KV pressure moves numbers.
- Align `--served-model-name` with the model name llama-swap actually
  sends, or the engine logs a mismatch every request (single-model
  aliasing papers over it; don't rely on it).
- Prompts over 2048 tokens prefill in chunks and chunk boundaries are
  NOT bit-exact on this backend (DESIGN 3.2, logged at load). Note it
  in any output-equality comparison vs llama.cpp.
- The quants differ (int4 NNCF IR vs IQ3_S / Q4_K_XL GGUF). This is
  engine-vs-engine at each engine's served artifact, not a
  matched-weights A/B. Say so in any report.
- MTP depth differs: arcint = 1-layer head; llama entries =
  `draft-mtp n-max 2`.
- Pull t/s for both from llama-swap's `metrics.sqlite` for the same
  prompt set.

## 9. Failure triage

| Symptom | Layer | First check |
|---|---|---|
| pull denied | registry | `podman login ghcr.io` with read:packages PAT; visibility flip |
| 4b stub silent | image | re-pull by digest; compare `podman inspect` |
| libopenvino/libtbb load error | image loader | should be impossible (ldconfig gate); if seen: wrong base, record digest |
| Level Zero / /dev/dri errors at model load | host driver / device recipe | §1a/1b; try `--privileged` once to isolate group-vs-driver |
| allowlist refusal | mount or flag pair | dir basename = alias `qwen38-intel-int4-ov`; `--model-id` = ENTRY ID `qwen3.8-27b-intel-int4` (they differ — dir is the alias, id is dotted) |
| `MTP head mismatch: allowlist present, artifact absent` (with `--mtp off` too) | allowlist presence pin | head files are a LOAD requirement, not a use option (§3b); install at least `openvino_mtp_lm_head.{xml,bin}` beside the base IR |
| `--mtp` unknown flag | flag spelling | §4d ground truth; card says `--mtp on`, `--mtp-layer exported` |
| accept ~0% | head pairing | shas vs SHASUMS256.txt; which layer served; stop, escalate |
| serves, wrong output | engine | capture /props + load log + request; do NOT re-run blindly |
| sampler shows MemAvailable < 4 GiB | host memory | stop the container NOW; this is the SOP watchdog firing |

## 10. After the window (SOP §5)

```bash
podman stop arcint-b60            # by name — never pkill -f
kill $SAMPLER                     # the recorded pid
fuser -v /dev/dri/*               # must be empty
pgrep -af arcint                  # must be empty (host AND podman ps -a)
```
Write the release line: timestamp, leftover count, card left as found.

## 11. Recording sheet (this leg's deliverable)

Fill and keep with the leg:

```
[ ] Image digest pulled: ____________ (expect e5d09a34…98b8e34)
[ ] marfrit-p15 seen in load log: yes / NO(→stop)
[ ] GPU.0 = e211 confirmed via: ____________
[ ] Leg 1 cold load time: ____ s   VRAM used @120k ctx: ____ GiB (ceiling 126672 w/ MTP resident)
[ ] Leg 1 decode t/s: ____  (ref 25.0 card)
[ ] Leg 2 layer served: ours / intel-exported
[ ] Leg 2 accept %: ____  decode t/s: ____  (refs 36.3@90.8 / 37.7-38.1@93.9)
[ ] Cold vs warm cache: ____ s vs ____ s
[ ] Flag deviations from this runbook (from --help ground truth): ____
[ ] Podman device-recipe variant that worked: annotation / keep-groups / privileged
[ ] Anything that made you STOP, with the captured log lines: ____
```
