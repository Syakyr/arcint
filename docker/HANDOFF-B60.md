# HANDOFF — spinning up the arcint toolbox on the Pro B60 box

Next session: the Docker images are built and pushed. This document is the
runbook for bringing the engine up on the machine with the Intel Arc Pro
B60 (24 GiB, Xe2, `xe` driver), what to verify, and what to record.

Read `INVESTIGATION.md` in this directory for the design rationale and
the full status log; this file is only the operational slice.

Evidence classes per repo convention: `measured-here` = verified from
this side (CI logs, registry API) on 2026-09-23; `unverified` = the
B60 leg itself, which is what the next session does.

---

## 1. What exists (all `measured-here`)

| Artifact | Tag | Digest |
|---|---|---|
| Engine image | `ghcr.io/syakyr/arcint:investigation` | `sha256:e5d09a34…98b8e34` |
| Engine immutable | `ghcr.io/syakyr/arcint:20260923T071234` | same |
| Patched OV base | `ghcr.io/syakyr/arcint-ov:2026.4.0-marfrit-p15` | `sha256:99b28425…1f776e` |

Provenance chain: engine built at commit `61aab5b` (branch
`docker-toolbox-investigation`) FROM the p15 base, which is OpenVINO
`71640275d29692354223572e77ef717e92b891d9` + patches **0003–0033**
(the deployed +p15 set, 31 patches) + the upstream nightly tokenizers
wheel. The base's `libopenvino` carries `marfrit-p15` in its version
string — the load log answers "which runtime" without a detour.

Registry access: **anonymous pull verified working** (`measured-here`,
HTTP 200 on the manifest). No login needed on the B60 box.

Already verified in CI (`measured-here`):
- compiles against the patched base;
- full device-free unit ladder green (`ctest -L unit`: unit, roundtrip,
  stress, acceptance-enumeration, acceptance-runner);
- stub smoke test green inside the final image (`--stub` serves `/props`);
- `ldconfig` resolves the OV libs tree inside the image (the
  build-tree-RUNPATH trap is closed — see INVESTIGATION.md §8).

NOT verified anywhere yet: anything that touches a real card. That is
this leg.

## 2. Host prerequisites on the B60 box

- Kernel with the **`xe`** driver and `/dev/dri/renderD*` present
  (`ls -l /dev/dri/`). arcint/DEVELOPMENT.md: i915-era paths untested.
- Docker (or podman — flags translate; `--group-add` becomes
  `--group-add keep-groups` under podman for group passthrough).
- Model artifacts: an OpenVINO **IR directory** from the allowlist
  (`models/allowlist-raw.json` — e.g. `qwen36-coder-b5-ov`,
  `qwen38-b7c1-ov`, `qwen35-2b-ov`) present on the host. The B60
  fleet's models live under `/models/ov/` on the packager's hosts;
  copy or mount whatever root holds them.
- **LAN only.** The engine has no auth, no TLS, permissive CORS
  (DEVELOPMENT.md). Bind to a LAN interface or put a reverse proxy in
  front; never expose 8080 raw.

## 3. Spin-up procedure

Run order matters: each step separates a failure class before the next.

**Step 1 — pull:**
```bash
docker pull ghcr.io/syakyr/arcint:investigation
```

**Step 2 — device-free sanity (no card needed):**
```bash
docker run --rm --entrypoint bash ghcr.io/syakyr/arcint:investigation \
  -c '/usr/bin/arcint --stub --port 8090 & S=$!; sleep 2; curl -fsS http://127.0.0.1:8090/props; kill $S'
```
Expected: `/props` JSON answers. Note the `--entrypoint bash` — the
image's ENTRYPOINT is `arcint` itself. Failure here = broken image,
not the host.

**Step 3 — card visibility (no model):**
```bash
docker run --rm --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  --entrypoint bash ghcr.io/syakyr/arcint:investigation \
  -c 'ls -l /dev/dri/ && /usr/bin/arcint --stub --port 8090 >/dev/null 2>&1 & sleep 1; echo ok'
```
(Step 3 mostly checks the device recipe passes muster; the real Level
Zero check is step 4's load log.)

**Step 4 — the real run.** Mount the model root read-only, give the
blob cache a named volume (a cold cache = full kernel recompile on
every start), and run with the flags from the systemd template:
```bash
docker run -d --name arcint-b60 \
  --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -v /path/to/models:/models:ro \
  -v arcint-cache:/var/cache/arcint \
  -p 8080:8080 \
  ghcr.io/syakyr/arcint:investigation \
  --model /models/ov/qwen36-coder-b5-ov \
  --model-id qwen3.6-27b-a3b-coder \
  --served-model-name qwen3.6-coder \
  --device GPU.0 \
  --host 0.0.0.0 --port 8080 \
  --n-ctx 262144 \
  --prefix-cache-mib 8192 \
  --cache-dir /var/cache/arcint \
  --queue-timeout 30 \
  -v
docker logs -f arcint-b60
```
Swap the `--model`/`--model-id` pair for whichever allowlisted artifact
is on the box; `--model-id` is the allowlist assertion and the engine
refuses a mismatched artifact.

**Step 5 — verify the patched runtime is the one loaded.** In the load
log (`docker logs arcint-b60`), the OpenVINO plugin version string
must contain **`marfrit-p15`**. If it says anything else, stop — wrong
base was pulled, record it.

**Step 6 — serve:**
```bash
curl -s http://localhost:8080/v1/models
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-coder","messages":[{"role":"user","content":"introduce yourself in one line"}],"max_tokens":64}'
```

## 4. Failure triage (which layer died)

| Symptom | Layer | First thing to check |
|---|---|---|
| pull fails 401/denied | registry | package visibility flipped to private? `docker login ghcr.io -u syakyr` with a `read:packages` PAT |
| step 2 `/props` silent | image | re-pull by digest `sha256:e5d09a34…`; compare `docker inspect` |
| `libopenvino`/`libtbb` load error | image loader | should be impossible now (ldconfig gate in base); if seen: wrong base tag, record digest |
| Level Zero / `/dev/dri` errors at model load | host driver / device recipe | host `xe` driver loaded? groups passed? try `--privileged` once to isolate group vs driver |
| model refused / allowlist error | mount or flag pair | `--model` path exists in container? `--model-id` matches the artifact per `models/allowlist-raw.json`? |
| serves but wrong output | engine | this is a real defect — capture `/props`, load log, and the request; do not re-run blindly |

## 5. What to record for the record (this leg's deliverable)

1. `docker images --digests` line for the pulled image (confirms the
   digest above).
2. The load-log lines: model load time, the `marfrit-p15` version
   string, VRAM reported/used on the 24 GiB card at `--n-ctx 262144`.
3. A decode rate from a real generation (compare against the README's
   bare-metal B60 numbers — the container should be within noise; a
   large gap is itself a finding).
4. Cold vs warm cache start times (drop the `arcint-cache` volume once
   and compare) — quantifies what the volume is worth.
5. Any deviation from the flags above that was needed (e.g. podman
   group handling, extra devices) — that updates this runbook.

## 6. Known open items (do not silently "fix" these)

- **Patch 0037 is truncated** in this repo and upstream (INVESTIGATION.md
  §5.1). The image is deliberately **p15** (0003–0033). Do NOT bump
  `PATCH_MAX` until the author re-exports 0037; a guessed patch is
  worse than the loud failure.
- **0034–0036 are NOT in the image** — they were never stamped into a
  deployed deb either. If the B60 leg needs a fix those patches carry,
  that changes the level story: stop and escalate rather than bumping
  `PATCH_MAX` ad hoc.
- **No auth/TLS** — LAN posture is by design; a public bind is an
  operator incident, not a feature gap.
- **Acceptance cells have not run in the container.** CI runs the
  device-free ladder only. If the B60 session wants the full 12-cell
  acceptance run against the image, that is `tests/acceptance/run.py`
  with the card — a separate, deliberate leg (test-ladder rules apply:
  never bare `ctest`).

## 7. Rebuild (if needed)

```bash
gh workflow run build-toolbox.yml --repo Syakyr/arcint \
  --ref docker-toolbox-investigation -f tiers=arcint -f channel=investigation
```
`tiers=arcint` reuses the pushed p15 base (minutes); `tiers=all`
rebuilds the base too (~90 min). Tier gating is computed in the `ns`
job and echoed to the run summary — every skip is explainable.
