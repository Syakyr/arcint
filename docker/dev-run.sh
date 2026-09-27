#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# dev-run.sh — one-command launch of the arcint dev image on the B60 host.
#
# Everything the container needs is encoded here so the loop is one command
# rather than a block of flags re-typed from a wiki page. Each flag is commented
# with the failure it prevents; do not strip them without reading why.
#
# Usage:
#   ./docker/dev-run.sh              # interactive shell in the dev container
#   ./docker/dev-run.sh --check      # verify the host + image, run nothing
#   ./docker/dev-run.sh -- <cmd>     # run a specific command instead of a shell
#
# Credentials are RUNTIME-ONLY. Nothing here is baked into an image:
#   GH_TOKEN              GitHub API token (agent push/PR)         [optional]
#   GITAMADEUS_KEY        path to a READ-ONLY deploy key file     [optional]
#   <ANY_OTHER_KEY>       any further API key is passed through (see PASS_THRU)
#
# The gitAmadeus key MUST be a read-only deploy key scoped to the one repo —
# never a personal SSH key. If the image or the key leaks, the blast radius of a
# read-only single-repo deploy key is "one repo readable"; a personal key is
# "your whole account".
# ---------------------------------------------------------------------------
set -euo pipefail

DEV_IMAGE="${DEV_IMAGE:-ghcr.io/syakyr/arcint-dev:dev}"
ARCINT_REPO="${ARCINT_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
GITAMADEUS_KEY="${GITAMADEUS_KEY:-}"
CARD_LOCK="${CARD_LOCK:-/tmp/arcint-b60.lock}"

# Env vars forwarded into the container. Add new API keys here.
PASS_THRU=(
  GH_TOKEN
  GITAMADEUS_KEY
  HF_TOKEN
  HF_HOME
  HTTPS_PROXY HTTP_PROXY NO_PROXY
  ARCINT_DEVICE
  ARCINT_PORT
)

MODE="shell"
CHECK_ONLY=0
declare -a USER_CMD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)   CHECK_ONLY=1; shift ;;
    --)        shift; USER_CMD=("$@"); MODE="cmd"; break ;;
    -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

ok()   { printf '  \033[32mOK\033[0m   %s\n' "$1"; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; }

# ---------------------------------------------------------------------------
# Host preflight. These are the checks that save hours of chasing a mis-set-up
# box, and several of them are specific to this experiment's hardware notes.
# ---------------------------------------------------------------------------
preflight() {
  echo "== host preflight =="

  # 1. Container runtime present
  local rt=""
  for c in docker podman; do command -v "$c" >/dev/null 2>&1 && { rt="$c"; break; }; done
  if [[ -z "$rt" ]]; then bad "no docker or podman on PATH"; return 1; fi
  ok "container runtime: $rt"
  RUNTIME="$rt"

  # 2. Render node present and group-accessible.
  #    Arc compute needs /dev/dri/renderD*; without it the GPU is invisible.
  if compgen -G "/dev/dri/renderD*" >/dev/null; then
    ok "render nodes: $(ls /dev/dri/renderD* 2>/dev/null | tr '\n' ' ')"
  else
    bad "no /dev/dri/renderD* — is the i915/xe driver loaded? (lspci -nnk | grep -A3 -i vga)"
    return 1
  fi

  # 3. Card count. The experiment assumes 2x B60; a single card changes the
  #    whole question, so say so loudly rather than silently proceeding.
  local ncards
  ncards=$(ls /dev/dri/renderD* 2>/dev/null | wc -l)
  if [[ "$ncards" -ge 2 ]]; then
    ok "render nodes present: $ncards (dual-card experiment is possible)"
  else
    warn "only $ncards render node — dual-B60 legs cannot run; single-card only"
  fi

  # 4. PCIe link width/speed. THE number for this experiment is x8 Gen4
  #    (~15.75 GB/s theoretical, ~13-15 practical). Boards silently negotiate
  #    lower, and a x4 or Gen3 link invalidates every bandwidth estimate.
  echo "  -- PCIe link (check LnkSta against LnkCap; a mismatch = silent downgrade) --"
  if command -v lspci >/dev/null 2>&1; then
    lspci -vvv 2>/dev/null | grep -E 'LnkSta:|LnkCap:' | head -8 || warn "lspci produced no link info"
  else
    warn "lspci not found — cannot verify link width/speed"
  fi

  # 5. Host RAM bandwidth context. The design assumes 8-channel DDR4-3200
  #    (~204.8 GB/s theoretical). Record what is actually there.
  if command -v numactl >/dev/null 2>&1; then
    echo "  -- NUMA / memory (record the NPS setting; it changes DMA locality) --"
    numactl --hardware 2>/dev/null | head -8 || warn "numactl produced nothing"
  else
    warn "numactl not installed — install numactl to record NUMA topology"
  fi

  # 6. Host memory total. 256 GB is the assumption that makes the "free RAM"
  #    caveat from the 2x3090 reference not apply to us.
  local memgb
  memgb=$(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo)
  if [[ "$memgb" -ge 200 ]]; then
    ok "host RAM: ${memgb} GB (headroom for ~60 GB experts + ~51 GB PLE table)"
  else
    warn "host RAM: ${memgb} GB — less than assumed; decode may be RAM-pressured"
  fi

  # 7. Disk space for caches + build tree + image layers.
  local freegb
  freegb=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
  if [[ -n "$freegb" && "$freegb" -lt 40 ]]; then
    warn "only ${freegb} GB free on / — image + build tree + caches may not fit"
  else
    ok "disk free on /: ${freegb:-unknown} GB"
  fi

  return 0
}

# ---------------------------------------------------------------------------
# Card lock. The card-window SOP assumes exclusive possession. A second process
# touching the GPU mid-leg invalidates the measurement AND looks like a code
# regression in whatever is under test — the worst possible failure mode, because
# it sends you chasing a bug that is not in the code.
# ---------------------------------------------------------------------------
acquire_lock() {
  if [[ -e "$CARD_LOCK" ]]; then
    local holder
    holder=$(cat "$CARD_LOCK" 2>/dev/null || echo "")
    if [[ -n "$holder" ]] && kill -0 "$holder" 2>/dev/null; then
      bad "card is locked by pid $holder ($CARD_LOCK)."
      echo "     Another session holds the B60. Do not run a leg on top of it."
      echo "     If the holder is gone but the lock is stale: rm -f $CARD_LOCK"
      return 1
    fi
    warn "stale lock from pid ${holder:-?} — reclaiming"
  fi
  echo $$ > "$CARD_LOCK"
  trap 'rm -f "$CARD_LOCK" 2>/dev/null || true' EXIT INT TERM
  ok "card lock acquired ($CARD_LOCK, pid $$)"
}

# ---------------------------------------------------------------------------
# Assemble the run.
# ---------------------------------------------------------------------------
build_args() {
  ARGS=()

  # GPU access. Arc needs the DRM render node plus the video/render groups.
  ARGS+=(--device /dev/dri:/dev/dri --group-add video --group-add render)

  # --pid=host: docs/sop-card-window.md requires a zombie sweep BY PID. Without
  # the host PID namespace the agent cannot see or kill an orphaned process
  # holding the GPU, and a zombie mid-leg poisons every later measurement.
  ARGS+=(--pid=host)

  # --network=host: the engine binds 127.0.0.1 inside its own netns, so a
  # host-side probe then has nothing to connect to. Measured, not theoretical:
  # run 35830030865, "curl (7)" while the stub was logging 'listening'.
  ARGS+=(--network=host)

  # Source tree + the operator's audited pi state.
  # ~/.pi is MOUNTED, never baked: the pi-audit protocol gates extensions on
  # approved pinned refs with a recorded snapshotSha256. Baking extensions in
  # would produce an agent nobody approved and nothing can verify.
  ARGS+=(-v "${ARCINT_REPO}:/work/arcint")
  ARGS+=(-v "${HOME}/.pi:/root/.pi")

  # Model weights: bind-mounted, never an image layer.
  if [[ -d "$MODELS_ROOT" ]]; then
    ARGS+=(-v "${MODELS_ROOT}:/models:ro")
  else
    warn "MODELS_ROOT=${MODELS_ROOT} does not exist — no models mounted"
  fi

  # OpenVINO GPU kernel cache. A cold cache turns EVERY start into a recompile.
  # Persistent and writable, same reason as docker-compose.example.yaml.
  ARGS+=(-v arcint-cache:/var/cache/arcint)

  # gitAmadeus deploy key: read-only bind mount, never baked.
  if [[ -n "$GITAMADEUS_KEY" && -f "$GITAMADEUS_KEY" ]]; then
    ARGS+=(-v "${GITAMADEUS_KEY}:/run/secrets/gitamadeus_key:ro")
    ok "gitAmadeus deploy key mounted read-only"
  else
    warn "no gitAmadeus deploy key (GITAMADEUS_KEY unset) — private clones unavailable"
  fi

  # Forward the API keys.
  for k in "${PASS_THRU[@]}"; do
    if [[ -n "${!k:-}" ]]; then ARGS+=(-e "$k"); fi
  done

  ARGS+=("$DEV_IMAGE")
}

# ---------------------------------------------------------------------------
main() {
  echo "dev image: $DEV_IMAGE"
  echo "repo:      $ARCINT_REPO"
  echo

  preflight || { echo; echo "preflight FAILED — fix the above before running a leg." >&2; exit 1; }

  if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo
    echo "== check-only: nothing launched =="
    echo "   to pull:  ${RUNTIME:-docker} pull $DEV_IMAGE"
    echo "   to run:   ./docker/dev-run.sh"
    exit 0
  fi

  acquire_lock || exit 1

  build_args

  echo
  if [[ "$MODE" == "cmd" ]]; then
    exec "${RUNTIME[@]}" run --rm -i "${ARGS[@]}" /bin/bash -c "${USER_CMD[*]}"
  else
    exec "${RUNTIME[@]}" run --rm -it "${ARGS[@]}" /bin/bash
  fi
}

main "$@"
