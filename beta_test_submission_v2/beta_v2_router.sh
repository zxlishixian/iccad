#!/usr/bin/env sh
# Experimental baseline-first anytime router.  It deliberately lives outside
# beta_test_submission until the new policy passes cold-cache validation.
set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${BETA_V2_PACKAGE_ROOT:-}" ]; then
  PACKAGE_ROOT=$BETA_V2_PACKAGE_ROOT
elif [ -x "$SCRIPT_DIR/fast/regr_fail_bucketing_fast/regr_fail_bucketing_fast" ]; then
  PACKAGE_ROOT=$SCRIPT_DIR
else
  PACKAGE_ROOT="$SCRIPT_DIR/beta_test_submission"
fi

export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

INPUT=""
OUTPUT=""
K=""
PREV=""
for ARG in "$@"; do
  case "$PREV" in
    --input) INPUT="$ARG" ;;
    --output) OUTPUT="$ARG" ;;
    --k) K="$ARG" ;;
  esac
  case "$ARG" in
    --input=*) INPUT=${ARG#--input=} ;;
    --output=*) OUTPUT=${ARG#--output=} ;;
    --k=*) K=${ARG#--k=} ;;
  esac
  PREV="$ARG"
done

FAST_BIN="${BETA_V2_FAST_BIN:-$PACKAGE_ROOT/fast/regr_fail_bucketing_fast/regr_fail_bucketing_fast}"
MULTIVIEW_BIN="${BETA_V2_MULTIVIEW_BIN:-$PACKAGE_ROOT/multiview/regr_fail_bucketing_multiview}"
FULL_BIN="${BETA_V2_FULL_BIN:-$PACKAGE_ROOT/regr_fail_bucketing_full}"

if [ -z "$INPUT" ] || [ ! -f "$INPUT" ] || [ -z "$OUTPUT" ] || [ -z "$K" ]; then
  echo "[beta-v2] invalid required arguments" >&2
  exit 2
fi

N_CASES=$(awk 'END { print (NR > 0) ? NR - 1 : 0 }' "$INPUT")
EXPECTED_LINES=$((N_CASES + 1))
OUT_DIR=$(dirname -- "$OUTPUT")
mkdir -p "$OUT_DIR" 2>/dev/null || exit 2
BASELINE_CANDIDATE="${OUTPUT}.baseline.$$.candidate"
EXPERT_CANDIDATE="${OUTPUT}.expert.$$.candidate"
ACTIVE_CHILD=""
ACTIVE_WATCHDOG=""
SELECTED="singleton"

monotonic_sec() {
  awk '{ printf "%d", $1 }' /proc/uptime
}

valid_output() {
  CANDIDATE=$1
  [ -s "$CANDIDATE" ] || return 1
  HEADER=$(sed -n '1{s/\r$//;p;}' "$CANDIDATE")
  [ "$HEADER" = "Case,bucket" ] || return 1
  ACTUAL_LINES=$(awk 'END { print NR }' "$CANDIDATE")
  [ "$ACTUAL_LINES" -eq "$EXPECTED_LINES" ] || return 1
  awk -F',' 'NR > 1 { if ($2 == "" || $2 == "\r") exit 1 }' "$CANDIDATE"
}

cleanup_processes() {
  if [ -n "$ACTIVE_CHILD" ]; then
    kill -TERM "$ACTIVE_CHILD" 2>/dev/null || true
    sleep 1
    kill -KILL "$ACTIVE_CHILD" 2>/dev/null || true
  fi
  if [ -n "$ACTIVE_WATCHDOG" ]; then
    kill "$ACTIVE_WATCHDOG" 2>/dev/null || true
  fi
  rm -f "$BASELINE_CANDIDATE" "$EXPERT_CANDIDATE" 2>/dev/null || true
}

handle_signal() {
  echo "[beta-v2] received termination signal; preserving $SELECTED output" >&2
  cleanup_processes
  exit 0
}
trap handle_signal TERM INT HUP
trap cleanup_processes EXIT

write_singletons() {
  TMP_OUTPUT="${OUTPUT}.emergency.$$.tmp"
  if awk -F',' '
    BEGIN { print "Case,bucket" }
    NR > 1 {
      case_id=$1
      sub(/^\xef\xbb\xbf/, "", case_id)
      sub(/^"/, "", case_id)
      sub(/"$/, "", case_id)
      sub(/\r$/, "", case_id)
      printf "%s,bucket_emergency_%06d\n", case_id, NR - 2
    }
  ' "$INPUT" > "$TMP_OUTPUT" && valid_output "$TMP_OUTPUT"; then
    mv -f "$TMP_OUTPUT" "$OUTPUT"
    return 0
  fi
  rm -f "$TMP_OUTPUT" 2>/dev/null || true
  return 1
}

publish_candidate() {
  CANDIDATE=$1
  NAME=$2
  if valid_output "$CANDIDATE"; then
    mv -f "$CANDIDATE" "$OUTPUT"
    SELECTED=$NAME
    echo "[beta-v2] published $NAME output" >&2
    return 0
  fi
  return 1
}

run_candidate() {
  LIMIT=$1
  CANDIDATE=$2
  shift 2
  rm -f "$CANDIDATE" 2>/dev/null || true
  "$@" &
  ACTIVE_CHILD=$!
  (
    sleep "$LIMIT"
    kill -TERM "$ACTIVE_CHILD" 2>/dev/null || exit 0
    sleep 1
    kill -KILL "$ACTIVE_CHILD" 2>/dev/null || true
  ) &
  ACTIVE_WATCHDOG=$!
  if wait "$ACTIVE_CHILD"; then
    STATUS=0
  else
    STATUS=$?
  fi
  kill "$ACTIVE_WATCHDOG" 2>/dev/null || true
  wait "$ACTIVE_WATCHDOG" 2>/dev/null || true
  ACTIVE_CHILD=""
  ACTIVE_WATCHDOG=""
  return "$STATUS"
}

START_SEC=$(monotonic_sec)
if ! write_singletons; then
  echo "[beta-v2] failed to create emergency output" >&2
  exit 2
fi
echo "[beta-v2] published singleton output cases=$N_CASES" >&2

# Map observable case count and soft k to the official scale.  Context size can
# only shorten expert work; it never raises the conservative Final deadline.
if [ "$N_CASES" -le 10 ]; then CASE_SCALE=2
elif [ "$N_CASES" -le 30 ]; then CASE_SCALE=4
elif [ "$N_CASES" -le 100 ]; then CASE_SCALE=8
elif [ "$N_CASES" -le 300 ]; then CASE_SCALE=16
elif [ "$N_CASES" -le 1000 ]; then CASE_SCALE=32
else CASE_SCALE=64
fi

if [ "$K" -le 2 ]; then K_SCALE=2
elif [ "$K" -le 4 ]; then K_SCALE=4
elif [ "$K" -le 8 ]; then K_SCALE=8
elif [ "$K" -le 16 ]; then K_SCALE=16
elif [ "$K" -le 32 ]; then K_SCALE=32
else K_SCALE=64
fi
if [ "$CASE_SCALE" -ge "$K_SCALE" ]; then OFFICIAL_SCALE=$CASE_SCALE; else OFFICIAL_SCALE=$K_SCALE; fi

INPUT_DIR=$(dirname -- "$INPUT")
PLAIN_BYTES=$(find "$INPUT_DIR" -type f -name '*.log' -printf '%s\n' 2>/dev/null | awk '{ total += $1 } END { printf "%.0f", total + 0 }')
GZIP_BYTES=$(find "$INPUT_DIR" -type f -name '*.log.gz' -printf '%s\n' 2>/dev/null | awk '{ total += $1 } END { printf "%.0f", total + 0 }')
EST_CONTEXT_LINES=$((PLAIN_BYTES / 128 + GZIP_BYTES / 24))
ONE_M_CONTEXT_EST_LINES="${BETA_V2_ONE_M_CONTEXT_EST_LINES:-1500000}"
LONG_CONTEXT_EST_LINES="${BETA_V2_LONG_CONTEXT_EST_LINES:-20000000}"
if [ "$EST_CONTEXT_LINES" -le "$ONE_M_CONTEXT_EST_LINES" ]; then CONTEXT_CLASS=one_m_like
elif [ "$EST_CONTEXT_LINES" -ge "$LONG_CONTEXT_EST_LINES" ]; then CONTEXT_CLASS=hundred_m_like
else CONTEXT_CLASS=ten_m_like
fi

if [ "$OFFICIAL_SCALE" -le 4 ]; then
  FINAL_LIMIT=30
  BASELINE_LIMIT="${BETA_V2_BASELINE_LIMIT:-7}"
  DESIRED_EXPERT_LIMIT="${BETA_V2_EXPERT_LIMIT:-18}"
elif [ "$CONTEXT_CLASS" = "hundred_m_like" ]; then
  FINAL_LIMIT=100
  BASELINE_LIMIT="${BETA_V2_BASELINE_LIMIT:-15}"
  DESIRED_EXPERT_LIMIT="${BETA_V2_EXPERT_LIMIT:-60}"
else
  FINAL_LIMIT=100
  BASELINE_LIMIT="${BETA_V2_BASELINE_LIMIT:-12}"
  if [ "$OFFICIAL_SCALE" -le 8 ]; then
    DESIRED_EXPERT_LIMIT="${BETA_V2_EXPERT_LIMIT:-72}"
  else
    DESIRED_EXPERT_LIMIT="${BETA_V2_EXPERT_LIMIT:-70}"
  fi
fi
EXIT_RESERVE="${BETA_V2_EXIT_RESERVE:-4}"
DEADLINE_SEC=$((START_SEC + FINAL_LIMIT - EXIT_RESERVE))

echo "[beta-v2] cases=$N_CASES k=$K scale=$OFFICIAL_SCALE context=$CONTEXT_CLASS final=${FINAL_LIMIT}s baseline=${BASELINE_LIMIT}s desired_expert=${DESIRED_EXPERT_LIMIT}s" >&2

# Stage 1: establish a meaningful deterministic result before any API call.
if [ "$N_CASES" -le "${BETA_V2_AGGLOM_MAX_CASES:-900}" ]; then
  echo "[beta-v2] trying deterministic agglomerative baseline" >&2
  if run_candidate "$BASELINE_LIMIT" "$BASELINE_CANDIDATE" "$FAST_BIN" "$@" --output "$BASELINE_CANDIDATE" --llm-mode none --cluster agglomerative; then
    publish_candidate "$BASELINE_CANDIDATE" baseline || true
  fi
else
  echo "[beta-v2] trying deterministic kmeans baseline" >&2
  if run_candidate "$BASELINE_LIMIT" "$BASELINE_CANDIDATE" "$FAST_BIN" "$@" --output "$BASELINE_CANDIDATE" --llm-mode none --cluster kmeans --cluster-factor 1.0; then
    publish_candidate "$BASELINE_CANDIDATE" baseline || true
  fi
fi
rm -f "$BASELINE_CANDIDATE" 2>/dev/null || true

# Stage 2: spend only the remaining bounded budget on a stronger candidate.
if [ -z "${LLM_MODEL_CONFIG:-}" ]; then
  echo "[beta-v2] embedding config unavailable; keeping $SELECTED" >&2
  exit 0
fi
if [ "$N_CASES" -gt "${BETA_V2_EXPERT_MAX_CASES:-300}" ]; then
  echo "[beta-v2] dataset too large for expert; keeping $SELECTED" >&2
  exit 0
fi

NOW_SEC=$(monotonic_sec)
REMAINING=$((DEADLINE_SEC - NOW_SEC))
if [ "$REMAINING" -lt 5 ]; then
  echo "[beta-v2] insufficient expert budget (${REMAINING}s); keeping $SELECTED" >&2
  exit 0
fi
if [ "$DESIRED_EXPERT_LIMIT" -lt "$REMAINING" ]; then EXPERT_LIMIT=$DESIRED_EXPERT_LIMIT; else EXPERT_LIMIT=$REMAINING; fi

if [ "$N_CASES" -le "${BETA_V2_MULTIVIEW_MAX_CASES:-160}" ] && [ -x "$MULTIVIEW_BIN" ]; then
  EXPERT_BIN=$MULTIVIEW_BIN
  EXPERT_NAME=multiview
else
  EXPERT_BIN=$FULL_BIN
  EXPERT_NAME=calibrated_dual
fi
echo "[beta-v2] trying $EXPERT_NAME candidate (limit=${EXPERT_LIMIT}s), baseline remains published" >&2
if run_candidate "$EXPERT_LIMIT" "$EXPERT_CANDIDATE" "$EXPERT_BIN" "$@" --output "$EXPERT_CANDIDATE"; then
  publish_candidate "$EXPERT_CANDIDATE" "$EXPERT_NAME" || true
else
  echo "[beta-v2] expert failed or timed out; keeping $SELECTED" >&2
fi
rm -f "$EXPERT_CANDIDATE" 2>/dev/null || true

TOTAL_SEC=$(( $(monotonic_sec) - START_SEC ))
echo "[beta-v2] completed selected=$SELECTED total=${TOTAL_SEC}s" >&2
exit 0
