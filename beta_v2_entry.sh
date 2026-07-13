#!/usr/bin/env sh
# Candidate entry point: public baseline-first runs can spend up to 22 seconds
# on the expert because a valid deterministic result is already published.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export BETA_V2_EXPERT_LIMIT="${BETA_V2_EXPERT_LIMIT:-22}"
exec "$SCRIPT_DIR/beta_v2_router.sh" "$@"
