#!/usr/bin/env bash
# agent2model — the whole pipeline, offline and free.
#
# Every step here runs with NO API key, NO GPU, and spends $0: it shows the
# procedure, the dataset shape, and the eval-report layout end to end. It's the
# demo to record (asciinema) and the "try it in 60 seconds" block for the README.
#
#   asciinema rec demo.cast -c "bash scripts/demo.sh"
#
# Real training/eval need a key + GPU (see the README); this is the free preview.
set -euo pipefail

# Slow the echoed commands down a touch so a screen recording is readable.
PAUSE="${DEMO_PAUSE:-1.2}"
run() { printf '\n\033[1;32m$ %s\033[0m\n' "$*"; sleep "$PAUSE"; eval "$*"; sleep "$PAUSE"; }

WORK="$(mktemp -d)"
cd "$WORK"
printf '\033[1;36m# agent2model — offline demo (no key, no GPU, $0)\033[0m\n'

# 1. Copy a bundled example into the working dir.
run "agent2model init travel_booking"

# 2. Compile the procedure into the IR (+ a Mermaid diagram), free & offline.
run "agent2model compile travel_booking/flowchart.yaml --out build/travel"

# 3. SEE the procedure you're about to compile into weights.
run "agent2model show build/travel --format summary"
run "agent2model show build/travel | head -20"

# 4. Preview the training dataset shape — templated, no API calls.
run "agent2model generate build/travel --n 3 --mock"
run "head -1 build/travel/dataset.jsonl | python -m json.tool | head -12"

# 5. Render the eval report layout from illustrative scores — no key, no GPU.
run "agent2model eval build/travel --demo --n 40"
run "ls -la build/travel/eval_report.*"

printf '\n\033[1;36m# That was the whole shape, for free. Real run: add a key + GPU (see README).\033[0m\n'
