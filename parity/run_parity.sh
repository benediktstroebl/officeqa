#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

OFFICEQA=..

# Claude Code: 246 tasks x 3 trials
uv run run_officeqa_claude_code.py \
    --csv "$OFFICEQA/officeqa_full.csv" \
    --corpus-dir "$OFFICEQA/treasury_bulletins_parsed/transformed" \
    --reward-py "$OFFICEQA/reward.py" \
    --output-dir results/claude_code \
    --trials 3

# Codex: 246 tasks x 3 trials
uv run run_officeqa_codex.py \
    --csv "$OFFICEQA/officeqa_full.csv" \
    --corpus-dir "$OFFICEQA/treasury_bulletins_parsed/transformed" \
    --reward-py "$OFFICEQA/reward.py" \
    --output-dir results/codex \
    --trials 3
