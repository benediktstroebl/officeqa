#!/usr/bin/env python3
"""Compare fork vs Harbor OfficeQA scores."""

import argparse
import json
import statistics
from pathlib import Path


def load_scores(base_dir: Path, n_trials: int) -> dict[int, dict[str, float]]:
    """Load {trial: {task_id: score}} from trial_N/scores.jsonl files."""
    out: dict[int, dict[str, float]] = {}
    for trial in range(1, n_trials + 1):
        path = base_dir / f"trial_{trial}" / "scores.jsonl"
        if not path.exists():
            print(f"  WARNING: missing {path}")
            continue
        scores: dict[str, float] = {}
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            scores[entry["task_id"]] = entry["score"]
        out[trial] = scores
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fork-dir", type=Path, required=True)
    p.add_argument("--harbor-dir", type=Path, required=True)
    p.add_argument("--trials", type=int, default=3)
    args = p.parse_args()

    fork = load_scores(args.fork_dir, args.trials)
    harbor = load_scores(args.harbor_dir, args.trials)

    # Per-trial accuracy
    for label, data in [("Fork", fork), ("Harbor", harbor)]:
        accs = []
        for trial in sorted(data):
            s = data[trial]
            c = sum(1 for v in s.values() if v > 0)
            acc = c / len(s) * 100
            accs.append(acc)
            print(f"  {label} trial {trial}: {c}/{len(s)} = {acc:.2f}%")
        if accs:
            print(
                f"  {label} mean: {statistics.mean(accs):.2f}% +/- "
                f"{statistics.stdev(accs) if len(accs) > 1 else 0:.2f}%"
            )
        print()

    # Agreement by majority vote
    all_ids = set()
    for d in [fork, harbor]:
        for s in d.values():
            all_ids.update(s)

    agree = 0
    for tid in sorted(all_ids):

        def majority(data):
            votes = [1 if data[t].get(tid, 0) > 0 else 0 for t in data]
            return sum(votes) > len(votes) / 2

        if majority(fork) == majority(harbor):
            agree += 1

    print(f"Agreement: {agree}/{len(all_ids)} = {agree / len(all_ids) * 100:.1f}%")


if __name__ == "__main__":
    main()
