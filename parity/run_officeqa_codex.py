#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["python-dotenv", "tqdm"]
# ///
"""Run codex on OfficeQA full-corpus tasks outside Harbor."""

import argparse
import asyncio
import csv
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from instruction_template import TEMPLATE


def load_completed(scores_path: Path) -> set[str]:
    """Return task_ids already in scores.jsonl."""
    done = set()
    if scores_path.exists():
        for line in scores_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["task_id"])
    return done


AGENT_TIMEOUT_SEC = 1800.0  # matches [agent] timeout_sec in task-full-corpus.toml


async def run_task(row, corpus_dir, model, traces_dir, score_answer, sem):
    uid = row["uid"].strip()
    task_id = f"officeqa-{uid.lower()}"

    work_dir = Path(f"/tmp/oqa/{uid.lower()}")
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    corpus_link = work_dir / "corpus"
    corpus_link.symlink_to(corpus_dir)
    answer_path = work_dir / "answer.txt"
    codex_home = work_dir / ".codex_home"
    codex_home.mkdir(parents=True, exist_ok=True)

    # Clean Python 3.12 venv via uv — prevents dependency spillover between tasks
    venv_dir = work_dir / ".venv"
    venv_proc = await asyncio.create_subprocess_exec(
        "uv",
        "venv",
        "--python",
        "3.12",
        str(venv_dir),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await venv_proc.wait()

    instruction = TEMPLATE.format(
        question=row["question"].strip(),
        corpus_dir=str(corpus_link),
        answer_path=str(answer_path),
    )

    env = os.environ.copy()
    env["CODEX_API_KEY"] = os.environ["OPENAI_API_KEY"]
    env["CODEX_HOME"] = str(codex_home)
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = f"{venv_dir / 'bin'}:{env.get('PATH', '')}"

    async with sem:
        proc = await asyncio.create_subprocess_exec(
            "codex",
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--model",
            model,
            "--config",
            "model_reasoning_effort=high",
            "--json",
            "--",
            instruction,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(work_dir),
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=AGENT_TIMEOUT_SEC
            )
        except asyncio.TimeoutError:
            proc.kill()
            stdout, stderr = await proc.communicate()
            print(f"  TIMEOUT: {task_id} killed after {AGENT_TIMEOUT_SEC}s")

    (traces_dir / f"{task_id}.jsonl").write_bytes(stdout)
    if stdout:
        (traces_dir / f"{task_id}.stdout.log").write_bytes(stdout)
    if stderr:
        (traces_dir / f"{task_id}.stderr.log").write_bytes(stderr)

    predicted = answer_path.read_text().strip() if answer_path.exists() else ""
    expected = row["answer"].strip()
    score = 0.0
    if predicted:
        try:
            score = score_answer(expected, predicted, 0.01)
        except Exception:
            pass

    # Full cleanup — venv, config, answer all deleted
    shutil.rmtree(work_dir, ignore_errors=True)

    return {
        "task_id": task_id,
        "uid": uid,
        "score": score,
        "predicted": predicted,
        "expected": expected,
    }


async def run_trial(
    rows, corpus_dir, model, output_dir, trial, n_trials, concurrent, score_answer
):
    traces_dir = output_dir / f"trial_{trial}" / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / f"trial_{trial}" / "scores.jsonl"

    completed = load_completed(scores_path)
    pending = [
        r for r in rows if f"officeqa-{r['uid'].strip().lower()}" not in completed
    ]

    if not pending:
        print(
            f"\n  Trial {trial}/{n_trials}: all {len(rows)} tasks already completed, skipping"
        )
        return
    if completed:
        print(
            f"\n  Resuming trial {trial}/{n_trials}: {len(completed)} done, {len(pending)} remaining"
        )

    sem = asyncio.Semaphore(concurrent)
    tasks = [
        run_task(row, corpus_dir, model, traces_dir, score_answer, sem)
        for row in pending
    ]

    correct = 0
    if scores_path.exists():
        for line in scores_path.read_text().splitlines():
            if line.strip() and json.loads(line)["score"] > 0:
                correct += 1

    pbar = tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"Trial {trial}/{n_trials}",
        unit="task",
    )
    for coro in pbar:
        result = await coro
        if result["score"] > 0:
            correct += 1
        pbar.set_postfix(acc=f"{correct}/{len(completed) + pbar.n + 1}")
        with scores_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "task_id": result["task_id"],
                        "uid": result["uid"],
                        "score": result["score"],
                        "predicted": result["predicted"],
                        "expected": result["expected"],
                    }
                )
                + "\n"
            )

    print(f"  Trial {trial}: {correct}/{len(rows)} = {correct / len(rows) * 100:.1f}%")


async def amain():
    load_dotenv(override=True)
    if not shutil.which("uv"):
        sys.exit(
            "ERROR: uv not found on PATH. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
        )
    if shutil.which("pdftotext"):
        sys.exit(
            "ERROR: pdftotext found on host but not in Harbor image. Run: brew uninstall poppler"
        )

    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--reward-py", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default="gpt-5-mini")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrent", type=int, default=8)
    args = p.parse_args()

    corpus_dir = args.corpus_dir.resolve()

    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit > 0:
        rows = rows[: args.limit]

    spec = importlib.util.spec_from_file_location("reward", args.reward_py)
    reward_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward_mod)
    score_answer = reward_mod.score_answer

    index_path = corpus_dir / "index.txt"
    if not index_path.exists():
        txt_files = sorted(corpus_dir.glob("*.txt"))
        index_path.write_text("\n".join(str(f) for f in txt_files) + "\n")

    for trial in range(1, args.trials + 1):
        await run_trial(
            rows,
            corpus_dir,
            args.model,
            args.output_dir,
            trial,
            args.trials,
            args.concurrent,
            score_answer,
        )


if __name__ == "__main__":
    asyncio.run(amain())
