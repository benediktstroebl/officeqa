"""Instruction template matching Harbor's instruction-full-corpus.md."""

TEMPLATE = """{question}

## Available Resources

You have access to the full U.S. Treasury Bulletin corpus at `{corpus_dir}`. This directory contains 697 parsed Treasury Bulletin text files (Markdown with tables), one per monthly bulletin issue.

**Corpus location:** `{corpus_dir}`
**File naming convention:** `treasury_bulletin_YYYY_MM.txt` (e.g., `treasury_bulletin_1941_01.txt`)
**File listing:** `{corpus_dir}/index.txt`

You must search through these files to find the relevant information to answer the question.

## Output

Write your final answer to `{answer_path}`. Numerical answers should be precise (scoring uses 1% tolerance)."""
