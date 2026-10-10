"""Write the fixed T7 benchmark: 1,000 seeded MMLU test questions -> data/mmlu_1k.jsonl.

    python scripts/t7_fetch_mmlu.py [parquet path or URL]

Run once on a machine with internet; the output is committed so compute nodes never
need the network.
"""
import json
import sys
from pathlib import Path

import pandas as pd

URL = "https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet"
OUT = Path(__file__).resolve().parent.parent / "data" / "mmlu_1k.jsonl"


def main(source: str = URL, n: int = 1000, seed: int = 0) -> None:
    # `source` may be a local copy (e.g. fetched with curl when Python's SSL store is
    # missing, as on a python.org macOS install); the sample depends only on the rows.
    df = pd.read_parquet(source)
    sample = df.sample(n=n, random_state=seed).reset_index(drop=True)
    with open(OUT, "w") as f:
        for i, r in sample.iterrows():
            f.write(json.dumps({"id": i, "subject": r["subject"], "question": r["question"],
                                "choices": [str(c) for c in r["choices"]], "answer": int(r["answer"])}) + "\n")
    print(f"{n} questions from {len(df)} ({sample['subject'].nunique()} subjects) -> {OUT}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else URL, *map(int, args[1:]))
