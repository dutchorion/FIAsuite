"""Measure the local AI detectors, alone and as ensembles, on a labelled folder (run in .venv):

    .venv/Scripts/python tools/bench_ai_detectors.py bench/openfake

The folder holds images and an index.json list of {"file", "label": "real" | "fake", "model"}. Scores are appended
to scores_partial.jsonl as they are computed, so a stopped run resumes where it left off. Writes scores.csv
next to it and prints AUC, generated images rated above 50 %, photographs rated above 50 %, and per-generator
detection. The ensembles are the mean of log-odds the app uses, with equal weights.
"""
import csv
import json
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fiasuite.analyses import ai            # noqa: E402
from fiasuite.core import Evidence          # noqa: E402

ai.use_gpu = lambda: False                  # measure what every installation gets


def auc(scores: np.ndarray, fake: np.ndarray) -> float:
    ranks = rankdata(scores)
    n_fake, n_real = fake.sum(), (~fake).sum()
    return float((ranks[fake].sum() - n_fake * (n_fake + 1) / 2) / (n_fake * n_real))


def ensemble(columns: list[np.ndarray]) -> np.ndarray:
    p = np.clip(np.stack(columns), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.log(p / (1 - p)).mean(axis=0)))


def main(folder: Path) -> None:
    index = json.loads((folder / "index.json").read_text())
    detectors = [e for e in ai.load_registry() if e.get("kind", "classifier") == "classifier"]
    partial = folder / "scores_partial.jsonl"
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            row = json.loads(line)
            done[row["file"]] = row
    scores, seconds = defaultdict(list), defaultdict(float)
    with open(partial, "a") as log:
        for n, item in enumerate(index, 1):
            row = done.get(item["file"])
            if row is None or any(e["id"] not in row["scores"] for e in detectors):
                ev = Evidence(str(folder / item["file"]))
                row = {"file": item["file"], "scores": {}, "seconds": {}}
                for entry in detectors:
                    t = time.perf_counter()
                    row["scores"][entry["id"]] = ai.classify(ev, entry, max_tiles=1)[0]   # the image score needs no map
                    row["seconds"][entry["id"]] = time.perf_counter() - t
                log.write(json.dumps(row) + "\n")
                log.flush()
            for entry in detectors:
                scores[entry["id"]].append(row["scores"][entry["id"]])
                seconds[entry["id"]] += row["seconds"][entry["id"]]
            if n % 25 == 0:
                print(f"  {n} / {len(index)}", flush=True)
    with open(folder / "scores.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "label", "model", *scores])
        for i, item in enumerate(index):
            writer.writerow([item["file"], item["label"], item["model"], *(f"{scores[k][i]:.6f}" for k in scores)])

    fake = np.array([item["label"] == "fake" for item in index])
    columns = {k: np.array(v) for k, v in scores.items()}
    for size in range(2, len(columns) + 1):
        for combo in combinations(list(scores), size):
            columns[" + ".join(combo)] = ensemble([columns[k] for k in combo])
    print(f"\n{fake.sum()} generated images, {(~fake).sum()} photographs")
    print(f"{'detector':45s} {'AUC':>6s} {'generated > 50 %':>18s} {'photos > 50 %':>15s} {'s / image':>10s}")
    for name, col in columns.items():
        speed = f"{seconds[name] / len(index):.2f}" if name in seconds else ""
        print(f"{name:45s} {auc(col, fake):6.3f} {int((col[fake] > 0.5).sum()):>11d} / {fake.sum():<4d} "
              f"{int((col[~fake] > 0.5).sum()):>8d} / {(~fake).sum():<4d} {speed:>10s}")
    models = sorted({item["model"] for item in index if item["label"] == "fake"},
                    key=lambda m: -sum(item["model"] == m for item in index))
    print("\nper generator, rated above 50 %:")
    shown = [k for k in columns if " + " not in k] + [k for k in columns if k.count(" + ") == len(scores) - 2 or
                                                       k.count(" + ") == len(scores) - 1]
    for m in models:
        rows = np.array([item["model"] == m for item in index])
        print(f"  {m:22s} " + "  ".join(f"{k.split(' + ')[0] if ' + ' not in k else k}: "
                                        f"{int((columns[k][rows] > 0.5).sum())}/{rows.sum()}" for k in shown))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
