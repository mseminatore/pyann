"""Merge bench_blas.py JSON lines into a markdown table (stdin -> stdout)."""

import json
import sys

rows = [json.loads(line) for line in sys.stdin if line.startswith("{")]
variants = list(dict.fromkeys(r["variant"] for r in rows))
workloads = list(dict.fromkeys(r["workload"] for r in rows))
by = {(r["variant"], r["workload"]): r for r in rows}

for metric, title in (("fit_s", "fit (s)"), ("predict_s", "predict (s)")):
    print(f"\n**{title}** — lower is better\n")
    print("| workload | " + " | ".join(variants) + " |")
    print("|---|" + "---|" * len(variants))
    for w in workloads:
        cells = [f"{by[(v, w)][metric]:.3f}" if (v, w) in by else "-" for v in variants]
        print(f"| {w} | " + " | ".join(cells) + " |")
