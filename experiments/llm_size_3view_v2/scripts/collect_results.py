"""Aggregate test BLEU/ROUGE across LLM size sweep runs.

Reads metrics.csv produced by Lightning's CSVLogger / TensorBoardLogger fallback in
each run dir, picks the row with the test/* metrics, and writes a markdown table
to results/summary.md and CSV to results/summary.csv.

Run after `train_3view_attn.sh` for each size.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# REPO_ROOT defaults to three levels up from this script (scripts/ → llm_size_3view_v2/
# → experiments/ → repo root), matching the layout used by train_3view_attn.sh.
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
LOG_ROOT = Path(os.environ.get("LOG_ROOT", REPO_ROOT / "logs"))
RESULT_DIR = REPO_ROOT / "experiments/llm_size_3view_v2/results"

LLM_SIZES = ["videollama3_2b", "qwen2_5_3b", "videollama3_7b"]
LLM_LABELS = {
    "videollama3_2b": "VideoLLaMA3-2B (1.5B, hidden=1536)",
    "qwen2_5_3b":     "Qwen2.5-3B-Instruct (hidden=2048)",
    "videollama3_7b": "VideoLLaMA3-7B (hidden=3584)",
}
METRICS = ["test/bleu1", "test/bleu2", "test/bleu3", "test/bleu4", "test/rougeL"]


def find_latest_metrics_csv(size: str) -> Optional[Path]:
    run_dir = LOG_ROOT / f"e1_3view_attn_{size}"
    if not run_dir.exists():
        return None
    candidates = list(run_dir.rglob("metrics.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_test_metrics(metrics_csv: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with open(metrics_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in METRICS:
                v = row.get(k, "")
                if v not in ("", None):
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
    return out


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    for size in LLM_SIZES:
        m_csv = find_latest_metrics_csv(size)
        if m_csv is None:
            print(f"[{size}] no metrics.csv found, skipping")
            rows.append({"size": size, "label": LLM_LABELS[size], **{k: "n/a" for k in METRICS}})
            continue
        metrics = extract_test_metrics(m_csv)
        rows.append({
            "size": size,
            "label": LLM_LABELS[size],
            **{k: f"{metrics.get(k, float('nan')):.2f}" if k in metrics else "n/a" for k in METRICS},
        })
        print(f"[{size}] {m_csv}: {metrics}")

    csv_out = RESULT_DIR / "summary.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["size", "label", *METRICS])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[write] {csv_out}")

    md_out = RESULT_DIR / "summary.md"
    headers = ["LLM size", *METRICS]
    lines = [
        "# LLM Size Sweep — SpaMo_3View Attn-Pool Fusion",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        row_cells = [r["label"], *[r[k] for k in METRICS]]
        lines.append("| " + " | ".join(row_cells) + " |")
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {md_out}")


if __name__ == "__main__":
    main()
