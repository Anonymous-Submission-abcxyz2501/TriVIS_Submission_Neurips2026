"""Build 4 signer-subset test CSVs in 3-view B-format.

Each subset has ~300 sentences, one per signer (group1–group4).

  group1 (300 rows):  first 300 from existing e2 test.csv (already 3-view)
  group4 (≤300 rows): same sentences as group1 subset, matched by Sign_sentence text
  group2 (300 rows):  ID_sentence 3001–3300 (seen by group1 during training)
  group3 (300 rows):  ID_sentence 5701–6000 (seen by group1 + group2 during training)

Output: four CSVs in the three_view e2 directory.
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

_data_root_env = os.environ.get("DATA_ROOT", "").strip()
if not _data_root_env:
    raise SystemExit(
        "Set DATA_ROOT to the directory containing CSLR_dataset/Full_dataset/...\n"
        "Example: DATA_ROOT=/data/trivis python build_signer_subset_csvs.py"
    )
DATA_ROOT = Path(_data_root_env).expanduser().resolve()
if not DATA_ROOT.is_dir():
    raise SystemExit(f"DATA_ROOT does not exist or is not a directory: {DATA_ROOT}")
FULL_CSV  = DATA_ROOT / "CSLR_dataset/Full_dataset/Full_seperate/Dataset_full_lab.csv"
E2_DIR    = DATA_ROOT / "CSLR_dataset/Full_dataset/Splits_dataset_1_2/three_view/e2_signer_independent_seen_sentence_lab"
E2_TEST   = E2_DIR / "test.csv"

OUT_FIELDS = [
    "Sentence_video_path_front", "Sentence_video_path_right", "Sentence_video_path_left",
    "ID_video", "ID_sentence", "group", "view", "scene",
    "video_belong_id", "Sentence", "Sign_sentence", "Category",
]

N = 300  # rows per subset


def load_full() -> dict:
    """Index Dataset_full_lab.csv: (group, ID_sentence) -> {view -> row}."""
    idx = defaultdict(dict)
    with open(FULL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            idx[(r["group"], r["ID_sentence"])][r["view"].strip().lower()] = r
    return idx


def to_3view(entry: dict) -> dict | None:
    """Combine front/left/right view rows into one 3-view record. Returns None if any view missing."""
    front = entry.get("front")
    left  = entry.get("left")
    right = entry.get("right")
    if not (front and left and right):
        return None
    base = front
    return {
        "Sentence_video_path_front": front["Sentence_video_path"],
        "Sentence_video_path_right": right["Sentence_video_path"],
        "Sentence_video_path_left":  left["Sentence_video_path"],
        "ID_video":        base["ID_video"],
        "ID_sentence":     base["ID_sentence"],
        "group":           base["group"],
        "view":            "multi",
        "scene":           base.get("scene", ""),
        "video_belong_id": base.get("video_belong_id", ""),
        "Sentence":        base.get("Sentence", ""),
        "Sign_sentence":   base.get("Sign_sentence", ""),
        "Category":        base.get("Category", ""),
    }


def write_csv(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for rec in records:
            w.writerow({k: rec.get(k, "") for k in OUT_FIELDS})
    print(f"  wrote {len(records)} rows → {path.name}")


def main() -> None:
    idx = load_full()
    print(f"Loaded {len(idx)} (group, ID_sentence) entries from {FULL_CSV.name}")

    # ── group1: first N rows from existing test.csv (already 3-view) ──────────
    with open(E2_TEST, encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    g1_rows = test_rows[:N]
    write_csv(E2_DIR / "test_subset_group1.csv", g1_rows)

    # ── group4: match group1 sentences by Sign_sentence text ──────────────────
    g1_texts = {r["Sign_sentence"] for r in g1_rows}
    g4_sent_ids = sorted(
        {sid for (grp, sid), views in idx.items() if grp == "group4"},
        key=int,
    )
    g4_rows = []
    for sid in g4_sent_ids:
        rec = to_3view(idx[("group4", sid)])
        if rec and rec["Sign_sentence"] in g1_texts:
            g4_rows.append(rec)
        if len(g4_rows) == N:
            break
    write_csv(E2_DIR / "test_subset_group4.csv", g4_rows)

    # ── group2: ID_sentence 3001–3300 (seen by group1 in training) ────────────
    g2_rows = []
    for sid in range(3001, 3001 + N * 5):  # scan enough range
        rec = to_3view(idx.get(("group2", str(sid)), {}))
        if rec:
            g2_rows.append(rec)
        if len(g2_rows) == N:
            break
    write_csv(E2_DIR / "test_subset_group2.csv", g2_rows)

    # ── group3: ID_sentence 5701–6000 (seen by group1 + group2 in training) ───
    g3_rows = []
    for sid in range(5701, 5701 + N * 5):
        rec = to_3view(idx.get(("group3", str(sid)), {}))
        if rec:
            g3_rows.append(rec)
        if len(g3_rows) == N:
            break
    write_csv(E2_DIR / "test_subset_group3.csv", g3_rows)

    print("\nDone. Summary:")
    for name in ["test_subset_group1.csv", "test_subset_group2.csv",
                 "test_subset_group3.csv", "test_subset_group4.csv"]:
        path = E2_DIR / name
        with open(path) as f:
            n = sum(1 for _ in f) - 1
        print(f"  {name}: {n} rows")


if __name__ == "__main__":
    main()
