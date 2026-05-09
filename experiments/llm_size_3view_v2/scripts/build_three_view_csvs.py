"""Build B-format (combined 3-path-col) CSVs from A-format CSVs.

A's train.csv is exploded (1 row × 1 view, all 3 views present per sentence),
but A's val.csv / test.csv are front-only (1 row × front view per sentence).
For val/test, we look up the missing left/right paths from `Dataset_full_lab.csv`
keyed by (group, ID_video).

A = Splits_dataset_1_2/three_view_as_one/e1_sentence_progressive_lab/{train,val,test}.csv
B = Splits_dataset_1_2/three_view/e1_sentence_progressive_lab/{train,val,test}.csv

Re-split rule (per user decision):
- B/val.csv  = combined A/val   (1,200 rows × 3-path-cols)
- B/test.csv = combined A/test  (1,200 rows × 3-path-cols)
- B/train.csv = combined A/train  +  random 50% val sentences  +  random 50% test sentences
  (overlap allowed — val/test rows ALSO appear in train)

Backups B/{name}.csv → B/{name}_old.csv before overwrite.
"""

import csv
import os
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# DATA_ROOT must point to the parent of `CSLR_dataset/` (e.g. the directory you
# downloaded the TriViS dataset into from Hugging Face).
_data_root_env = os.environ.get("DATA_ROOT", "").strip()
if not _data_root_env:
    raise SystemExit(
        "Set DATA_ROOT to the directory containing CSLR_dataset/Full_dataset/...\n"
        "Example: DATA_ROOT=/data/trivis python build_three_view_csvs.py"
    )
DATA_ROOT = Path(_data_root_env).expanduser().resolve()
if not DATA_ROOT.is_dir():
    raise SystemExit(f"DATA_ROOT does not exist or is not a directory: {DATA_ROOT}")
A_DIR = DATA_ROOT / "CSLR_dataset/Full_dataset/Splits_dataset_1_2/three_view_as_one/e1_sentence_progressive_lab"
B_DIR = DATA_ROOT / "CSLR_dataset/Full_dataset/Splits_dataset_1_2/three_view/e1_sentence_progressive_lab"
FULL_CSV = DATA_ROOT / "CSLR_dataset/Full_dataset/Full_seperate/Dataset_full_lab.csv"

OUT_FIELDS = [
    "Sentence_video_path_front", "Sentence_video_path_right", "Sentence_video_path_left",
    "ID_video", "ID_sentence", "group", "view", "scene",
    "video_belong_id", "Sentence", "Sign_sentence", "Category",
]

SAMPLE_FRACTION = 0.5
SEED = 42


def load_a(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_full_index(full_rows: List[Dict]) -> Dict[Tuple[str, str], Dict[str, Dict]]:
    """Index Dataset_full_lab.csv: (group, ID_video) → {view → row}."""
    idx: Dict[Tuple[str, str], Dict[str, Dict]] = defaultdict(dict)
    for r in full_rows:
        idx[(r["group"], r["ID_video"])][r["view"].strip().lower()] = r
    return idx


def combine_per_sentence(
    rows: List[Dict],
    full_index: Dict[Tuple[str, str], Dict[str, Dict]],
) -> "OrderedDict[Tuple[str, str], Dict]":
    """Group A rows by (group, ID_video) → combined record with 3 path columns.

    For each (group, ID_video) seen in `rows`, fill missing view paths by lookup
    in `full_index`. Preserves first-seen order. Skips records still missing
    any view after lookup.
    """
    by_key: "OrderedDict[Tuple[str, str], Dict]" = OrderedDict()
    col_map = {
        "front": "Sentence_video_path_front",
        "left": "Sentence_video_path_left",
        "right": "Sentence_video_path_right",
    }
    for r in rows:
        key = (r["group"], r["ID_video"])
        if key not in by_key:
            by_key[key] = {
                "Sentence_video_path_front": "",
                "Sentence_video_path_right": "",
                "Sentence_video_path_left": "",
                "ID_video": r["ID_video"],
                "ID_sentence": r["ID_sentence"],
                "group": r["group"],
                "view": "multi",
                "scene": r.get("scene", ""),
                "video_belong_id": r.get("video_belong_id", ""),
                "Sentence": r.get("Sentence", ""),
                "Sign_sentence": r.get("Sign_sentence", ""),
                "Category": r.get("Category", ""),
            }
        view = r.get("view", "").strip().lower()
        if view in col_map:
            by_key[key][col_map[view]] = r["Sentence_video_path"]

    # Fill missing views from full_index
    n_filled = 0
    for key, rec in by_key.items():
        per_view = full_index.get(key, {})
        for v, col in col_map.items():
            if not rec[col] and v in per_view:
                rec[col] = per_view[v]["Sentence_video_path"]
                n_filled += 1
    if n_filled:
        print(f"[combine] filled {n_filled} missing view paths from Dataset_full_lab.csv")

    complete: "OrderedDict[Tuple[str, str], Dict]" = OrderedDict()
    skipped = 0
    for k, rec in by_key.items():
        if (rec["Sentence_video_path_front"]
                and rec["Sentence_video_path_left"]
                and rec["Sentence_video_path_right"]):
            complete[k] = rec
        else:
            skipped += 1
    if skipped:
        print(f"[combine] skipped {skipped} records still missing one or more views after lookup")
    return complete


def write_b(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for rec in records:
            w.writerow({k: rec.get(k, "") for k in OUT_FIELDS})


def backup(path: Path) -> None:
    if path.exists():
        bak = path.with_name(path.stem + "_old.csv")
        if bak.exists():
            bak.unlink()
        path.rename(bak)
        print(f"[backup] {path.name} → {bak.name}")


def main() -> None:
    rng = random.Random(SEED)

    print(f"[load] reading A from {A_DIR}")
    a_train = load_a(A_DIR / "train.csv")
    a_val = load_a(A_DIR / "val.csv")
    a_test = load_a(A_DIR / "test.csv")
    print(f"  A/train: {len(a_train)} rows | A/val: {len(a_val)} rows | A/test: {len(a_test)} rows")

    print(f"[load] indexing {FULL_CSV.name}")
    full_rows = load_a(FULL_CSV)
    full_index = build_full_index(full_rows)
    print(f"  {len(full_rows)} rows → {len(full_index)} unique (group, ID_video) keys")

    train_combined = list(combine_per_sentence(a_train, full_index).values())
    val_combined = list(combine_per_sentence(a_val, full_index).values())
    test_combined = list(combine_per_sentence(a_test, full_index).values())
    print(f"[combine] train: {len(train_combined)} | val: {len(val_combined)} | test: {len(test_combined)}")

    n_val_sample = int(round(len(val_combined) * SAMPLE_FRACTION))
    n_test_sample = int(round(len(test_combined) * SAMPLE_FRACTION))
    val_sampled = rng.sample(val_combined, n_val_sample)
    test_sampled = rng.sample(test_combined, n_test_sample)
    print(f"[sample 50%] val→train: {n_val_sample} | test→train: {n_test_sample} (seed={SEED})")

    train_final = train_combined + val_sampled + test_sampled
    print(f"[final] B/train: {len(train_final)} | B/val: {len(val_combined)} | B/test: {len(test_combined)}")

    print(f"[write] backup + write B at {B_DIR}")
    backup(B_DIR / "train.csv")
    backup(B_DIR / "val.csv")
    backup(B_DIR / "test.csv")
    write_b(B_DIR / "train.csv", train_final)
    write_b(B_DIR / "val.csv", val_combined)
    write_b(B_DIR / "test.csv", test_combined)

    print("\n[stats]")
    print(f"  B/train.csv : {len(train_final)} rows  (= {len(train_combined)} + {n_val_sample} + {n_test_sample})")
    print(f"  B/val.csv   : {len(val_combined)} rows  (unchanged)")
    print(f"  B/test.csv  : {len(test_combined)} rows  (unchanged)")
    print("[done]")


if __name__ == "__main__":
    main()
