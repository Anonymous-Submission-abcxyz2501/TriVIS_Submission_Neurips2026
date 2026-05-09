"""Build 3-view UNSEEN test CSV (mixed):
    1000 group3 sentences not in split-1-2 train
  + random 50% of val.csv
  + random 50% of test.csv

Output : Splits_dataset_1_2/three_view/e1_sentence_progressive_lab/test_unseen.csv
         (B-format: same schema as val.csv / test.csv)

Sampling:
- Group3 unseen: filter master to group=='group3', combine 3 views per ID_video,
  keep only ID_sentence ∉ train, then random.Random(42).sample(..., 1000).
- Val/Test 50%: random.Random(42).sample(range(N), N//2) on already-loaded rows.
- Same RNG seed=42 advances across the 3 sampling steps in order
  (group3 → val → test) for reproducibility.
"""

import csv
import os
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_data_root_env = os.environ.get("DATA_ROOT", "").strip()
if not _data_root_env:
    raise SystemExit(
        "Set DATA_ROOT to the directory containing CSLR_dataset/Full_dataset/...\n"
        "Example: DATA_ROOT=/data/trivis python build_unseen_test_csv.py"
    )
DATA_ROOT = Path(_data_root_env).expanduser().resolve()
if not DATA_ROOT.is_dir():
    raise SystemExit(f"DATA_ROOT does not exist or is not a directory: {DATA_ROOT}")
FULL_CSV = DATA_ROOT / "CSLR_dataset/Full_dataset/Full_seperate/Dataset_full_lab.csv"
B_DIR = DATA_ROOT / "CSLR_dataset/Full_dataset/Splits_dataset_1_2/three_view/e1_sentence_progressive_lab"
TRAIN_CSV = B_DIR / "train.csv"
VAL_CSV = B_DIR / "val.csv"
TEST_CSV = B_DIR / "test.csv"
OUT_CSV = B_DIR / "test_unseen.csv"

OUT_FIELDS = [
    "Sentence_video_path_front", "Sentence_video_path_right", "Sentence_video_path_left",
    "ID_video", "ID_sentence", "group", "view", "scene",
    "video_belong_id", "Sentence", "Sign_sentence", "Category",
]

TARGET_GROUP = "group3"
N_SENTENCES = 1000
VAL_FRAC = 0.5
TEST_FRAC = 0.5
SEED = 42

VIEW_TO_COL = {
    "front": "Sentence_video_path_front",
    "left":  "Sentence_video_path_left",
    "right": "Sentence_video_path_right",
}


def load_csv(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def combine_group3_records(rows: List[Dict]) -> "OrderedDict[Tuple[str, str], Dict]":
    """Group A rows (group3 only) by (group, ID_video) -> combined 3-view record.

    Skips records that don't have all 3 views populated.
    """
    by_key: "OrderedDict[Tuple[str, str], Dict]" = OrderedDict()
    for r in rows:
        if r.get("group") != TARGET_GROUP:
            continue
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
        if view in VIEW_TO_COL:
            by_key[key][VIEW_TO_COL[view]] = r["Sentence_video_path"]

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
        print(f"[combine] skipped {skipped} group3 records missing >=1 view")
    return complete


def main() -> None:
    rng = random.Random(SEED)

    print(f"[load] master  : {FULL_CSV}")
    full_rows = load_csv(FULL_CSV)
    print(f"  total rows: {len(full_rows)}")

    print(f"[load] seen   : {TRAIN_CSV}")
    train_rows = load_csv(TRAIN_CSV)
    seen_sentence_ids = {r["ID_sentence"] for r in train_rows}
    print(f"  unique seen ID_sentence in train: {len(seen_sentence_ids)}")

    print(f"[combine] group3 records ({TARGET_GROUP})")
    combined = combine_group3_records(full_rows)
    print(f"  combined records (3-view complete): {len(combined)}")
    print(f"  unique ID_sentence: {len({r['ID_sentence'] for r in combined.values()})}")

    # Filter: only records whose ID_sentence not in seen set
    unseen_records = [r for r in combined.values() if r["ID_sentence"] not in seen_sentence_ids]
    unseen_sentence_ids = sorted({r["ID_sentence"] for r in unseen_records})
    print(f"[filter] unseen records: {len(unseen_records)} | unique unseen sentences: {len(unseen_sentence_ids)}")

    if len(unseen_sentence_ids) < N_SENTENCES:
        raise RuntimeError(
            f"Not enough unseen sentences: have {len(unseen_sentence_ids)}, need {N_SENTENCES}"
        )

    chosen_ids = set(rng.sample(unseen_sentence_ids, N_SENTENCES))
    print(f"[sample] picked {len(chosen_ids)} unseen sentences (seed={SEED})")

    # Pick 1 record per chosen sentence (first encountered = stable order)
    by_sent: "OrderedDict[str, Dict]" = OrderedDict()
    for r in unseen_records:
        sid = r["ID_sentence"]
        if sid in chosen_ids and sid not in by_sent:
            by_sent[sid] = r

    g3_records = list(by_sent.values())
    print(f"  group3 unseen records: {len(g3_records)}")

    # Step 2: random 50% of val.csv
    print(f"[load] val    : {VAL_CSV}")
    val_rows = load_csv(VAL_CSV)
    n_val = int(len(val_rows) * VAL_FRAC)
    val_idx = sorted(rng.sample(range(len(val_rows)), n_val))
    val_sample = [val_rows[i] for i in val_idx]
    print(f"  val total={len(val_rows)}, sampled={len(val_sample)} (frac={VAL_FRAC})")

    # Step 3: random 50% of test.csv
    print(f"[load] test   : {TEST_CSV}")
    test_rows = load_csv(TEST_CSV)
    n_test = int(len(test_rows) * TEST_FRAC)
    test_idx = sorted(rng.sample(range(len(test_rows)), n_test))
    test_sample = [test_rows[i] for i in test_idx]
    print(f"  test total={len(test_rows)}, sampled={len(test_sample)} (frac={TEST_FRAC})")

    out_records = g3_records + val_sample + test_sample
    print(f"[final] {len(out_records)} records "
          f"(group3={len(g3_records)} + val={len(val_sample)} + test={len(test_sample)})")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for rec in out_records:
            w.writerow({k: rec.get(k, "") for k in OUT_FIELDS})

    print(f"[write] -> {OUT_CSV} ({len(out_records)} rows)")
    print("[done]")


if __name__ == "__main__":
    main()
