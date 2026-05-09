#!/usr/bin/env python3
"""
Extract CLIP-ViT and VideoMAE features for samples listed in one or more CSV files.

This script uses CSV rows as the source of truth, resolves each relative video path
to a frame directory under ``--frame_root``, and saves SpaMo-compatible feature files:

  <clip_save_root>/<split>/<fileid>_s2wrapping.npy
  <mae_save_root>/<split>/<fileid>_overlap-8.npy
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_frame_features import (
    ClipFeatureReader,
    MaeFeatureReader,
    extract_clip_features,
    extract_mae_features,
    load_frame_paths,
)


def normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def pick_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    by_norm = {normalize_col(name): name for name in fieldnames}
    for candidate in candidates:
        key = normalize_col(candidate)
        if key in by_norm:
            return by_norm[key]
    return None


def make_file_id(video_rel: str, explicit_id: str, seen: Dict[str, int]) -> str:
    base = Path(video_rel.rstrip("/")).name or explicit_id or "sample"
    candidate = f"{explicit_id}_{base}" if explicit_id else base
    if candidate not in seen:
        seen[candidate] = 1
        return candidate
    suffix = seen[candidate]
    seen[candidate] += 1
    return f"{candidate}_{suffix}"


def format_gpu_memory(device: str) -> str:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return "gpu=n/a"

    try:
        device_index = torch.device(device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
    except RuntimeError:
        device_index = torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
    return f"gpu={allocated:.1f}/{reserved:.1f}G"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CLIP and MAE features for samples listed in CSV files."
    )
    parser.add_argument("--csv_paths", nargs="+", required=True, help="One or more CSV files.")
    parser.add_argument("--frame_root", required=True, help="Root directory containing extracted frame folders.")
    parser.add_argument("--clip_save_root", required=True, help="Root directory to save CLIP features.")
    parser.add_argument("--mae_save_root", required=True, help="Root directory to save MAE features.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature files.")

    parser.add_argument("--view_column", default="Sentence_video_path_front")
    parser.add_argument("--label_column", default="sign-sentence")
    parser.add_argument("--id_column", default="ID_sentence")

    parser.add_argument("--clip_model_name", default="openai/clip-vit-large-patch14")
    parser.add_argument("--clip_batch_size", type=int, default=32)
    parser.add_argument("--s2_mode", default="s2wrapping")
    parser.add_argument("--scales", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--clip_nth_layer", type=int, default=-1)

    parser.add_argument("--mae_model_name", default="MCG-NJU/videomae-large")
    parser.add_argument("--mae_batch_size", type=int, default=16)
    parser.add_argument("--mae_nth_layer", type=int, default=-1)
    parser.add_argument("--overlap_size", type=int, default=8)
    return parser.parse_args()


def read_csv_rows(csv_path: str, args: argparse.Namespace) -> List[Tuple[str, Path, str]]:
    rows: List[Tuple[str, Path, str]] = []
    seen_ids: Dict[str, int] = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")

        view_col = pick_column(
            reader.fieldnames,
            [
                args.view_column,
                "Sentence_video_path_front",
                "Sentence_video_path_right",
                "Sentence_video_path_left",
                "Sentence_video_path",
            ],
        )
        if view_col is None:
            raise ValueError(f"Cannot find a video path column in {csv_path}")

        label_col = pick_column(
            reader.fieldnames,
            [
                args.label_column,
                "Sign_sentence",
                "sign_sentence",
                "sign-sentence",
            ],
        )
        if label_col is None:
            raise ValueError(f"Cannot find a label column in {csv_path}")

        id_col = pick_column(reader.fieldnames, [args.id_column, "ID_sentence", "id_sentence", "id"])

        for row in reader:
            video_rel = (row.get(view_col, "") or "").strip().replace("\\", "/").lstrip("/")
            label_text = (row.get(label_col, "") or "").strip()
            if not video_rel or not label_text:
                continue

            video_rel_no_ext = str(Path(video_rel).with_suffix(""))
            explicit_id = (row.get(id_col, "") or "").strip() if id_col else ""
            file_id = make_file_id(video_rel_no_ext, explicit_id, seen_ids)
            rows.append((file_id, Path(video_rel_no_ext), video_rel))

    return rows


def main() -> None:
    args = parse_args()

    frame_root = Path(args.frame_root).resolve()
    clip_save_root = Path(args.clip_save_root).resolve()
    mae_save_root = Path(args.mae_save_root).resolve()

    clip_reader = ClipFeatureReader(
        model_name=args.clip_model_name,
        cache_dir=args.cache_dir,
        device=args.device,
        s2_mode=args.s2_mode,
        scales=args.scales,
        nth_layer=args.clip_nth_layer,
    )
    mae_reader = MaeFeatureReader(
        model_name=args.mae_model_name,
        cache_dir=args.cache_dir,
        device=args.device,
        nth_layer=args.mae_nth_layer,
    )

    clip_suffix = f"_{args.s2_mode}" if args.s2_mode else ""
    mae_suffix = f"_overlap-{args.overlap_size}"

    total_processed = 0
    total_skipped = 0

    print(
        f"[extract] device={args.device} clip_batch={args.clip_batch_size} "
        f"mae_batch={args.mae_batch_size} {format_gpu_memory(args.device)}"
    )

    for csv_path in args.csv_paths:
        rows = read_csv_rows(csv_path, args)
        if not rows:
            print(f"[warn] No usable rows found in {csv_path}")
            continue

        progress = tqdm(
            rows,
            desc=f"Extracting {Path(csv_path).stem}",
            unit="sample",
            dynamic_ncols=True,
            file=sys.stdout,
        )
        progress.set_postfix_str(format_gpu_memory(args.device))
        progress.refresh()

        split_processed = 0
        split_skipped = 0
        for file_id, relative_frame_path, original_video_rel in progress:
            frame_dir = frame_root / relative_frame_path
            feature_rel_dir = relative_frame_path.parent
            feature_name = relative_frame_path.name

            clip_dir = clip_save_root / feature_rel_dir
            mae_dir = mae_save_root / feature_rel_dir
            clip_dir.mkdir(parents=True, exist_ok=True)
            mae_dir.mkdir(parents=True, exist_ok=True)

            clip_path = clip_dir / f"{feature_name}{clip_suffix}.npy"
            mae_path = mae_dir / f"{feature_name}{mae_suffix}.npy"

            if not args.overwrite and clip_path.exists() and mae_path.exists():
                split_skipped += 1
                progress.set_postfix_str(
                    f"{feature_name} skip={split_skipped} {format_gpu_memory(args.device)}"
                )
                progress.refresh()
                continue

            if not frame_dir.exists():
                raise FileNotFoundError(
                    f"Frame directory not found for CSV path '{original_video_rel}': {frame_dir}"
                )

            progress.set_postfix_str(f"{feature_name} {format_gpu_memory(args.device)}")
            progress.refresh()

            frame_paths = load_frame_paths(str(frame_dir))
            clip_features = extract_clip_features(frame_paths, clip_reader, args.clip_batch_size)
            mae_features = extract_mae_features(
                frame_paths,
                mae_reader,
                args.mae_batch_size,
                args.overlap_size,
            )

            np.save(clip_path, clip_features)
            np.save(mae_path, mae_features)
            split_processed += 1
            progress.set_postfix_str(
                f"{feature_name} ok={split_processed} skip={split_skipped} {format_gpu_memory(args.device)}"
            )
            progress.refresh()

        total_processed += split_processed
        total_skipped += split_skipped
        print(
            f"Finished CSV '{Path(csv_path).name}': processed={split_processed}, "
            f"skipped={split_skipped}, total={len(rows)}"
        )

    print(f"Done. processed={total_processed}, skipped={total_skipped}")


if __name__ == "__main__":
    main()
