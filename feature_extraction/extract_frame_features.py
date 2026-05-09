#!/usr/bin/env python3
"""
Extract CLIP-ViT and VideoMAE features from a single frame directory.

Example
-------
python scripts/extract_frame_features.py \
  --frame_path /path/to/Group1/Buoi_2/Front/2438_group1_02438_front_lab_061 \
  --feature_name 2438_2438_group1_02438_front_lab_061 \
  --clip_save_dir features/full_dataset/clip-vit-large-patch14_feat_Full_dataset/test \
  --mae_save_dir features/full_dataset/mae_feat_Full_dataset/test \
  --device cuda:0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    CLIPVisionModel,
    VideoMAEImageProcessor,
    VideoMAEModel,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.helpers import sliding_window_for_list
from utils.s2wrapper import forward as multiscale_forward


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ClipFeatureReader:
    def __init__(
        self,
        model_name: str,
        cache_dir: str | None,
        device: str,
        s2_mode: str,
        scales: List[int],
        nth_layer: int,
    ) -> None:
        self.device = device
        self.s2_mode = s2_mode
        self.scales = scales
        self.nth_layer = nth_layer
        self.model = CLIPVisionModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            cache_dir=cache_dir,
        ).to(device).eval()
        self.image_processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    @torch.no_grad()
    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs).hidden_states
        return outputs[self.nth_layer]

    @torch.no_grad()
    def get_feats(self, frames: List[Image.Image]) -> torch.Tensor:
        pixel_values = self.image_processor(list(frames), return_tensors="pt").to(self.device).pixel_values
        if self.s2_mode == "s2wrapping":
            outputs = multiscale_forward(
                self.forward_features,
                pixel_values,
                scales=self.scales,
                num_prefix_token=1,
            )
        else:
            outputs = self.forward_features(pixel_values)
        return outputs[:, 0]


class MaeFeatureReader:
    def __init__(
        self,
        model_name: str,
        cache_dir: str | None,
        device: str,
        nth_layer: int,
    ) -> None:
        self.device = device
        self.nth_layer = nth_layer
        self.image_processor = VideoMAEImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = VideoMAEModel.from_pretrained(model_name, cache_dir=cache_dir).to(device).eval()

    @torch.no_grad()
    def get_feats(self, frame_windows: List[List[Image.Image]]) -> torch.Tensor:
        inputs = self.image_processor(images=frame_windows, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True).hidden_states
        return outputs[self.nth_layer][:, 0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP and MAE features from a frame directory.")
    parser.add_argument("--frame_path", required=True, help="Directory containing extracted frames.")
    parser.add_argument(
        "--feature_name",
        default=None,
        help="Output feature basename. Defaults to the frame directory name.",
    )
    parser.add_argument("--clip_save_dir", required=True, help="Directory to save CLIP features.")
    parser.add_argument("--mae_save_dir", required=True, help="Directory to save MAE features.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache_dir", default=None)

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


def load_frame_paths(frame_path: str) -> List[Path]:
    frame_dir = Path(frame_path)
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame path not found: {frame_dir}")

    if frame_dir.is_dir():
        frames = sorted(
            p for p in frame_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        frames = sorted(
            p for p in frame_dir.parent.glob(frame_dir.name)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    if not frames:
        raise ValueError(f"No image frames found under: {frame_path}")

    return frames


def open_images(paths: List[Path]) -> List[Image.Image]:
    return [Image.open(path).convert("RGB") for path in paths]


def extract_clip_features(
    frame_paths: List[Path],
    reader: ClipFeatureReader,
    batch_size: int,
) -> np.ndarray:
    frames = open_images(frame_paths)
    outputs = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        outputs.append(reader.get_feats(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def extract_mae_features(
    frame_paths: List[Path],
    reader: MaeFeatureReader,
    batch_size: int,
    overlap_size: int,
) -> np.ndarray:
    usable_paths = list(frame_paths)
    if len(usable_paths) < 16:
        usable_paths.extend([usable_paths[-1]] * (16 - len(usable_paths)))

    frame_windows = sliding_window_for_list(
        usable_paths,
        window_size=16,
        overlap_size=overlap_size,
    )
    videos = [open_images(window) for window in frame_windows]

    outputs = []
    for start in range(0, len(videos), batch_size):
        batch = videos[start : start + batch_size]
        outputs.append(reader.get_feats(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main() -> None:
    args = parse_args()
    feature_name = args.feature_name or Path(args.frame_path).name
    frame_paths = load_frame_paths(args.frame_path)

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

    clip_features = extract_clip_features(frame_paths, clip_reader, args.clip_batch_size)
    mae_features = extract_mae_features(frame_paths, mae_reader, args.mae_batch_size, args.overlap_size)

    os.makedirs(args.clip_save_dir, exist_ok=True)
    os.makedirs(args.mae_save_dir, exist_ok=True)

    clip_postfix = f"_{args.s2_mode}" if args.s2_mode else ""
    mae_postfix = f"_overlap-{args.overlap_size}"

    clip_path = Path(args.clip_save_dir) / f"{feature_name}{clip_postfix}.npy"
    mae_path = Path(args.mae_save_dir) / f"{feature_name}{mae_postfix}.npy"

    np.save(clip_path, clip_features)
    np.save(mae_path, mae_features)

    print(f"Saved CLIP feature to: {clip_path}")
    print(f"Saved MAE feature to: {mae_path}")
    print(f"CLIP shape: {clip_features.shape}")
    print(f"MAE shape: {mae_features.shape}")


if __name__ == "__main__":
    main()
