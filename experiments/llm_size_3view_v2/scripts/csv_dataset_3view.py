"""CSV-based 3-view dataset for SpaMo_3View.

Reads B-format CSV (1 row × 3 path cols: Sentence_video_path_{front,left,right}),
loads pre-extracted CLIP + MAE features per view, returns a sample dict matching
the format expected by `LlamaSLTThreeView.get_inputs()`:

    {
      "id", "text", "gloss", "lang", "vid_path", "bool_mask_pos": None,
      "pixel_value_front", "glor_value_front",
      "pixel_value_left",  "glor_value_left",
      "pixel_value_right", "glor_value_right",
      "en_text", "es_text", "fr_text",
    }

Feature path layout (per view):
    {clip_feat_root}/{video_rel_no_ext}{clip_postfix}.npy
    {mae_feat_root}/{video_rel_no_ext}{mae_postfix}.npy

where video_rel_no_ext is the CSV's path with extension stripped, e.g.
"Group1/Buoi_2/Front/2401_group1_02401_front_lab_061".
"""

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


VIEW_NAMES = ("front", "left", "right")
VIEW_TO_PATHCOL = {
    "front": "Sentence_video_path_front",
    "left": "Sentence_video_path_left",
    "right": "Sentence_video_path_right",
}


class CsvDataset3View(Dataset):
    def __init__(
        self,
        csv_path: str,
        clip_feat_root: str,
        mae_feat_root: str,
        frame_root: str,
        label_column: str = "Sign_sentence",
        clip_postfix: str = "_s2wrapping",
        mae_postfix: str = "_overlap-8",
        lang: str = "Vietnamese",
    ):
        self.clip_feat_root = Path(clip_feat_root)
        self.mae_feat_root = Path(mae_feat_root)
        self.frame_root = Path(frame_root)
        self.clip_postfix = clip_postfix
        self.mae_postfix = mae_postfix
        self.lang = lang
        self.label_column = label_column
        self.samples = self._load_csv(csv_path)
        print(f"[CsvDataset3View] loaded {len(self.samples)} samples from {csv_path}")

    def _load_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            for v, col in VIEW_TO_PATHCOL.items():
                if col not in fields:
                    raise ValueError(f"CSV missing column '{col}' (view={v}): {csv_path}")
            if self.label_column not in fields:
                raise ValueError(f"CSV missing label column '{self.label_column}': {csv_path}")
            id_col = "ID_sentence" if "ID_sentence" in fields else (
                "ID_video" if "ID_video" in fields else None
            )

            for row in reader:
                paths = {}
                ok = True
                for v, col in VIEW_TO_PATHCOL.items():
                    rel = (row.get(col, "") or "").strip().replace("\\", "/").lstrip("/")
                    if not rel:
                        ok = False
                        break
                    paths[v] = rel
                label_text = (row.get(self.label_column, "") or "").strip()
                if not ok or not label_text:
                    continue
                sample_id = (row.get(id_col, "") or "").strip() if id_col else ""
                samples.append({
                    "paths": paths,
                    "text": label_text,
                    "id": sample_id or Path(paths["front"]).stem,
                    "front_video_rel": paths["front"],
                })
        return samples

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.strip()
        if not text.endswith("."):
            text = f"{text}."
        return text

    def _load_view_features(self, video_rel: str) -> Dict[str, torch.Tensor]:
        rel_no_ext = str(Path(video_rel).with_suffix(""))
        clip_path = self.clip_feat_root / f"{rel_no_ext}{self.clip_postfix}.npy"
        mae_path = self.mae_feat_root / f"{rel_no_ext}{self.mae_postfix}.npy"

        if clip_path.exists():
            pixel_value = torch.tensor(np.load(clip_path))
        else:
            print(f"Warning: CLIP feature not found: {clip_path}")
            pixel_value = torch.tensor([])

        if mae_path.exists():
            glor_value = torch.tensor(np.load(mae_path))
        else:
            print(f"Warning: MAE feature not found: {mae_path}")
            glor_value = torch.tensor([])

        return {"pixel_value": pixel_value, "glor_value": glor_value}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        s = self.samples[index]
        result: Dict[str, Any] = {"bool_mask_pos": None}

        for v in VIEW_NAMES:
            feats = self._load_view_features(s["paths"][v])
            result[f"pixel_value_{v}"] = feats["pixel_value"]
            result[f"glor_value_{v}"] = feats["glor_value"]
            result[f"num_frames_{v}"] = (
                feats["pixel_value"].shape[0] if feats["pixel_value"].ndim >= 2 else 0
            )
            result[f"fileid_{v}"] = Path(s["paths"][v]).stem

        text_norm = self._normalize_text(s["text"])
        result["id"] = s["id"]
        result["text"] = text_norm
        result["gloss"] = s["text"]
        result["lang"] = self.lang
        result["vid_path"] = str(self.frame_root / s["front_video_rel"])
        result["en_text"] = text_norm
        result["es_text"] = text_norm
        result["fr_text"] = text_norm
        return result

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        return batch
