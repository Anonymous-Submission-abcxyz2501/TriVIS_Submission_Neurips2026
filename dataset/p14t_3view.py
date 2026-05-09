import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Union


VIEW_NAMES = ("front", "left", "right")


class Phoenix14TThreeView(torch.utils.data.Dataset):
    """Three-view dataset — returns CLIP+MAE features for front/left/right per sample.

    The preparation script runs once per view over the same CSV rows in order, so
    sample index i in `{view}_info_ml.npy` is the same underlying sentence across
    all views. This dataset asserts count alignment at init time.
    """

    def __init__(
        self,
        anno_root_front: str,
        anno_root_left: str,
        anno_root_right: str,
        feat_root_front: str,
        feat_root_left: str,
        feat_root_right: str,
        mae_feat_root_front: str,
        mae_feat_root_left: str,
        mae_feat_root_right: str,
        vid_root: str,
        mode: str = "dev",
        spatial_postfix: str = "",
        spatiotemporal_postfix: Union[str, List[str]] = "",
    ):
        super().__init__()
        self.mode = mode
        self.spatial_postfix = spatial_postfix
        self.spatiotemporal_postfix = spatiotemporal_postfix
        self.vid_root = Path(vid_root)

        self.views = {
            "front": {
                "anno_root": Path(anno_root_front),
                "feat_root": Path(feat_root_front),
                "mae_feat_root": Path(mae_feat_root_front),
            },
            "left": {
                "anno_root": Path(anno_root_left),
                "feat_root": Path(feat_root_left),
                "mae_feat_root": Path(mae_feat_root_left),
            },
            "right": {
                "anno_root": Path(anno_root_right),
                "feat_root": Path(feat_root_right),
                "mae_feat_root": Path(mae_feat_root_right),
            },
        }

        self.data: Dict[str, Dict] = {}
        for v, paths in self.views.items():
            anno_path = paths["anno_root"] / f"{mode}_info_ml.npy"
            if not anno_path.exists():
                raise FileNotFoundError(f"[{v}] annotation not found: {anno_path}")
            self.data[v] = np.load(anno_path, allow_pickle=True).item()

        # Annotation dicts contain a 'prefix' key plus integer sample keys 0..N-1.
        lengths = [len(self.data[v]) - 1 for v in VIEW_NAMES]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Per-view annotations are misaligned: {dict(zip(VIEW_NAMES, lengths))}"
            )
        self._N = lengths[0]

        # Per-index text/gloss alignment across views. Preparation scripts run
        # once per view over the same CSV rows, but there is no global contract
        # enforcing this — verify once to catch silent sample swaps.
        front_data = self.data["front"]
        for idx in range(self._N):
            front_text = front_data[idx]["text"]
            front_gloss = front_data[idx]["gloss"]
            for v in ("left", "right"):
                if self.data[v][idx]["text"] != front_text:
                    raise ValueError(
                        f"Text misalignment at index {idx}: front={front_text!r} "
                        f"{v}={self.data[v][idx]['text']!r}"
                    )
                if self.data[v][idx]["gloss"] != front_gloss:
                    raise ValueError(
                        f"Gloss misalignment at index {idx}: front={front_gloss!r} "
                        f"{v}={self.data[v][idx]['gloss']!r}"
                    )

        for v in VIEW_NAMES:
            sdir = self.views[v]["feat_root"] / self.mode
            mdir = self.views[v]["mae_feat_root"] / self.mode
            if not sdir.exists():
                raise FileNotFoundError(f"[{v}] spatial feat dir missing: {sdir}")
            if not mdir.exists():
                raise FileNotFoundError(f"[{v}] spatiotemporal feat dir missing: {mdir}")

        # Fail-fast: every sample must have all 6 feature files (3 views × spatial+MAE).
        # Runs once at instantiation; on 4800 samples this adds ~1-2s.
        missing: List[str] = []
        for v in VIEW_NAMES:
            feat_dir = self.views[v]["feat_root"] / self.mode
            mae_dir = self.views[v]["mae_feat_root"] / self.mode
            for idx in range(self._N):
                fid = self.data[v][idx]["fileid"]
                sp = feat_dir / f"{fid}{self.spatial_postfix}.npy"
                if not sp.exists():
                    missing.append(str(sp))
                if isinstance(self.spatiotemporal_postfix, str):
                    mp_paths = [mae_dir / f"{fid}{self.spatiotemporal_postfix}.npy"]
                else:
                    mp_paths = [mae_dir / f"{fid}{pf}.npy" for pf in self.spatiotemporal_postfix]
                for mp in mp_paths:
                    if not mp.exists():
                        missing.append(str(mp))
        if missing:
            raise FileNotFoundError(
                f"[{mode}] {len(missing)} feature files missing. First 5:\n  "
                + "\n  ".join(missing[:5])
            )
        print(f"[Phoenix14TThreeView/{mode}] verified {self._N} samples × all feature files present.")

    def __len__(self) -> int:
        return self._N

    def _load_spatial(self, view: str, file_id: str) -> torch.Tensor:
        feat_path = self.views[view]["feat_root"] / self.mode / f"{file_id}{self.spatial_postfix}.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"[{view}] spatial feature missing: {feat_path}")
        return torch.tensor(np.load(feat_path))

    def _load_spatiotemporal(
        self, view: str, file_id: str
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        mae_root = self.views[view]["mae_feat_root"] / self.mode
        if isinstance(self.spatiotemporal_postfix, str):
            p = mae_root / f"{file_id}{self.spatiotemporal_postfix}.npy"
            if not p.exists():
                raise FileNotFoundError(f"[{view}] MAE feature missing: {p}")
            return torch.tensor(np.load(p))
        out = []
        for postfix in self.spatiotemporal_postfix:
            p = mae_root / f"{file_id}{postfix}.npy"
            if not p.exists():
                raise FileNotFoundError(f"[{view}] MAE feature missing: {p}")
            out.append(torch.tensor(np.load(p)))
        return out

    def _normalize_text(self, text: str) -> str:
        text = text.strip()
        if not text.endswith("."):
            text = f"{text}."
        return text

    def __getitem__(self, index: int) -> Dict[str, Any]:
        result: Dict[str, Any] = {"bool_mask_pos": None}

        for v in VIEW_NAMES:
            d = self.data[v][index]
            file_id = d["fileid"]

            pixel_value = self._load_spatial(v, file_id)
            glor_value = self._load_spatiotemporal(v, file_id)

            result[f"pixel_value_{v}"] = pixel_value
            result[f"glor_value_{v}"] = glor_value
            result[f"num_frames_{v}"] = len(pixel_value) if pixel_value is not None else 0
            result[f"fileid_{v}"] = file_id

        front = self.data["front"][index]
        folder = front.get("folder", "")
        result["id"] = front["fileid"]
        result["text"] = self._normalize_text(front["text"])
        result["gloss"] = front["gloss"]
        result["lang"] = front.get("lang", "Vietnamese")
        result["vid_path"] = str(self.vid_root / folder) if folder else str(self.vid_root)
        for lang in ("en", "es", "fr"):
            result[f"{lang}_text"] = front.get(f"{lang}_text", front["text"])
        result["original_info"] = front

        return result

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        return batch
