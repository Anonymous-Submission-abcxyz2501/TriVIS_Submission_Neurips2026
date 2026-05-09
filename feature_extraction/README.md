# Feature extraction

The 3-view fusion model consumes **two** pre-extracted feature streams per video:

| Stream | Backbone | Postfix on disk | Resulting `.npy` shape |
|---|---|---|---|
| Spatial      | CLIP-ViT-L/14 with multi-scale (`s2wrapping`) | `_s2wrapping.npy`  | `(T, 2048)` |
| Spatiotemporal | VideoMAE-large with `overlap_size=8`        | `_overlap-8.npy`   | `(T, 1024)` |

`T` = number of frames (CLIP) or windows (VideoMAE) per video.

The configs reference the features by their flat `<feat_root>/<rel_path>{postfix}.npy`
layout, so the output directory of each extractor becomes `CLIP_FEAT_ROOT` /
`MAE_FEAT_ROOT` for both training and evaluation.

> **Disk budget.** Full TriViS (84,453 videos × 2 streams) is ~80 GB on disk.
> If you only need to evaluate the released checkpoint, extract features for
> `val.csv`, `test.csv`, `test_unseen.csv`, and `test_real_group1.csv` only
> (~7,200 videos × 2 streams ≈ 8 GB).

---

## Recommended path: one CSV-driven sweep per split

`extract_csv_features.py` extracts both streams together, takes a CSV (or
several) as input, and walks the front/left/right paths declared in each row.

```bash
python feature_extraction/extract_csv_features.py \
    --csv_paths splits/three_view/e1_sentence_progressive_lab/train.csv \
                splits/three_view/e1_sentence_progressive_lab/val.csv \
                splits/three_view/e1_sentence_progressive_lab/test.csv \
                splits/three_view/e1_sentence_progressive_lab/test_unseen.csv \
                splits/three_view/e1_sentence_progressive_lab/test_real_group1.csv \
    --frame_root      "$FRAME_ROOT" \
    --clip_save_root  "$CLIP_FEAT_ROOT" \
    --mae_save_root   "$MAE_FEAT_ROOT" \
    --view_column     Sentence_video_path_front \
    --label_column    Sign_sentence \
    --id_column       ID_video \
    --s2_mode         s2wrapping \
    --scales 1 2 \
    --overlap_size    8 \
    --clip_batch_size 32 \
    --mae_batch_size  16 \
    --device          cuda:0
```

Repeat with `--view_column Sentence_video_path_left` and
`Sentence_video_path_right` so all three views are extracted.

`FRAME_ROOT` should point to a directory containing **decoded frame folders**.
For each video referenced by the CSV the decoder expects:
`$FRAME_ROOT/<rel_path_without_extension>/<frame_idx>.jpg`. Decode the
TriViS videos to JPGs once with your preferred frame extractor (ffmpeg /
decord) before running the feature extractor.

---

## Lower-level extractors

Use these only if you need fine-grained control or want to extract a single
stream:

```bash
# CLIP-ViT-L/14 (spatial)
python feature_extraction/vit_extract_feature.py \
    --anno_root  splits/three_view/e1_sentence_progressive_lab \
    --video_root "$FRAME_ROOT" \
    --save_dir   "$CLIP_FEAT_ROOT" \
    --s2_mode    s2wrapping \
    --scales 1 2 \
    --batch_size 32 \
    --device     cuda:0

# VideoMAE-large (spatiotemporal)
python feature_extraction/mae_extract_feature.py \
    --anno_root  splits/three_view/e1_sentence_progressive_lab \
    --video_root "$FRAME_ROOT" \
    --save_dir   "$MAE_FEAT_ROOT" \
    --overlap_size 8 \
    --batch_size 32 \
    --device     cuda:0
```

`extract_frame_features.py` is a small per-frame helper used by
`extract_csv_features.py`; it is not normally invoked directly.

---

## Sanity check after extraction

```bash
python - <<'PY'
import numpy as np, glob, os, sys
clip = sorted(glob.glob(os.path.join(os.environ["CLIP_FEAT_ROOT"], "**/*_s2wrapping.npy"), recursive=True))
mae  = sorted(glob.glob(os.path.join(os.environ["MAE_FEAT_ROOT"],  "**/*_overlap-8.npy"),  recursive=True))
print(f"CLIP files: {len(clip)}; first shape: {np.load(clip[0]).shape if clip else 'none'}")
print(f"MAE  files: {len(mae)}; first shape:  {np.load(mae[0]).shape  if mae  else 'none'}")
PY
```

Expected: CLIP shape `(T, 2048)`, MAE shape `(T, 1024)`. The training config
`finetune_3view_attn_videollama3_7b.yaml` declares `input_size: 2048` and
`inter_hidden: 768` matching these dimensions.
