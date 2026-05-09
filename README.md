# TriViS — Multi-View Fusion Baseline (`Spamo_MV`)

> Reference implementation for the multi-view fusion model accompanying the paper
> **"TriViS: A Large-Scale Multi-View Benchmark for Vietnamese Sign Language Recognition"**.
>
> *Released as part of the paper review process.*

## Abstract

This paper presents **TriViS**, the first large-scale multi-view Vietnamese dataset
for Continuous Sign Language Recognition and Translation, designed as a controlled
benchmark for studying dataset design in underrepresented sign languages. TriViS
contains 12,000 sentences and 84,453 synchronized videos captured from three
viewpoints (frontal, 45° left, and 45° right) across both controlled studio
recordings and outdoor environments. The synchronized multi-view setup enables
systematic analysis of viewpoint sensitivity, view complementarity, and other data
collection factors that are difficult to isolate in uncontrolled web-scale corpora.
Along with the dataset, we introduce a simple multi-view fusion framework that
integrates synchronized visual streams into a unified representation for
sentence-level recognition and translation. Using state-of-the-art CSLR and CSLT
baselines, we conduct extensive experiments to evaluate the contribution of
individual viewpoints, multi-view fusion strategies, sentence-independent
generalization, and the effect of LLM scale. Our results show that multi-view
observations consistently outperform single-view baselines and provide new
empirical insights into how viewpoint configuration and controlled data
collection affect continuous sign language understanding.

## Contents

- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Dataset](#dataset)
- [External model weights](#external-model-weights)
- [Feature extraction](#feature-extraction)
- [Train](#train)
- [Evaluate](#evaluate)
- [Pretrained checkpoints](#pretrained-checkpoints)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Repository layout

```
Spamo_MV/
├── README.md                              # this file
├── LICENSE
├── requirements.txt
├── main.py                                # generic Lightning entry (see scripts/ for the 3-view entry)
│
├── spamo/                                 # model code
│   ├── llama_slt_3view.py                 # LlamaSLTThreeView base + ViewMerge
│   ├── llama_slt.py                       # 1-view baseline (transitive import)
│   ├── asb.py / callbacks.py / clip_loss.py / mm_projector.py / tconv.py / ...
│   └── ...
├── dataset/                               # data loading
│   ├── p14t_3view.py
│   ├── p14t.py
│   └── datamodule.py
├── utils/                                 # metrics + helpers
│   ├── evaluate.py                        # BLEU / ROUGE-L / WER (jiwer)
│   └── helpers.py
│
├── experiments/llm_size_3view_v2/
│   ├── README.md                          # quick reference for this experiment
│   ├── configs/
│   │   └── finetune_3view_attn_videollama3_7b.yaml   # released 7B config
│   └── scripts/
│       ├── main_3view_csv.py              # train entry  (CSV-driven 3-view)
│       ├── test_3view_csv.py              # test  entry  (single CSV split)
│       ├── csv_dataset_3view.py           # 3-view CSV dataset
│       ├── llama_slt_3view_attn.py        # LlamaSLT3ViewAttnPool (fusion model)
│       ├── train_3view_attn.sh            # train driver shell
│       └── test_all_3view.sh              # test driver shell
│
├── feature_extraction/                    # CLIP / VideoMAE feature extractors
│   ├── README.md
│   ├── vit_extract_feature.py             # CLIP-ViT-L/14 + s2wrapping
│   ├── mae_extract_feature.py             # VideoMAE + overlap-8
│   └── extract_csv_features.py            # CSV-driven combined extractor
│
└── splits/three_view/e1_sentence_progressive_lab/
    ├── train.csv                          # 4,800 sentences × 3 views
    ├── val.csv                            # 1,200 sentences × 3 views
    ├── test.csv                           # 1,200 sentences × 3 views
    ├── test_unseen.csv                    # 2,200 rows: 1,000 unseen Group3 + 600 val + 600 test
    └── test_real_group1.csv               # 1,000 lab→real domain-shift evaluation
```

## Setup

```bash
git clone <this-repo> Spamo_MV
cd Spamo_MV

conda create -n spamo_mv python=3.10 -y
conda activate spamo_mv

pip install -r requirements.txt
```

The pinned dependencies were validated on PyTorch 2.0.1 + CUDA 11.8 with
`pytorch_lightning==1.9.5` and `transformers==4.32.0`. Newer Lightning / TF
versions are not guaranteed to work (model/trainer hooks were stable for the
training runs reported in the paper at these versions).

## Dataset

TriViS is hosted on Hugging Face:

| Variant | Link |
|---|---|
| Full dataset (84,453 videos) | <https://huggingface.co/datasets/Abcxyz2501/Full_TriVis> |
| Representative subset        | <https://huggingface.co/datasets/Abcxyz2501/Trivis_Representative_Samples> |

```bash
# Full dataset (≈ 256 GB)
huggingface-cli download Abcxyz2501/Full_TriVis \
    --repo-type=dataset --local-dir "$DATA_ROOT"

# Subset (≈ a few GB) — handy for smoke tests
huggingface-cli download Abcxyz2501/Trivis_Representative_Samples \
    --repo-type=dataset --local-dir "$DATA_ROOT/subset"
```

The CSV splits in `splits/three_view/e1_sentence_progressive_lab/` reference
videos by the relative paths used inside the HF release, so once the dataset
is extracted the CSVs work as-is.

After downloading, decode each video to per-frame JPGs:

```
$FRAME_ROOT/<rel_path_without_extension>/<frame_idx>.jpg
```

Use ffmpeg / decord / your tool of choice; the `feature_extraction/` scripts
expect this layout.

## External model weights

The 7B config uses **VideoLLaMA3-7B** as the LLM backbone. Download the public
weights and point `VIDEOLLAMA3_7B_PATH` at the snapshot directory:

```bash
huggingface-cli download DAMO-NLP-SG/VideoLLaMA3-7B \
    --local-dir /path/to/videollama3_7b_local
export VIDEOLLAMA3_7B_PATH=/path/to/videollama3_7b_local
```

Tokenizers and minor model assets are pulled from the standard HF cache; set
`HF_CACHE_DIR` if you want to override the default `~/.cache/huggingface`.

## Feature extraction

The model consumes pre-extracted CLIP-ViT-L/14 (`_s2wrapping.npy`) and
VideoMAE-large (`_overlap-8.npy`) features. See
[`feature_extraction/README.md`](feature_extraction/README.md) for the full
guide; the canonical command is:

```bash
export FRAME_ROOT=/path/to/decoded/frames
export CLIP_FEAT_ROOT=/path/to/output/clip_features
export MAE_FEAT_ROOT=/path/to/output/mae_features

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
    --device          cuda:0
```

Repeat with `Sentence_video_path_left` / `Sentence_video_path_right` so all
three views are extracted. Output volume: ~80 GB for the full dataset, ~8 GB
for the eval-only splits.

## Train

```bash
export REPO_ROOT=$(pwd)
export VIDEOLLAMA3_7B_PATH=/path/to/videollama3_7b_local
export CLIP_FEAT_ROOT=/path/to/clip_features
export MAE_FEAT_ROOT=/path/to/mae_features
export FRAME_ROOT=/path/to/decoded/frames

LLM_SIZE=videollama3_7b \
  bash experiments/llm_size_3view_v2/scripts/train_3view_attn.sh
```

Default batch size for the 7B config is 8 (override via `BATCH_SIZE=4` for
GPUs with < 80 GB VRAM). Logs land at
`$REPO_ROOT/logs/$USER/e1_3view_attn_videollama3_7b/<run_tag>/`.

## Evaluate

```bash
export REPO_ROOT=$(pwd)
export VIDEOLLAMA3_7B_PATH=/path/to/videollama3_7b_local
export CLIP_FEAT_ROOT=/path/to/clip_features
export MAE_FEAT_ROOT=/path/to/mae_features
export FRAME_ROOT=/path/to/decoded/frames
export CKPT_PATH=/path/to/best.ckpt

# All four splits (val, test, test_unseen, test_real_group1):
bash experiments/llm_size_3view_v2/scripts/test_all_3view.sh

# A subset:
SPLITS="val test_unseen" \
  bash experiments/llm_size_3view_v2/scripts/test_all_3view.sh
```

Each run produces a per-split `.log` plus a combined `.summary` under
`experiments/llm_size_3view_v2/results/`, with BLEU-1..4, ROUGE-L
(precision/recall/F1), and WER.

## Pretrained checkpoints

`Updating…`

## Expected results

Numbers below come from our 7B training run and are intended as a sanity
target after a successful reproduction.

| Split                  | BLEU-4 | ROUGE-L F1 |
|------------------------|--------|------------|
| `val`                  | 26.71  | …          |
| `test`                 | 26.7x  | …          |
| `test_unseen`          | …      | …          |
| `test_real_group1`     | …      | …          |

## Citation

```bibtex
@misc{trivis2026,
  title  = {TriViS: A Large-Scale Multi-View Benchmark for Vietnamese Sign Language Recognition},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Under review.}
}
```

## License

`TBD` — license to be selected by the authors prior to public release.

## Acknowledgements

This codebase builds on the SpaMo SLT framework, the VideoLLaMA3 visual–LM
backbone, and the standard Phoenix14T evaluation conventions.
