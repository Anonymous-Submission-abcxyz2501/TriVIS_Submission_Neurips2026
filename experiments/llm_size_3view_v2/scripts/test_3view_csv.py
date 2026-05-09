"""test_3view_csv.py — Test-only entry for SpaMo_3View (CSV-based 3-view).

Loads a model + ckpt and runs trainer.test() on a single CSV. Designed for
val / test / test_unseen evaluation post-training. Does NOT modify codebase.

Usage:
    python test_3view_csv.py \\
        --config <path/to/config.yaml> \\
        --ckpt <path/to/best.ckpt> \\
        --csv <path/to/{val,test,test_unseen}.csv> \\
        --clip_feat_root ... --mae_feat_root ... --frame_root ... \\
        --label_column Sign_sentence \\
        --batch_size 8 \\
        --split_name test_unseen
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader

SPAMO_ROOT = Path(__file__).resolve().parents[3]
THIS_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPAMO_ROOT))
sys.path.insert(0, str(THIS_SCRIPTS_DIR))

from utils.helpers import instantiate_from_config
from csv_dataset_3view import CsvDataset3View


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SpaMo_3View test-only on a CSV split")
    p.add_argument("-c", "--config", required=True, help="YAML config (model section)")
    p.add_argument("--ckpt", required=True, help="Checkpoint .ckpt path")
    p.add_argument("--csv", required=True, help="Test CSV (B-format: 3 path cols)")
    p.add_argument("--clip_feat_root", required=True)
    p.add_argument("--mae_feat_root", required=True)
    p.add_argument("--frame_root", required=True)
    p.add_argument("--label_column", default="Sign_sentence")
    p.add_argument("--clip_postfix", default="_s2wrapping")
    p.add_argument("--mae_postfix", default="_overlap-8")
    p.add_argument("--lang", default="Vietnamese")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--split_name", default="test", help="Label printed in output")
    p.add_argument("-s", "--seed", type=int, default=0)
    return p


def main():
    opt = get_parser().parse_args()
    seed_everything(opt.seed)

    print(f"[test_3view] config = {opt.config}")
    print(f"[test_3view] ckpt   = {opt.ckpt}")
    print(f"[test_3view] csv    = {opt.csv} (split_name={opt.split_name})")

    config = OmegaConf.load(opt.config)
    model = instantiate_from_config(config.model)
    print(f"[test_3view] model class = {type(model).__name__}")

    state = torch.load(opt.ckpt, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[test_3view] missing keys: {len(missing)} (showing 5): {missing[:5]}")
    if unexpected:
        print(f"[test_3view] unexpected keys: {len(unexpected)} (showing 5): {unexpected[:5]}")

    ds = CsvDataset3View(
        csv_path=opt.csv,
        clip_feat_root=opt.clip_feat_root,
        mae_feat_root=opt.mae_feat_root,
        frame_root=opt.frame_root,
        label_column=opt.label_column,
        clip_postfix=opt.clip_postfix,
        mae_postfix=opt.mae_postfix,
        lang=opt.lang,
    )
    print(f"[test_3view] {len(ds)} samples")

    loader = DataLoader(
        ds, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, collate_fn=CsvDataset3View.collate_fn,
    )

    trainer_cfg = OmegaConf.to_container(
        config.lightning.trainer, resolve=True
    ) if "lightning" in config and "trainer" in config.lightning else {}
    # strip training-only keys
    for k in ["max_epochs", "accumulate_grad_batches", "gradient_clip_val",
              "check_val_every_n_epoch", "num_sanity_val_steps"]:
        trainer_cfg.pop(k, None)
    trainer_cfg.setdefault("accelerator", "gpu")
    trainer_cfg.setdefault("devices", 1)
    trainer_cfg.setdefault("precision", "bf16")

    trainer = Trainer(**trainer_cfg, logger=False)

    print(f"\n[test_3view] === [{opt.split_name}] running trainer.test ===")
    results = trainer.test(model, dataloaders=loader)
    print(f"\n[test_3view] === [{opt.split_name}] DONE ===")
    print(f"[test_3view] {opt.split_name} metrics:")
    for r in results:
        for k, v in r.items():
            print(f"    {k:30s} = {v:.4f}" if isinstance(v, float) else f"    {k:30s} = {v}")


if __name__ == "__main__":
    main()
