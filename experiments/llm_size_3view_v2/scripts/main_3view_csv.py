"""main_3view_csv.py — Lightning entry for SpaMo_3View with CSV-based 3-view loader.

Differences from SpaMo_3View/main.py:
- No DataModuleFromConfig — instead instantiates `CsvDataset3View` directly for
  train/val/test, all from B-format CSVs (1 row × 3 path cols).
- Model class instantiated via YAML target (defaults to `LlamaSLT3ViewAttnPool`
  via the corresponding config files in `experiments/llm_size_3view_v2/configs/`).

Train/val/test loaders all use the same CsvDataset3View → 3 video features per
sample (front/left/right). LlamaSLTThreeView.shared_step does true 3-view
fusion (per-view encoder + view-merge module → LM).
"""

import argparse
import datetime
import glob
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader

SPAMO_ROOT = Path(__file__).resolve().parents[3]
THIS_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPAMO_ROOT))
sys.path.insert(0, str(THIS_SCRIPTS_DIR))

from utils.helpers import instantiate_from_config
from spamo.callbacks import SetupCallback
from csv_dataset_3view import CsvDataset3View


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SpaMo_3View — 3-view CSV training (LLM size sweep, attn-pool fusion)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config", nargs="*", default=[],
                   help="YAML config file(s) — model + lightning")
    p.add_argument("--train_csv", required=True,
                   help="Train CSV (B-format: 3 path cols)")
    p.add_argument("--val_csv", required=True,
                   help="Val CSV (B-format: 3 path cols)")
    p.add_argument("--test_csv", required=True,
                   help="Test CSV (B-format: 3 path cols)")
    p.add_argument("--clip_feat_root", required=True)
    p.add_argument("--mae_feat_root", required=True)
    p.add_argument("--frame_root", required=True)
    p.add_argument("--label_column", default="Sign_sentence")
    p.add_argument("--clip_postfix", default="_s2wrapping")
    p.add_argument("--mae_postfix", default="_overlap-8")
    p.add_argument("--lang", default="Vietnamese")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("-s", "--seed", type=int, default=0)
    p.add_argument("-n", "--name", type=str, default="")
    p.add_argument("--postfix", type=str, default="")
    p.add_argument("-l", "--logdir", type=str, default="logs")
    p.add_argument("-r", "--resume", default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("-e", "--evaluation", type=str, default="bleu",
                   choices=["bleu", "mse"])
    return p


def _ds_kwargs(opt) -> dict:
    return dict(
        clip_feat_root=opt.clip_feat_root,
        mae_feat_root=opt.mae_feat_root,
        frame_root=opt.frame_root,
        label_column=opt.label_column,
        clip_postfix=opt.clip_postfix,
        mae_postfix=opt.mae_postfix,
        lang=opt.lang,
    )


def main():
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    opt, _ = get_parser().parse_known_args()
    if opt.name and opt.resume:
        raise ValueError("--name and --resume cannot both be specified.")
    seed_everything(opt.seed)

    if opt.resume:
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", opt.ckpt) if opt.ckpt else None
        nowname = logdir.split("/")[-1]
    else:
        name = ("_" + opt.name) if opt.name else ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)
        ckpt = opt.ckpt
    ckptdir = os.path.join(logdir, "checkpoints")

    if opt.resume:
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.config = base_configs + (opt.config or [])
    configs = [OmegaConf.load(c) for c in (opt.config or [])]
    config = OmegaConf.merge(*configs) if configs else OmegaConf.create()
    lightning_config = config.pop("lightning", OmegaConf.create())
    if "data" in config:
        config.pop("data")
    trainer_config = OmegaConf.to_container(
        lightning_config.get("trainer", OmegaConf.create()), resolve=True
    )
    lightning_config.trainer = trainer_config

    model = instantiate_from_config(config.model)
    print(f"[main] model class = {type(model).__name__}")
    if ckpt is not None and not opt.resume:
        model.load_pretrained_weights(ckpt)

    train_ds = CsvDataset3View(csv_path=opt.train_csv, **_ds_kwargs(opt))
    val_ds = CsvDataset3View(csv_path=opt.val_csv, **_ds_kwargs(opt))
    test_ds = CsvDataset3View(csv_path=opt.test_csv, **_ds_kwargs(opt))
    print(f"[data] train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, collate_fn=CsvDataset3View.collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, collate_fn=CsvDataset3View.collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, collate_fn=CsvDataset3View.collate_fn,
    )

    callbacks = []
    if hasattr(lightning_config, "callback"):
        callbacks = [
            instantiate_from_config(lightning_config.callback[k])
            for k in lightning_config.callback.keys()
        ]
    if opt.evaluation == "bleu":
        callbacks.append(ModelCheckpoint(
            dirpath=ckptdir,
            filename="epoch={epoch:05}-step={step:07}-bleu4={val/bleu4:.2f}",
            monitor=model.monitor, auto_insert_metric_name=False,
            save_top_k=1, mode="max",
        ))
        callbacks.append(EarlyStopping(
            monitor=model.monitor, verbose=True, patience=50, mode="max"
        ))
    else:
        callbacks.append(ModelCheckpoint(
            dirpath=ckptdir,
            filename="epoch={epoch:05}-step={step:07}-loss={val/contra_loss:.4f}",
            monitor=model.monitor, auto_insert_metric_name=False,
            save_top_k=1, mode="min",
        ))
        callbacks.append(EarlyStopping(
            monitor=model.monitor, verbose=True, patience=50, mode="min"
        ))
    callbacks.append(SetupCallback(
        resume=opt.resume, now=now, logdir=logdir, ckptdir=ckptdir,
        cfgdir=os.path.join(logdir, "configs"),
        config=config, lightning_config=lightning_config,
    ))

    logger = pl.loggers.TensorBoardLogger(version=nowname, save_dir=logdir)
    trainer = Trainer(**trainer_config, logger=logger, callbacks=callbacks)

    ckpt_path = ckpt if opt.resume else None
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    best_ckpt = getattr(trainer.checkpoint_callback, "best_model_path", None)
    if best_ckpt and os.path.exists(best_ckpt):
        print(f"\n[test] Loading best checkpoint: {best_ckpt}")
        import torch
        state = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(state["state_dict"])
    else:
        print("[test] No best checkpoint — using final model weights.")

    print(f"\n[test] === {Path(opt.test_csv).stem} ({opt.test_csv}) ===")
    print(f"[test]     {len(test_ds)} samples (true 3-view)")
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
