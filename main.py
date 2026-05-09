import argparse
import datetime
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.trainer import Trainer

from utils.helpers import instantiate_from_config
from spamo.callbacks import SetupCallback


def str2bool(v: Any) -> bool:
    """Convert string representation to boolean.
    
    Args:
        v: Input value to convert
        
    Returns:
        Boolean representation of input
        
    Raises:
        ArgumentTypeError: If input cannot be interpreted as boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser() -> argparse.ArgumentParser:
    """Create argument parser with all required CLI options.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(description='SpaMo training and evaluation')
    parser.add_argument(
        '-c', '--config', nargs='*', metavar='base_config.yaml', default=list(),
        help='Configuration files to load'
    )
    parser.add_argument(
        '-t', '--train', type=str2bool, default=True, nargs='?',
        help='Run in training mode'
    )
    parser.add_argument(
        '--test', type=bool, default=False,
        help='Run in testing mode'
    )
    parser.add_argument(
        '-s', '--seed', type=int, default=0,
        help='Seed for random number generators'
    )
    parser.add_argument(
        '-f', '--fast_dev_run', action='store_true', default=False,
        help='Run a test batch for debugging'
    )
    parser.add_argument(
        '-n', '--name', type=str, const=True, default='', nargs='?',
        help='Postfix for log directory'
    )
    parser.add_argument(
        '--postfix', type=str, default='',
        help='Additional postfix for log directory'
    )
    parser.add_argument(
        '-l', '--logdir', type=str, default='logs',
        help='Base directory for logging'
    )
    parser.add_argument(
        '-r', '--resume', default=None,
        help='Resume training from checkpoint directory'
    )
    parser.add_argument(
        '--no_test', type=bool, default=True,
        help='Skip test phase after training'
    )
    parser.add_argument(
        '--ckpt', type=str, default=None,
        help='Checkpoint file for resuming or testing'
    )
    parser.add_argument(
        '-e', '--evaluation', type=str, default='mse',
        help='Evaluation metric to use'
    )
    return parser


def load_configs(config_paths: List[str]) -> OmegaConf:
    """Load and merge multiple configuration files.
    
    Args:
        config_paths: List of paths to configuration files
        
    Returns:
        Merged configuration
    """
    configs = [OmegaConf.load(cfg) for cfg in config_paths]
    return OmegaConf.merge(*configs)


def setup_logging_dirs(opt: argparse.Namespace) -> tuple:
    """Set up logging directories and determine checkpoint path.
    
    Args:
        opt: Command line arguments
        
    Returns:
        Tuple of (logdir, checkpoint_path, nowname)
        
    Raises:
        ValueError: If resuming from a non-existent directory
    """
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f"Cannot find checkpoint directory: {opt.resume}")
            
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", opt.ckpt) if opt.ckpt else None
        nowname = logdir.split("/")[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.config:
            cfg_fname = os.path.split(opt.config[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)
        ckpt = opt.ckpt
    
    return logdir, ckpt, nowname


def configure_callbacks(
    opt: argparse.Namespace, 
    model: pl.LightningModule, 
    ckptdir: str, 
    lightning_config: OmegaConf,
    logdir: str,
    now: str,
    config: OmegaConf
) -> List:
    """Configure training callbacks.
    
    Args:
        opt: Command line arguments
        model: Lightning module
        ckptdir: Directory for checkpoints
        lightning_config: Lightning configuration
        logdir: Directory for logs
        now: Current timestamp
        config: Full configuration
        
    Returns:
        List of callbacks
    """
    callbacks = [
        instantiate_from_config(lightning_config.callback[callback]) 
        for callback in lightning_config.callback.keys()
    ]
    
    # Add checkpointing and early stopping based on metric type
    if opt.evaluation == "bleu":
        callbacks.append(ModelCheckpoint(
            dirpath=ckptdir, 
            filename="epoch={epoch:05}-step={step:07}-bleu4={val/bleu4:.2f}", 
            monitor=model.monitor, 
            auto_insert_metric_name=False, 
            save_top_k=1, 
            mode="max"
        ))
        callbacks.append(EarlyStopping(
            monitor=model.monitor, verbose=True, patience=200, mode="max"
        ))
    else:
        callbacks.append(ModelCheckpoint(
            dirpath=ckptdir, 
            filename="epoch={epoch:05}-step={step:07}-loss={val/contra_loss:.4f}", 
            monitor=model.monitor, 
            auto_insert_metric_name=False, 
            save_top_k=1, 
            mode="min"
        ))
        callbacks.append(EarlyStopping(
            monitor=model.monitor, verbose=True, patience=50, mode="min"
        ))
    
    # Setup callback for logging configuration
    callbacks.append(SetupCallback(
        resume=opt.resume, 
        now=now, 
        logdir=logdir, 
        ckptdir=ckptdir, 
        cfgdir=os.path.join(logdir, "configs"),
        config=config, 
        lightning_config=lightning_config
    ))
    
    return callbacks


def configure_logger(logdir: str, nowname: str) -> Dict:
    """Configure the logger.
    
    Args:
        logdir: Directory for logs
        nowname: Name for the current run
        
    Returns:
        Logger configuration
    """
    return {
        "target": "pytorch_lightning.loggers.TensorBoardLogger",
        "params": {
            "version": nowname,
            "save_dir": logdir
        }
    }


def main():
    """Main entry point for training and testing."""
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    
    # Parse arguments
    parser = get_parser()
    opt, _ = parser.parse_known_args()
    
    # Validate arguments
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both. "
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    
    # Set up directories and checkpoint path
    logdir, ckpt, nowname = setup_logging_dirs(opt)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    # Set random seed for reproducibility
    seed_everything(opt.seed)
    
    # Load configuration files
    if opt.resume or opt.test:
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.config = base_configs + opt.config
        
    config = load_configs(opt.config)
    lightning_config = config.pop("lightning", OmegaConf.create())
    
    # Configure trainer
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    if opt.fast_dev_run:
        trainer_config["fast_dev_run"] = True
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config
    
    # Instantiate data module
    data = instantiate_from_config(config.data)
    data.setup()
    
    # Instantiate model
    model = instantiate_from_config(config.model)
    
    # Configure trainer with callbacks and logger for non-dev runs
    if not opt.fast_dev_run:
        logger_cfg = configure_logger(logdir, nowname)
        trainer_opt.logger = instantiate_from_config(logger_cfg)
        
        trainer_opt.callbacks = configure_callbacks(
            opt, model, ckptdir, lightning_config, logdir, now, config
        )
    
    # Create trainer
    trainer = Trainer(**vars(trainer_opt))
    
    # Run training or testing
    if opt.train:
        if opt.resume is not None:
            trainer.fit(model, data, ckpt_path=ckpt)
        else:
            if ckpt is not None:
                model.load_pretrained_weights(ckpt)
                trainer.fit(model, data)
            else:
                trainer.fit(model, data)
            
            if not opt.no_test:
                trainer.test(model, data)
    elif opt.test:
        trainer.test(model, data, ckpt_path=ckpt)


if __name__ == '__main__':
    main()
