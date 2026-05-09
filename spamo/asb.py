import torch
from typing import Dict, List, Optional, Union, Tuple

import pytorch_lightning as pl

from abc import ABC, abstractmethod
from utils.helpers import instantiate_from_config
from torch.optim.lr_scheduler import LambdaLR


class AbstractSLT(pl.LightningModule, ABC):
    """
    Abstract Sign Language Translation (SLT) Module: An abstract PyTorch Lightning module that defines a common interface
    for translating sign language from video inputs to text. Specific visual and textual models are to be defined in subclasses.
    """
    def __init__(
        self,
        lr: float = 0.0001,
        monitor: Optional[str] = None,
        scheduler_config: Optional[Dict] = None,
        max_length: int = 128,
        beam_size: int = 5,
    ):
        super().__init__()
        # Initialize module parameters
        self.lr = lr
        self.monitor = monitor
        self.scheduler_config = scheduler_config
        self.max_length = max_length
        self.beam_size = beam_size

    @abstractmethod
    def prepare_models(self) -> None:
        """
        Subclasses should implement this method to prepare the visual and textual models.
        """
        pass

    @abstractmethod
    def shared_step(self, inputs: Dict, split: str, batch_idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Implements the logic common to training, validation, and testing steps.
        
        Args:
            inputs: Dict containing input data
            split: Current split (train, val, test)
            batch_idx: Index of current batch
            
        Returns:
            Tuple of (loss, log_dict)
        """
        pass

    @abstractmethod
    def get_inputs(self, batch: List) -> Dict:
        """
        Prepares input data from a batch for processing.
        
        Args:
            batch: Raw batch data from dataloader
            
        Returns:
            Processed inputs dictionary
        """
        pass
    
    def training_step(self, batch: List, batch_idx: int) -> torch.Tensor:
        """Perform a training step."""
        inputs = self.get_inputs(batch)
        loss, log_dict = self.shared_step(inputs, "train", batch_idx)
        self.log_dict(log_dict, batch_size=len(inputs['text']), sync_dist=True)
        return loss
    
    def validation_step(self, batch: List, batch_idx: int) -> None:
        """Perform a validation step."""
        inputs = self.get_inputs(batch)
        _, log_dict = self.shared_step(inputs, "val", batch_idx)
        self.log_dict(log_dict, batch_size=len(inputs['text']), sync_dist=True)

    def test_step(self, batch: List, batch_idx: int) -> None:
        """Perform a testing step."""
        inputs = self.get_inputs(batch)
        _, log_dict = self.shared_step(inputs, "test", batch_idx)
        self.log_dict(log_dict, batch_size=len(inputs['text']), sync_dist=True)

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict]:
        """Configure the optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=1e-8)
        
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            lr_scheduler = {'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                            'interval': 'step',
                            'frequency': 1}
            return [optimizer], [lr_scheduler]
        return optimizer