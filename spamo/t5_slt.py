import os
import torch
import torch.nn as nn
import random
import math
from typing import Dict, List, Optional, Tuple, Any

import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import BertConfig, BertModel
from peft import LoraConfig, get_peft_model, TaskType

from spamo.tconv import TemporalConv
from utils.helpers import create_mask, derangement
from spamo.mm_projector import build_vision_projector
from utils.evaluate import evaluate_results
from spamo.clip_loss import clip_loss
from spamo.asb import AbstractSLT
from transformers import get_cosine_schedule_with_warmup


os.environ["TOKENIZERS_PARALLELISM"] = "false"


torch.set_float32_matmul_precision('high')


class FlanT5SLT(AbstractSLT):
    """
    FlanT5-based Sign Language Translation model with multimodal capabilities.
    """
    def __init__(
        self, 
        tuning_type: str = 'lora', 
        model_name: Optional[str] = None, 
        frame_sample_rate: int = 1, 
        prompt: str = '',
        input_size: int = 1024,
        fusion_mode: str = 'joint',
        inter_hidden: int = 768,
        max_frame_len: int = 1024,
        max_txt_len: int = 64,
        cross_modal_align: bool = False,
        warm_up_steps: Optional[int] = None,
        combined_loss: bool = False,
        alpha: float = 0.1,
        use_resampler: bool = False,
        sampling_length: int = 64,
        cache_dir: str = "/data3/models",
        use_in_context: bool = False,
        num_in_context: int = 0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Configuration parameters
        self.input_size = input_size
        self.prompt = prompt
        self.model_name = model_name
        self.frame_sample_rate = frame_sample_rate
        self.fusion_mode = fusion_mode
        self.inter_hidden = inter_hidden
        self.max_frame_len = max_frame_len
        self.max_txt_len = max_txt_len
        self.tuning_type = tuning_type
        self.cross_modal_align = cross_modal_align
        self.warm_up_steps = warm_up_steps
        self.combined_loss = combined_loss
        self.alpha = alpha
        self.use_resampler = use_resampler
        self.sampling_length = sampling_length
        self.cache_dir = cache_dir
        self.use_in_context = use_in_context
        self.num_in_context = num_in_context
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        self.prepare_models(model_name)

        # Apply the selected tuning strategy
        if tuning_type == 'freeze':
            self._freeze_model()
        elif tuning_type == 'lora':
            self._apply_lora()

        self.set_container()
        
    # def load_pretrained_weights(self, checkpoint_path: str) -> None:
    #     """Load weights from a pretrained checkpoint."""
    #     checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
    #     # Get model's state dict
    #     model_state_dict = self.state_dict()
    #     checkpoint_state_dict = checkpoint['state_dict']
        
    #     # Filter out mismatched keys
    #     filtered_state_dict = {}
    #     for k, v in checkpoint_state_dict.items():
    #         if k in model_state_dict and v.size() == model_state_dict[k].size():
    #             filtered_state_dict[k] = v
        
    #     # Load the filtered state dict
    #     self.load_state_dict(filtered_state_dict)
    #     print(f'Checkpoint loaded from {checkpoint_path}. Loaded {len(filtered_state_dict)}/{len(checkpoint_state_dict)} parameters.')
    
    def load_pretrained_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        print(f'Checkpoint is loaded from {checkpoint_path}.')

    def _apply_lora(self) -> None:
        """Apply LoRA adapter to the T5 model."""
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q", "v"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.t5_model = get_peft_model(self.t5_model, lora_config)
        print("LoRA adapter applied to T5 model.")

    def _freeze_model(self) -> None:
        """Freeze the T5 model parameters."""
        self.t5_model.eval()
        for params in self.t5_model.parameters():
            params.requires_grad = False
        print("T5 model frozen.")

    def set_container(self) -> None:
        self.generated = []
        self.references = []

    def prepare_models(self, t5_model: str) -> None:
        """
        Prepare the textual and visual models.
        
        Args:
            t5_model: Name or path of the T5 model to use
        """
        
        # Load the textual model
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, 
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16, 
        )
        
        # Load the tokenizer
        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            t5_model, 
            cache_dir=self.cache_dir,
            max_length=self.max_txt_len,
        )

        # Load the vision projectors
        self.spatio_proj = build_vision_projector('linear', self.input_size, self.inter_hidden)
        self.spatiotemp_proj = build_vision_projector('linear', 1024, self.inter_hidden)
        self.fusion_proj = build_vision_projector('mlp2x_gelu', self.inter_hidden, self.t5_model.config.hidden_size)
        
        # Load the temporal encoder
        self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)

        # if self.cross_modal_align:
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def prepare_inputs(
        self, 
        visual_outputs: torch.Tensor, 
        visual_mask: torch.Tensor, 
        samples: Dict, 
        split: str, 
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        """
        Prepare combined inputs for the T5 model.
        
        Args:
            visual_outputs: Visual features
            visual_mask: Mask for visual features
            samples: Input samples
            split: Current split (train, val, test)
            batch_idx: Current batch index
            
        Returns:
            Tuple of (joint_outputs, joint_mask, output_tokens, targets)
        """
        bs = visual_outputs.shape[0]
        
        # Prepare the prompt with language information
        prompts = [f'{self.prompt}'] * bs
        prompts = [p.format(l) for p, l in zip(prompts, samples['lang'])]
        
        if self.use_in_context:
            prompts = [f"{p} {c}" for p, c in zip(prompts, samples['ex_lang_trans'])]
        
        # Tokenize prompts
        input_tokens = self.t5_tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Get lengths for visual and prompt sequences
        visual_lengths = visual_mask.sum(1)
        prompt_lengths = input_tokens.attention_mask.sum(1)
        new_lengths = visual_lengths + prompt_lengths
        
        # Convert tokens to embeddings
        input_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        
        # Concatenate visual and text embeddings
        joint_outputs = []
        for i in range(bs):
            vis_out = visual_outputs[i, :visual_lengths[i], :]
            prompt_embeds = input_embeds[i, :prompt_lengths[i], :]
            concat_sample = torch.cat((vis_out, prompt_embeds), dim=0)
            joint_outputs.append(concat_sample)
        
        # Pad the combined embeddings
        joint_outputs = pad_sequence(joint_outputs, batch_first=True)
        joint_mask = create_mask(seq_lengths=new_lengths.tolist(), device=self.device)
        
        # Tokenize target texts
        output_tokens = self.t5_tokenizer(
            samples['text'],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)
        
        # Prepare target labels (replace pad tokens with -100)
        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )
        
        return joint_outputs, joint_mask, output_tokens, targets

    def prepare_visual_inputs(self, samples: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare visual inputs based on the fusion mode.
        
        Args:
            samples: Input samples containing visual features
            
        Returns:
            Tuple of (visual_outputs, visual_masks)
        """
        # Determine which visual features to use based on fusion mode
        if self.fusion_mode in ['joint']:
            spatial = spatiotemporal = True
        else:
            spatial = self.fusion_mode == 'spatial'
            spatiotemporal = self.fusion_mode == 'spatiotemporal'

        # Process spatial features if needed
        if spatial:
            pixel_values = pad_sequence(samples['pixel_values'], batch_first=True)
            spatial_outputs = self.spatio_proj(pixel_values)
            spatial_mask = create_mask(seq_lengths=samples['num_frames'], device=self.device)
        
        # Process spatiotemporal features if needed
        if spatiotemporal:
            spatiotemporal_outputs = pad_sequence(samples['glor_values'], batch_first=True)
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            spatiotemporal_mask = create_mask(seq_lengths=samples['glor_lengths'], device=self.device)
        
        # Combine features for joint mode
        if self.fusion_mode == 'joint':
            bs = spatial_outputs.shape[0]
            spatial_length = spatial_mask.sum(1)
            spatiotemporal_length = spatiotemporal_mask.sum(1)
            new_length = spatial_length + spatiotemporal_length
            
            # Concatenate spatial and spatiotemporal features for each sample
            joint_outputs = []
            for i in range(bs):
                valid_spatial_output = spatial_outputs[i, :spatial_length[i], :]
                valid_spatiotemporal_output = spatiotemporal_outputs[i, :spatiotemporal_length[i], :]
                concat_sample = torch.cat((valid_spatial_output, valid_spatiotemporal_output), dim=0)
                joint_outputs.append(concat_sample)
            joint_outputs = pad_sequence(joint_outputs, batch_first=True)
            
            # Apply temporal encoder
            visual_conv_outputs = self.temporal_encoder(
                joint_outputs.permute(0,2,1), torch.tensor(new_length.tolist(), device=self.device)
            )
            
            visual_outputs = visual_conv_outputs['visual_feat'].permute(1,0,2)
            visual_masks = create_mask(
                seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(), 
                device=self.device
            ) 
        else:
            # Use single feature type
            if spatial:
                spatial_conv_outputs = self.temporal_encoder(
                    spatial_outputs.permute(0,2,1), torch.tensor(samples['num_frames'], device=self.device)
                )
                visual_outputs = spatial_conv_outputs['visual_feat'].permute(1,0,2)
                visual_masks = create_mask(
                    seq_lengths=spatial_conv_outputs['feat_len'].to(torch.int).tolist(), 
                    device=self.device
                )
            elif spatiotemporal:
                visual_outputs = spatiotemporal_outputs
                visual_masks = spatiotemporal_mask
            else:
                raise NotImplementedError("Invalid fusion mode")
        
        return visual_outputs, visual_masks

    def get_inputs(self, batch: List) -> Dict:
        """
        Process batch inputs into a structured dictionary.
        
        Args:
            batch: Raw batch from dataloader
            
        Returns:
            Processed inputs dictionary
        """
        pixel_values, glor_values, masks, ids = [], [], [], []
        texts, glosses = [], []
        num_frames, glor_lengths, langs = [], [], []
        ex_lang_translations = []
        
        max_frame_len = self.max_frame_len

        for sample in batch:
            if sample['pixel_value'].shape[0] != 0:
                # Calculate number of frames after sampling
                nframe = math.ceil(sample['num_frames'] / self.frame_sample_rate)
                pval = sample['pixel_value'][::self.frame_sample_rate]

                # Collect metadata
                ids.append(sample['id'])
                texts.append(sample['text'].lower())
                glosses.append(sample['gloss'])
                langs.append(sample['lang'])
                
                _ex_lang_trans = [
                    f"{sample['en_text']}={sample['text']}",
                    f"{sample['fr_text']}={sample['text']}",
                    f"{sample['es_text']}={sample['text']}"
                ]
                _ex_lang_trans = _ex_lang_trans[:self.num_in_context]
                ex_lang_translations.append(' '.join(_ex_lang_trans))
                
                # Handle too long sequences with random cropping
                if nframe > max_frame_len:
                    nframe = max_frame_len
                    start_index = random.randint(0, pval.size(0) - max_frame_len)
                    pval = pval[start_index:start_index + max_frame_len]
                
                # Store processed visual features
                num_frames.append(nframe)
                pixel_values.append(pval)
                
                # Process glor values if available
                if sample['glor_value'] is not None:
                    if isinstance(sample['glor_value'], list):
                        glor_values.append(torch.cat(sample['glor_value'], dim=0))
                        glor_lengths.append(sum(len(g) for g in sample['glor_value']))
                    else:
                        glor_values.append(sample['glor_value'])
                        glor_lengths.append(len(sample['glor_value']))
        
        ex_lang_translations = derangement(ex_lang_translations)
        
        # Return structured dictionary
        return {
            'pixel_values': pixel_values,
            'glor_values': glor_values,
            'bool_mask_pos': masks,
            'ids': ids,
            'text': texts,
            'ex_lang_trans': ex_lang_translations,
            'gloss': glosses,
            'lang': langs,
            'num_frames': num_frames,
            'glor_lengths': glor_lengths,
        }

    def visual_textual_align(self, visual_outputs: torch.Tensor, visual_masks: torch.Tensor, samples: Dict) -> torch.Tensor:
        """
        Calculate visual-textual alignment loss.
        
        Args:
            visual_outputs: Visual features
            visual_masks: Mask for visual features
            samples: Input samples
            
        Returns:
            Contrastive loss
        """
        # Tokenize target texts
        output_tokens = self.t5_tokenizer(
            samples['text'],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)
        
        # Get text embeddings
        text_embeds = self.t5_model.encoder.embed_tokens(output_tokens.input_ids)
        
        # Mean pooling for visual and text embeddings
        image_embeds = visual_outputs.mean(1)  # global pooling
        text_embeds = text_embeds.mean(1)  # global pooling
        
        # Normalize features
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Calculate cosine similarities with temperature scaling
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        # Calculate contrastive loss
        loss = clip_loss(logits_per_text)
        
        return loss

    def shared_step(self, inputs: Dict, split: str, batch_idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Shared logic for training, validation and testing steps.
        
        Args:
            inputs: Input dictionary
            split: Current split (train, val, test)
            batch_idx: Current batch index
            
        Returns:
            Tuple of (loss, log_dict)
        """
        # Prepare visual inputs and project to match text embedding dimensions
        visual_outputs, visual_masks = self.prepare_visual_inputs(inputs)
        visual_outputs = self.fusion_proj(visual_outputs)
        
        # Initialize logging dictionary
        log_dict = {}
        
        # STEP 1: Determine training mode and prepare inputs accordingly
        if self.cross_modal_align:
            # For pure contrastive learning or warm-up phase
            if self.warm_up_steps is None and not self.combined_loss:
                # Pure contrastive learning mode
                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )
                
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                log_dict[f"{split}/contra_loss"] = cont_loss
                loss = cont_loss
                
            elif self.warm_up_steps is not None and self.global_step <= self.warm_up_steps:
                # Warm-up phase with contrastive learning
                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )
                
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                log_dict[f"{split}/contra_loss"] = cont_loss
                loss = cont_loss
                
            else:
                # Combined loss mode (regular training + contrastive)
                input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                    visual_outputs, visual_masks, inputs, split, batch_idx
                )
                
                # Forward pass through T5 model
                outputs = self.t5_model(
                    inputs_embeds=input_embeds,
                    attention_mask=input_masks,
                    decoder_attention_mask=output_tokens.attention_mask,
                    labels=targets,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                t5_loss = outputs.loss
                log_dict[f"{split}/loss"] = t5_loss
                
                # Add contrastive component if using combined loss
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                loss = t5_loss + self.alpha * cont_loss
                
                log_dict[f"{split}/contra_loss"] = cont_loss
                log_dict[f"{split}/combined_loss"] = loss
        else:
            # Standard training without contrastive learning
            input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                visual_outputs, visual_masks, inputs, split, batch_idx
            )
            
            # Forward pass through T5 model
            outputs = self.t5_model(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                decoder_attention_mask=output_tokens.attention_mask,
                labels=targets,
                output_hidden_states=True,
                return_dict=True
            )
            
            loss = outputs.loss
            log_dict[f"{split}/loss"] = loss

        # STEP 2: Handle evaluation phase (validation/testing)
        if split != "train":
            # Prepare inputs for text generation
            input_embeds, input_masks, _, _ = self.prepare_inputs(
                visual_outputs, visual_masks, inputs, split, batch_idx
            )
            
            # Generate translations
            generated = self.t5_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                num_beams=5,
                max_length=self.max_txt_len,
                top_p=0.9,
                do_sample=True,
            )
            
            # Decode generated outputs and references
            generated_strings = self.t5_tokenizer.batch_decode(generated, skip_special_tokens=True)
            generated_strings = [gen.lower() for gen in generated_strings]
            
            reference_strings = self.t5_tokenizer.batch_decode(output_tokens.input_ids, skip_special_tokens=True)
            reference_strings = [ref.lower() for ref in reference_strings]

            self.generated.extend(generated_strings)
            self.references.extend(reference_strings)
            
            # Calculate evaluation metrics
            # eval_res = evaluate_results(
            #     predictions=generated_strings,
            #     references=reference_strings,
            #     split=split,
            #     tokenizer='zh' if inputs['lang'][0] == 'Chinese' else '13a',
            #     device=self.device
            # )
            
            # Add evaluation results to logging
            # log_dict.update(eval_res)

        return loss, log_dict

    def on_validation_epoch_end(self) -> None:
        # Print some examples of generated translations and references with colors
        print("\n===== Validation Examples =====")
        for i in range(min(5, len(self.generated))):
            print(f"\033[94mReference: {self.references[i]}\033[0m")  # Blue color for references
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m")    # Green color for generated
            print("-" * 50)
            
        # Calculate evaluation metrics
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='val',
            # tokenizer='zh' if outputs['lang'][0] == 'Chinese' else '13a',
            device=self.device
        )
        
        # Add evaluation results to logging
        # log_dict.update(eval_res)

        self.log_dict(eval_res, sync_dist=True)

        self.set_container()

    def on_test_epoch_end(self) -> None:
        # Print some examples of generated translations and references with colors
        print("\n===== Validation Examples =====")
        for i in range(min(5, len(self.generated))):
            print(f"\033[94mReference: {self.references[i]}\033[0m")  # Blue color for references
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m")    # Green color for generated
            print("-" * 50)
            
        # Calculate evaluation metrics
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='test',
            device=self.device
        )

        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            eps=1e-8, 
            weight_decay=0.01, 
            betas=(0.9, 0.98)
        )
        
        # Calculate total steps based on PyTorch Lightning trainer settings
        if hasattr(self.trainer, 'estimated_stepping_batches'):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            # Fallback calculation if the attribute doesn't exist
            max_epochs = self.trainer.max_epochs
            train_dataloader = self.trainer.train_dataloader
            if hasattr(train_dataloader, 'dataloader'):
                train_dataloader = train_dataloader.dataloader
            
            batches_per_epoch = len(train_dataloader)
            total_steps = batches_per_epoch * max_epochs
            
            # Account for gradient accumulation if used
            if hasattr(self.trainer, 'accumulate_grad_batches'):
                total_steps = total_steps // self.trainer.accumulate_grad_batches
        
        warmup_steps = int(total_steps * 0.1)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }