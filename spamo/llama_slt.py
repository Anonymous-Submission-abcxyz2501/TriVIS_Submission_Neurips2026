import os
import math
import random
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationMixin
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from spamo.tconv import TemporalConv
from utils.helpers import create_mask, derangement
from spamo.mm_projector import build_vision_projector
from utils.evaluate import evaluate_results
from spamo.clip_loss import clip_loss
from spamo.asb import AbstractSLT


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")

DEFAULT_LLAMA_PRETRAINED_PATH = os.environ.get(
    "VIDEOLLAMA3_7B_PATH", ""
)


class LlamaSLT(AbstractSLT):
    """SpaMo feature-fusion SLT with a LLaMA-family causal LM backend."""

    def __init__(
        self,
        tuning_type: str = "lora",
        model_name: Optional[str] = None,
        frame_sample_rate: int = 1,
        prompt: str = "",
        input_size: int = 1024,
        fusion_mode: str = "joint",
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
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.prompt = prompt
        self.model_name = model_name or DEFAULT_LLAMA_PRETRAINED_PATH
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

        self.prepare_models(self.model_name)

        if tuning_type == "freeze":
            self._freeze_model()
        elif tuning_type == "lora":
            self._apply_lora()

        self.set_container()

    def load_pretrained_weights(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])
        print(f"Checkpoint is loaded from {checkpoint_path}.")

    def _apply_lora(self) -> None:
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.llm_model = get_peft_model(self.llm_model, lora_config)
        print("LoRA adapter applied to LLaMA model.")

    def _freeze_model(self) -> None:
        self.llm_model.eval()
        for params in self.llm_model.parameters():
            params.requires_grad = False
        print("LLaMA model frozen.")

    def set_container(self) -> None:
        self.generated = []
        self.references = []

    def _resolve_model_path(self, model_name: str) -> str:
        if os.path.isfile(os.path.join(model_name, "config.json")):
            return model_name

        if os.path.isdir(model_name):
            preferred_dirs = ("videollama3_7b_local", "videollama3_7b_cslr_merged")
            for dirname in preferred_dirs:
                candidate = os.path.join(model_name, dirname)
                if os.path.isfile(os.path.join(candidate, "config.json")):
                    return candidate

            for entry in sorted(os.listdir(model_name)):
                candidate = os.path.join(model_name, entry)
                if os.path.isfile(os.path.join(candidate, "config.json")):
                    return candidate

        if model_name.startswith("meta-llama/") and os.path.isdir(DEFAULT_LLAMA_PRETRAINED_PATH):
            print(
                f"Requested gated model '{model_name}', falling back to local weights at "
                f"'{DEFAULT_LLAMA_PRETRAINED_PATH}'."
            )
            return DEFAULT_LLAMA_PRETRAINED_PATH

        return model_name

    def prepare_models(self, model_name: str) -> None:
        model_path = self._resolve_model_path(model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=self.cache_dir,
            max_length=self.max_txt_len,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                self.llm_model.resize_token_embeddings(len(self.tokenizer))

        if self.tokenizer.bos_token_id is None:
            self.bos_token_id = self.tokenizer.eos_token_id
        else:
            self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.spatio_proj = build_vision_projector("linear", self.input_size, self.inter_hidden)
        self.spatiotemp_proj = build_vision_projector("linear", 1024, self.inter_hidden)
        self.fusion_proj = build_vision_projector(
            "mlp2x_gelu", self.inter_hidden, self.llm_model.config.hidden_size
        )

        self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def _build_prompts(self, samples: Dict) -> List[str]:
        bs = len(samples["text"])
        prompts = [f"{self.prompt}"] * bs
        prompts = [p.format(l) for p, l in zip(prompts, samples["lang"])]
        if self.use_in_context:
            prompts = [f"{p} {c}" for p, c in zip(prompts, samples["ex_lang_trans"])]
        return prompts

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask.unsqueeze(-1).to(x.dtype)
        denom = w.sum(dim=1).clamp(min=1e-6)
        return (x * w).sum(dim=1) / denom

    def prepare_inputs(
        self,
        visual_outputs: torch.Tensor,
        visual_mask: torch.Tensor,
        samples: Dict,
        split: str,
        batch_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        del split, batch_idx

        prompts = self._build_prompts(samples)
        prompt_tokens = self.tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        output_tokens = self.tokenizer(
            samples["text"],
            padding="longest",
            truncation=True,
            max_length=max(self.max_txt_len - 1, 1),
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        embed_tokens = self.llm_model.get_input_embeddings()
        prompt_embeds = embed_tokens(prompt_tokens.input_ids)
        bos_embed = embed_tokens(
            torch.tensor([self.bos_token_id], device=self.device, dtype=torch.long)
        )

        joint_outputs = []
        labels = []
        seq_lengths = []

        for i in range(visual_outputs.shape[0]):
            vis_len = int(visual_mask[i].sum().item())
            prompt_len = int(prompt_tokens.attention_mask[i].sum().item())
            target_len = int(output_tokens.attention_mask[i].sum().item())

            target_ids = output_tokens.input_ids[i, :target_len]
            if self.eos_token_id is not None:
                eos_id = torch.tensor([self.eos_token_id], device=self.device, dtype=torch.long)
                target_ids = torch.cat([target_ids, eos_id], dim=0)

            target_embeds = embed_tokens(target_ids)
            sample_embeds = torch.cat(
                [
                    bos_embed,
                    visual_outputs[i, :vis_len, :],
                    prompt_embeds[i, :prompt_len, :],
                    target_embeds,
                ],
                dim=0,
            )
            joint_outputs.append(sample_embeds)

            prefix_len = 1 + vis_len + prompt_len
            sample_labels = torch.full(
                (sample_embeds.size(0),), -100, dtype=torch.long, device=self.device
            )
            sample_labels[prefix_len:] = target_ids
            labels.append(sample_labels)
            seq_lengths.append(sample_embeds.size(0))

        joint_outputs = pad_sequence(joint_outputs, batch_first=True)
        targets = pad_sequence(labels, batch_first=True, padding_value=-100)
        joint_mask = create_mask(seq_lengths=seq_lengths, device=self.device)

        return joint_outputs, joint_mask, output_tokens, targets

    def prepare_generation_inputs(
        self,
        visual_outputs: torch.Tensor,
        visual_mask: torch.Tensor,
        samples: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompts = self._build_prompts(samples)
        prompt_tokens = self.tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        embed_tokens = self.llm_model.get_input_embeddings()
        prompt_embeds = embed_tokens(prompt_tokens.input_ids)
        bos_embed = embed_tokens(
            torch.tensor([self.bos_token_id], device=self.device, dtype=torch.long)
        )

        seqs = []
        seq_lengths = []
        for i in range(visual_outputs.shape[0]):
            vis_len = int(visual_mask[i].sum().item())
            prompt_len = int(prompt_tokens.attention_mask[i].sum().item())
            sample_seq = torch.cat(
                [
                    bos_embed,
                    visual_outputs[i, :vis_len, :],
                    prompt_embeds[i, :prompt_len, :],
                ],
                dim=0,
            )
            seqs.append(sample_seq)
            seq_lengths.append(sample_seq.size(0))

        prefix_embeds = pad_sequence(seqs, batch_first=True)
        prefix_mask = create_mask(seq_lengths=seq_lengths, device=self.device)
        return prefix_embeds, prefix_mask

    def prepare_visual_inputs(self, samples: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_mode in ["joint"]:
            spatial = spatiotemporal = True
        else:
            spatial = self.fusion_mode == "spatial"
            spatiotemporal = self.fusion_mode == "spatiotemporal"

        if spatial:
            pixel_values = pad_sequence(samples["pixel_values"], batch_first=True)
            spatial_outputs = self.spatio_proj(pixel_values)
            spatial_mask = create_mask(seq_lengths=samples["num_frames"], device=self.device)

        if spatiotemporal:
            spatiotemporal_outputs = pad_sequence(samples["glor_values"], batch_first=True)
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            spatiotemporal_mask = create_mask(seq_lengths=samples["glor_lengths"], device=self.device)

        if self.fusion_mode == "joint":
            bs = spatial_outputs.shape[0]
            spatial_length = spatial_mask.sum(1)
            spatiotemporal_length = spatiotemporal_mask.sum(1)
            new_length = spatial_length + spatiotemporal_length

            joint_outputs = []
            for i in range(bs):
                valid_spatial_output = spatial_outputs[i, : spatial_length[i], :]
                valid_spatiotemporal_output = spatiotemporal_outputs[i, : spatiotemporal_length[i], :]
                concat_sample = torch.cat((valid_spatial_output, valid_spatiotemporal_output), dim=0)
                joint_outputs.append(concat_sample)
            joint_outputs = pad_sequence(joint_outputs, batch_first=True)

            visual_conv_outputs = self.temporal_encoder(
                joint_outputs.permute(0, 2, 1),
                torch.tensor(new_length.tolist(), device=self.device),
            )

            visual_outputs = visual_conv_outputs["visual_feat"].permute(1, 0, 2)
            visual_masks = create_mask(
                seq_lengths=visual_conv_outputs["feat_len"].to(torch.int).tolist(),
                device=self.device,
            )
        else:
            if spatial:
                spatial_conv_outputs = self.temporal_encoder(
                    spatial_outputs.permute(0, 2, 1),
                    torch.tensor(samples["num_frames"], device=self.device),
                )
                visual_outputs = spatial_conv_outputs["visual_feat"].permute(1, 0, 2)
                visual_masks = create_mask(
                    seq_lengths=spatial_conv_outputs["feat_len"].to(torch.int).tolist(),
                    device=self.device,
                )
            elif spatiotemporal:
                visual_outputs = spatiotemporal_outputs
                visual_masks = spatiotemporal_mask
            else:
                raise NotImplementedError("Invalid fusion mode")

        return visual_outputs, visual_masks

    def get_inputs(self, batch: List) -> Dict:
        pixel_values, glor_values, masks, ids = [], [], [], []
        texts, glosses = [], []
        num_frames, glor_lengths, langs = [], [], []
        ex_lang_translations = []

        max_frame_len = self.max_frame_len

        for sample in batch:
            if sample["pixel_value"].shape[0] != 0:
                nframe = math.ceil(sample["num_frames"] / self.frame_sample_rate)
                pval = sample["pixel_value"][:: self.frame_sample_rate]

                ids.append(sample["id"])
                texts.append(sample["text"].lower())
                glosses.append(sample["gloss"])
                langs.append(sample["lang"])

                _ex_lang_trans = [
                    f"{sample['en_text']}={sample['text']}",
                    f"{sample['fr_text']}={sample['text']}",
                    f"{sample['es_text']}={sample['text']}",
                ]
                _ex_lang_trans = _ex_lang_trans[: self.num_in_context]
                ex_lang_translations.append(" ".join(_ex_lang_trans))

                if nframe > max_frame_len:
                    nframe = max_frame_len
                    start_index = random.randint(0, pval.size(0) - max_frame_len)
                    pval = pval[start_index : start_index + max_frame_len]

                num_frames.append(nframe)
                pixel_values.append(pval)

                if sample["glor_value"] is not None:
                    if isinstance(sample["glor_value"], list):
                        glor_values.append(torch.cat(sample["glor_value"], dim=0))
                        glor_lengths.append(sum(len(g) for g in sample["glor_value"]))
                    else:
                        glor_values.append(sample["glor_value"])
                        glor_lengths.append(len(sample["glor_value"]))

        if self.use_in_context and len(set(ex_lang_translations)) > 1:
            ex_lang_translations = derangement(ex_lang_translations)


        return {
            "pixel_values": pixel_values,
            "glor_values": glor_values,
            "bool_mask_pos": masks,
            "ids": ids,
            "text": texts,
            "ex_lang_trans": ex_lang_translations,
            "gloss": glosses,
            "lang": langs,
            "num_frames": num_frames,
            "glor_lengths": glor_lengths,
        }

    def visual_textual_align(
        self, visual_outputs: torch.Tensor, visual_masks: torch.Tensor, samples: Dict
    ) -> torch.Tensor:
        output_tokens = self.tokenizer(
            samples["text"],
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        text_embeds = self.llm_model.get_input_embeddings()(output_tokens.input_ids)

        image_embeds = self._masked_mean(visual_outputs, visual_masks)
        text_embeds = self._masked_mean(text_embeds, output_tokens.attention_mask)

        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale

        return clip_loss(logits_per_text)

    def shared_step(self, inputs: Dict, split: str, batch_idx: int) -> Tuple[torch.Tensor, Dict]:
        visual_outputs, visual_masks = self.prepare_visual_inputs(inputs)
        visual_outputs = self.fusion_proj(visual_outputs)

        log_dict = {}

        if self.cross_modal_align:
            if self.warm_up_steps is None and not self.combined_loss:
                with torch.no_grad():
                    _, _, _, _ = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )

                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                log_dict[f"{split}/contra_loss"] = cont_loss
                loss = cont_loss

            elif self.warm_up_steps is not None and self.global_step <= self.warm_up_steps:
                with torch.no_grad():
                    _, _, _, _ = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )

                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                log_dict[f"{split}/contra_loss"] = cont_loss
                loss = cont_loss

            else:
                input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                    visual_outputs, visual_masks, inputs, split, batch_idx
                )

                outputs = self.llm_model(
                    inputs_embeds=input_embeds,
                    attention_mask=input_masks,
                    labels=targets,
                    output_hidden_states=True,
                    return_dict=True,
                )

                lm_loss = outputs.loss
                log_dict[f"{split}/loss"] = lm_loss

                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                loss = lm_loss + self.alpha * cont_loss

                log_dict[f"{split}/contra_loss"] = cont_loss
                log_dict[f"{split}/combined_loss"] = loss
        else:
            input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                visual_outputs, visual_masks, inputs, split, batch_idx
            )

            outputs = self.llm_model(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                labels=targets,
                output_hidden_states=True,
                return_dict=True,
            )

            loss = outputs.loss
            log_dict[f"{split}/loss"] = loss

        in_sanity_check = bool(getattr(self.trainer, "sanity_checking", False))
        if split != "train" and not in_sanity_check:
            prefix_embeds, prefix_masks = self.prepare_generation_inputs(
                visual_outputs, visual_masks, inputs
            )

            # VideoLLaMA3 overrides generate() to block inputs_embeds. Bypass all
            # model-specific overrides by calling GenerationMixin.generate directly.
            # LoRA weights stay active since get_peft_model() replaces Linear layers
            # in-place; Qwen2's forward() accepts inputs_embeds normally.
            _base = getattr(self.llm_model, "base_model", self.llm_model)
            _actual = getattr(_base, "model", _base)
            generated = GenerationMixin.generate(
                _actual,
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_masks,
                num_beams=self.beam_size,
                max_new_tokens=self.max_txt_len,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )

            generated_strings = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            generated_strings = [gen.lower().strip() for gen in generated_strings]
            reference_strings = [ref.lower().strip() for ref in inputs["text"]]

            self.generated.extend(generated_strings)
            self.references.extend(reference_strings)

        return loss, log_dict

    def on_validation_epoch_end(self) -> None:
        if bool(getattr(self.trainer, "sanity_checking", False)):
            self.set_container()
            return

        print("\n===== Validation Examples =====")
        for i in range(min(5, len(self.generated))):
            print(f"\033[94mReference: {self.references[i]}\033[0m")
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m")
            print("-" * 50)

        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split="val",
            device=self.device,
        )

        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def on_test_epoch_end(self) -> None:
        print("\n===== Validation Examples =====")
        for i in range(min(5, len(self.generated))):
            print(f"\033[94mReference: {self.references[i]}\033[0m")
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m")
            print("-" * 50)

        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split="test",
            device=self.device,
        )

        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            eps=1e-8,
            weight_decay=0.01,
            betas=(0.9, 0.98),
        )

        if hasattr(self.trainer, "estimated_stepping_batches"):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            max_epochs = self.trainer.max_epochs
            train_dataloader = self.trainer.train_dataloader
            if hasattr(train_dataloader, "dataloader"):
                train_dataloader = train_dataloader.dataloader

            batches_per_epoch = len(train_dataloader)
            total_steps = batches_per_epoch * max_epochs

            if hasattr(self.trainer, "accumulate_grad_batches"):
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
