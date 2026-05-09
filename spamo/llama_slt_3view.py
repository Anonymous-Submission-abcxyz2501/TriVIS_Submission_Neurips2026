import os
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

VIEW_NAMES: Tuple[str, ...] = ("front", "left", "right")
ADAPTER_PREFIXES: Tuple[str, ...] = (
    "spatio_proj",
    "spatiotemp_proj",
    "temporal_encoder",
    "fusion_proj",
)


def _is_empty(x: Any) -> bool:
    """True for None or a Tensor with no elements (e.g. torch.tensor([]))."""
    return x is None or (isinstance(x, torch.Tensor) and x.numel() == 0)


class ViewMerge(nn.Module):
    """Merge three view features: concat along D then project [B, T, 3D] → [B, T, D].

    Zero-init: weight[:, :D] = I, weight[:, D:] = 0, bias = 0.
    When loading from a 3-view checkpoint, load_state_dict overwrites these
    zeros with trained weights. For fresh init, this yields identity (front only).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(3 * dim, dim, bias=True)
        with torch.no_grad():
            self.proj.weight[:, :dim] = torch.eye(dim)
            self.proj.weight[:, dim:] = 0.0
            self.proj.bias.zero_()

    def forward(
        self,
        vis_front: torch.Tensor,
        vis_left: torch.Tensor,
        vis_right: torch.Tensor,
        mask_left: Optional[torch.Tensor] = None,
        mask_right: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.proj(torch.cat([vis_front, vis_left, vis_right], dim=-1))


class LlamaSLTThreeView(AbstractSLT):
    """Three-view SpaMo — three per-view sign adapters share a single LLM.

    Each view gets an independent {spatio_proj, spatiotemp_proj, temporal_encoder,
    fusion_proj} chain. Per-view outputs are concatenated along time per sample
    before being fed to the shared LLM. Supports a "blind" 1-view inference mode:
    inactive views are replaced with zero-tensors of the same shape (preserving
    their lengths) so the LLM still sees the full 3-view token layout.
    """

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
        active_views: Sequence[str] = VIEW_NAMES,
        warmstart_path: Optional[str] = None,
        warmstart_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # 3-view adapter is hardcoded to joint-mode (CLIP+MAE concat per view).
        # 1-view ckpts used to warm-start were also trained exclusively with
        # fusion_mode='joint'. Raise on any other value to avoid silent drift.
        if fusion_mode != "joint":
            raise ValueError(
                f"LlamaSLTThreeView supports fusion_mode='joint' only, "
                f"got {fusion_mode!r}."
            )

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

        self._set_active_views(active_views)

        self.prepare_models(self.model_name)

        if tuning_type == "freeze":
            self._freeze_model()
        elif tuning_type == "lora":
            self._apply_lora()

        self.set_container()

        if warmstart_path:
            mode = (warmstart_mode or "").lower()
            if mode == "1view":
                self.load_warmstart_1view(warmstart_path)
            elif mode == "3view":
                self.load_warmstart_3view(warmstart_path)
            else:
                raise ValueError(
                    f"warmstart_mode must be '1view' or '3view', got: {warmstart_mode!r}"
                )

    # ------------------------------------------------------------------
    # Active-view management + warmstart loaders
    # ------------------------------------------------------------------

    def _set_active_views(self, active_views: Sequence[str]) -> None:
        if not active_views:
            raise ValueError("active_views must contain at least one of front/left/right")
        chosen = tuple(v.lower() for v in active_views)
        for v in chosen:
            if v not in VIEW_NAMES:
                raise ValueError(f"Unknown view '{v}'. Must be one of {VIEW_NAMES}.")
        self.active_views = chosen

    def set_active_views(self, active_views: Sequence[str]) -> None:
        """Runtime toggle — call before evaluation to pick blind-mode views."""
        self._set_active_views(active_views)

    def load_pretrained_weights(self, checkpoint_path: str) -> None:
        """Load a full 3-view checkpoint. Kept for parity with main.py."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        self.load_state_dict(sd)
        print(f"Checkpoint loaded from {checkpoint_path}.")

    def load_warmstart_1view(self, path: str) -> None:
        """Replicate a 1-view SpaMo ckpt's adapter keys across all 3 views."""
        checkpoint = torch.load(path, map_location="cpu")
        sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        remap: Dict[str, torch.Tensor] = {}
        adapter_keys_seen = set()
        for k, v in sd.items():
            matched_prefix = None
            for prefix in ADAPTER_PREFIXES:
                if k.startswith(prefix + "."):
                    matched_prefix = prefix
                    break
            if matched_prefix is not None:
                rest = k[len(matched_prefix) + 1:]
                adapter_keys_seen.add(k)
                for view in VIEW_NAMES:
                    remap[f"{matched_prefix}_{view}.{rest}"] = v.clone()
            else:
                remap[k] = v

        missing, unexpected = self.load_state_dict(remap, strict=False)
        n_head_replicated = len(adapter_keys_seen) * len(VIEW_NAMES)
        n_lora = sum(1 for k in remap if ".lora_" in k)
        print(
            f"[warmstart:1view] adapter_src_keys={len(adapter_keys_seen)} "
            f"replicated={n_head_replicated} LoRA={n_lora} "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"from={path}"
        )

    def load_warmstart_3view(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"[warmstart:3view] missing={len(missing)} unexpected={len(unexpected)} "
            f"from={path}"
        )

    # ------------------------------------------------------------------
    # LLM / adapter setup
    # ------------------------------------------------------------------

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

        llm_hidden = self.llm_model.config.hidden_size
        for view in VIEW_NAMES:
            setattr(
                self,
                f"spatio_proj_{view}",
                build_vision_projector("linear", self.input_size, self.inter_hidden),
            )
            setattr(
                self,
                f"spatiotemp_proj_{view}",
                build_vision_projector("linear", 1024, self.inter_hidden),
            )
            setattr(
                self,
                f"temporal_encoder_{view}",
                TemporalConv(self.inter_hidden, self.inter_hidden),
            )
            setattr(
                self,
                f"fusion_proj_{view}",
                build_vision_projector("mlp2x_gelu", self.inter_hidden, llm_hidden),
            )

        # Concat 3 views along feature dim then project to LLM hidden.
        self.view_merge = ViewMerge(dim=llm_hidden)

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    # ------------------------------------------------------------------
    # Prompt / LM input assembly (identical to 1-view LlamaSLT)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Per-view adapter + cross-view concat
    # ------------------------------------------------------------------

    def _adapter_forward(
        self,
        view: str,
        pixel_values: List[torch.Tensor],
        num_frames: List[int],
        glor_values: List[torch.Tensor],
        glor_lengths: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatio_proj = getattr(self, f"spatio_proj_{view}")
        spatiotemp_proj = getattr(self, f"spatiotemp_proj_{view}")
        temporal_encoder = getattr(self, f"temporal_encoder_{view}")
        fusion_proj = getattr(self, f"fusion_proj_{view}")

        spatial_padded = pad_sequence(pixel_values, batch_first=True)
        spatial_outputs = spatio_proj(spatial_padded)
        spatial_mask = create_mask(seq_lengths=num_frames, device=self.device)

        spatiotemp_padded = pad_sequence(glor_values, batch_first=True)
        spatiotemporal_outputs = spatiotemp_proj(spatiotemp_padded)
        spatiotemporal_mask = create_mask(seq_lengths=glor_lengths, device=self.device)

        bs = spatial_outputs.shape[0]
        spatial_length = spatial_mask.sum(1)
        spatiotemporal_length = spatiotemporal_mask.sum(1)
        new_length = spatial_length + spatiotemporal_length

        joint_outputs = []
        for i in range(bs):
            valid_spatial = spatial_outputs[i, : spatial_length[i], :]
            valid_spatiotemp = spatiotemporal_outputs[i, : spatiotemporal_length[i], :]
            joint_outputs.append(torch.cat([valid_spatial, valid_spatiotemp], dim=0))
        joint_outputs = pad_sequence(joint_outputs, batch_first=True)

        conv_out = temporal_encoder(
            joint_outputs.permute(0, 2, 1),
            torch.tensor(new_length.tolist(), device=self.device),
        )
        visual = conv_out["visual_feat"].permute(1, 0, 2)
        vmask = create_mask(
            seq_lengths=conv_out["feat_len"].to(torch.int).tolist(),
            device=self.device,
        )

        visual = fusion_proj(visual)
        return visual, vmask

    def prepare_visual_inputs(
        self, samples: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        per_view: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for v in VIEW_NAMES:
            per_view[v] = self._adapter_forward(
                v,
                samples[f"pixel_values_{v}"],
                samples[f"num_frames_{v}"],
                samples[f"glor_values_{v}"],
                samples[f"glor_lengths_{v}"],
            )

        vis_front, mask_front = per_view["front"]
        vis_left, mask_left = per_view["left"]
        vis_right, mask_right = per_view["right"]

        # Align side views to T_front. Frames are synchronized across cameras
        # so in practice T_left == T_right == T_front, but temporal_encoder
        # can yield ±1 if spatial+spatiotemp concat rounds differently per
        # view. Pad with zeros (view_merge([front_t, 0, 0]) == front_t at
        # init, so trailing pad is harmless).
        T_ref = vis_front.size(1)

        def _align(x: torch.Tensor, m: torch.Tensor, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
            if x.size(1) == T:
                return x, m
            if x.size(1) < T:
                x_out = F.pad(x, (0, 0, 0, T - x.size(1)))
                m_out = F.pad(m, (0, T - m.size(1)), value=False)
            else:
                x_out = x[:, :T, :]
                m_out = m[:, :T]
            return x_out, m_out

        vis_left, mask_left = _align(vis_left, mask_left, T_ref)
        vis_right, mask_right = _align(vis_right, mask_right, T_ref)

        # Zero-out padded side timesteps so view_merge sees clean zeros there.
        vis_left = vis_left * mask_left.unsqueeze(-1).to(vis_left.dtype)
        vis_right = vis_right * mask_right.unsqueeze(-1).to(vis_right.dtype)

        fused = self.view_merge(
            vis_front, vis_left, vis_right,
            mask_left=mask_left, mask_right=mask_right,
        )

        per_view_aligned = {
            "front": (vis_front, mask_front),
            "left": (vis_left, mask_left),
            "right": (vis_right, mask_right),
        }
        return fused, mask_front, per_view_aligned

    # ------------------------------------------------------------------
    # Batch collation — per-view buckets + blind-mode zero masking
    # ------------------------------------------------------------------

    def get_inputs(self, batch: List) -> Dict:
        texts, glosses, ids, langs = [], [], [], []
        ex_lang_translations: List[str] = []

        per_view: Dict[str, Dict[str, list]] = {
            v: {
                "pixel_values": [],
                "glor_values": [],
                "num_frames": [],
                "glor_lengths": [],
            }
            for v in VIEW_NAMES
        }

        max_frame_len = self.max_frame_len

        for sample in batch:
            sid = sample.get("id", "?")

            # Dataset fail-fast guarantees all 6 feature files exist. Validate
            # the in-memory tensors are non-empty and fail loud if not — we'd
            # rather crash than silently train on zero features.
            for v in VIEW_NAMES:
                pval = sample.get(f"pixel_value_{v}")
                gval = sample.get(f"glor_value_{v}")
                if _is_empty(pval):
                    raise ValueError(f"Empty CLIP feature for {v}/{sid}")
                if _is_empty(gval) or (isinstance(gval, list) and len(gval) == 0):
                    raise ValueError(f"Empty MAE feature for {v}/{sid}")

            ids.append(sid)
            texts.append(sample["text"].lower())
            glosses.append(sample["gloss"])
            langs.append(sample["lang"])

            _ex = [
                f"{sample.get('en_text', sample['text'])}={sample['text']}",
                f"{sample.get('fr_text', sample['text'])}={sample['text']}",
                f"{sample.get('es_text', sample['text'])}={sample['text']}",
            ]
            _ex = _ex[: self.num_in_context]
            ex_lang_translations.append(" ".join(_ex))

            for v in VIEW_NAMES:
                pval = sample[f"pixel_value_{v}"]
                gval = sample[f"glor_value_{v}"]

                nframe = math.ceil(pval.shape[0] / self.frame_sample_rate)
                pval = pval[:: self.frame_sample_rate]
                if nframe > max_frame_len:
                    nframe = max_frame_len
                    start = random.randint(0, pval.size(0) - max_frame_len)
                    pval = pval[start : start + max_frame_len]

                if isinstance(gval, list):
                    gval_cat = torch.cat(gval, dim=0)
                else:
                    gval_cat = gval
                glen = int(gval_cat.shape[0])

                if v not in self.active_views:
                    pval = torch.zeros_like(pval)
                    gval_cat = torch.zeros_like(gval_cat)

                per_view[v]["pixel_values"].append(pval)
                per_view[v]["num_frames"].append(nframe)
                per_view[v]["glor_values"].append(gval_cat)
                per_view[v]["glor_lengths"].append(glen)

        if self.use_in_context and len(set(ex_lang_translations)) > 1:
            ex_lang_translations = derangement(ex_lang_translations)

        out: Dict[str, Any] = {
            "ids": ids,
            "text": texts,
            "gloss": glosses,
            "lang": langs,
            "ex_lang_trans": ex_lang_translations,
            "bool_mask_pos": [],
        }
        for v in VIEW_NAMES:
            for k, lst in per_view[v].items():
                out[f"{k}_{v}"] = lst
        return out

    # ------------------------------------------------------------------
    # Contrastive alignment + shared step (no outer fusion_proj)
    # ------------------------------------------------------------------

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

    def _per_view_contrastive(
        self, per_view: Dict[str, Tuple[torch.Tensor, torch.Tensor]], inputs: Dict
    ) -> torch.Tensor:
        losses = [
            self.visual_textual_align(vis, mask, inputs)
            for (vis, mask) in per_view.values()
        ]
        return sum(losses) / len(losses)

    def shared_step(self, inputs: Dict, split: str, batch_idx: int) -> Tuple[torch.Tensor, Dict]:
        # fusion_proj_{view} is applied inside _adapter_forward — no outer projection.
        visual_outputs, visual_masks, per_view = self.prepare_visual_inputs(inputs)

        log_dict: Dict[str, torch.Tensor] = {}

        if self.cross_modal_align:
            if self.warm_up_steps is None and not self.combined_loss:
                with torch.no_grad():
                    _, _, _, _ = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )

                cont_loss = self._per_view_contrastive(per_view, inputs)
                log_dict[f"{split}/contra_loss"] = cont_loss
                loss = cont_loss

            elif self.warm_up_steps is not None and self.global_step <= self.warm_up_steps:
                with torch.no_grad():
                    _, _, _, _ = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )

                cont_loss = self._per_view_contrastive(per_view, inputs)
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

                cont_loss = self._per_view_contrastive(per_view, inputs)
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

    # ------------------------------------------------------------------
    # Epoch-end hooks + optimizer (identical to 1-view)
    # ------------------------------------------------------------------

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
        print("\n===== Test Examples =====")
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

    def _build_param_groups(self) -> List[Dict[str, Any]]:
        """Differential LR: LoRA + front adapter at base, side adapters 2x,
        view_merge 3x (fresh module). logit_scale goes in base group."""
        side_keys = ("_left", "_right")
        groups: Dict[str, List[nn.Parameter]] = {
            "base": [],         # LoRA + front adapter + logit_scale
            "side_adapter": [], # left/right adapter chains
            "view_merge": [],   # token-wise fusion linear
        }
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("view_merge."):
                groups["view_merge"].append(p)
            elif any(key in name for key in side_keys):
                groups["side_adapter"].append(p)
            else:
                groups["base"].append(p)

        lr = self.lr
        param_groups = [
            {"params": groups["base"],         "lr": lr * 1.0, "name": "base"},
            {"params": groups["side_adapter"], "lr": lr * 2.0, "name": "side_adapter"},
            {"params": groups["view_merge"],   "lr": lr * 3.0, "name": "view_merge"},
        ]
        n_base = sum(p.numel() for p in groups["base"]) / 1e6
        n_side = sum(p.numel() for p in groups["side_adapter"]) / 1e6
        n_merge = sum(p.numel() for p in groups["view_merge"]) / 1e6
        print(
            f"[param_groups] base={n_base:.1f}M@{lr*1.0:.2e}  "
            f"side={n_side:.1f}M@{lr*2.0:.2e}  "
            f"view_merge={n_merge:.1f}M@{lr*3.0:.2e}"
        )
        return param_groups

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._build_param_groups(),
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
