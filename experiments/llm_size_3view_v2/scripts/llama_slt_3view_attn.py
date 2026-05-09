"""LlamaSLT3ViewAttnPool — 3-view fusion with view-embedding + per-timestep attn pool.

Replaces `LlamaSLTThreeView.view_merge` (Linear[3D→D] feature concat) with:
    1. view_embed: nn.Embedding(3, D_LLM), zero-init  → identifies view at every timestep.
    2. attn_query : learnable [D_LLM] vector, zero-init → softmax weights = uniform 1/3.

Init effect: at epoch 0, fused = mean(front, left, right). Optimizer breaks symmetry.

Training/eval/get_inputs/prepare_inputs all inherited from LlamaSLTThreeView unchanged
(true 3-view per forward pass, time-aligned to T_front).
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spamo.llama_slt_3view import LlamaSLTThreeView, VIEW_NAMES


class LlamaSLT3ViewAttnPool(LlamaSLTThreeView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        D = self.llm_model.config.hidden_size
        if hasattr(self, "view_merge"):
            del self.view_merge
        self.view_embed = nn.Embedding(len(VIEW_NAMES), D)
        nn.init.zeros_(self.view_embed.weight)
        self.attn_query = nn.Parameter(torch.zeros(D))
        self._last_attn_weights: Optional[torch.Tensor] = None  # [B, T, 3] cpu
        self._last_vis_mask: Optional[torch.Tensor] = None       # [B, T] cpu
        print(f"[LlamaSLT3ViewAttnPool] D_LLM={D}, view_embed=Emb({len(VIEW_NAMES)},{D}), attn_query=[{D}]")

    @staticmethod
    def _align(
        x: torch.Tensor, m: torch.Tensor, T: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.size(1) == T:
            return x, m
        if x.size(1) < T:
            x = F.pad(x, (0, 0, 0, T - x.size(1)))
            m = F.pad(m, (0, T - m.size(1)), value=False)
        else:
            x = x[:, :T, :]
            m = m[:, :T]
        return x, m

    def prepare_visual_inputs(
        self, samples: Dict[str, Any]
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

        vis_f, mask_f = per_view["front"]
        vis_l, mask_l = per_view["left"]
        vis_r, mask_r = per_view["right"]

        T_ref = vis_f.size(1)
        vis_l, mask_l = self._align(vis_l, mask_l, T_ref)
        vis_r, mask_r = self._align(vis_r, mask_r, T_ref)

        vis_l = vis_l * mask_l.unsqueeze(-1).to(vis_l.dtype)
        vis_r = vis_r * mask_r.unsqueeze(-1).to(vis_r.dtype)

        ve = self.view_embed.weight.to(vis_f.dtype)  # [3, D]
        vis_f_id = vis_f + ve[0]
        vis_l_id = vis_l + ve[1]
        vis_r_id = vis_r + ve[2]

        # Stack along view-axis → [B, T, 3, D]
        X = torch.stack([vis_f_id, vis_l_id, vis_r_id], dim=2)

        D = X.size(-1)
        q = self.attn_query.to(X.dtype).view(1, 1, 1, D)
        scores = (X * q).sum(dim=-1, keepdim=True) / (D ** 0.5)  # [B, T, 3, 1]
        weights = F.softmax(scores, dim=2)
        self._last_attn_weights = weights.squeeze(-1).detach().cpu()  # [B, T, 3]
        self._last_vis_mask = mask_f.detach().cpu()                   # [B, T]
        fused = (weights * X).sum(dim=2)  # [B, T, D]

        per_view_aligned = {
            "front": (vis_f, mask_f),
            "left": (vis_l, mask_l),
            "right": (vis_r, mask_r),
        }
        return fused, mask_f, per_view_aligned

    def _build_param_groups(self) -> List[Dict[str, Any]]:
        """Differential LR groups: LoRA+front=1×, side adapters=2×, fusion modules=3×.

        Replaces the base class's `view_merge` group with two new groups:
          - view_embed (learnable view identity)
          - attn_query (attention pool query vector)
        """
        side_keys = ("_left", "_right")
        groups: Dict[str, List[nn.Parameter]] = {
            "base": [],          # LoRA + front adapter + logit_scale
            "side_adapter": [],  # left/right adapter chains
            "view_embed": [],
            "attn_query": [],
        }
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("view_embed."):
                groups["view_embed"].append(p)
            elif name == "attn_query":
                groups["attn_query"].append(p)
            elif any(key in name for key in side_keys):
                groups["side_adapter"].append(p)
            else:
                groups["base"].append(p)

        lr = self.lr
        param_groups = [
            {"params": groups["base"],         "lr": lr * 1.0, "name": "base"},
            {"params": groups["side_adapter"], "lr": lr * 2.0, "name": "side_adapter"},
            {"params": groups["view_embed"],   "lr": lr * 3.0, "name": "view_embed"},
            {"params": groups["attn_query"],   "lr": lr * 3.0, "name": "attn_query"},
        ]
        n_base = sum(p.numel() for p in groups["base"]) / 1e6
        n_side = sum(p.numel() for p in groups["side_adapter"]) / 1e6
        n_ve = sum(p.numel() for p in groups["view_embed"]) / 1e6
        n_aq = sum(p.numel() for p in groups["attn_query"]) / 1e6
        print(
            f"[param_groups] base={n_base:.1f}M@{lr*1.0:.2e}  "
            f"side={n_side:.1f}M@{lr*2.0:.2e}  "
            f"view_embed={n_ve:.3f}M@{lr*3.0:.2e}  "
            f"attn_query={n_aq:.4f}M@{lr*3.0:.2e}"
        )
        return param_groups
