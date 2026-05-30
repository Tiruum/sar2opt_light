"""PatchNCE loss + patch-sampler MLP, ported from CUT.

Source: Park, Efros, Zhang, Zhu. "Contrastive Learning for Unpaired
Image-to-Image Translation", ECCV 2020. arXiv:2007.15651.
Reference implementation: https://github.com/taesungp/contrastive-unpaired-translation
License: BSD 2-clause. Copyright (c) Taesung Park.

Adaptations for SAR2OPT-V1 (paired translation, not unpaired):
  * Eager MLP construction in `__init__` -- avoids the original lazy
    create_mlp pattern that confuses Lightning checkpointing and EMA.
  * Per-image negatives only (`nce_includes_all_negatives_from_minibatch=False`)
    -- matches CUT default and the SEN1-2 paired regime.
  * `feature_dims` is provided up front (matches the generator backbone's
    `encoder_dims`) so MLP shapes are deterministic at module init.
  * `num_layers = len(feature_dims)` PatchNCE losses are summed and averaged
    by the caller (one wrapper class in losses.py).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["PatchNCELoss", "PatchSampleF"]


class _L2Normalize(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).sum(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / norm


class PatchSampleF(nn.Module):
    """Sample fixed-count feature patches per layer, project via MLP, L2-norm.

    Given a list of feature maps (B, C_l, H_l, W_l) the sampler:
      1. flattens spatial dims    -> (B, H_l*W_l, C_l)
      2. picks `num_patches` random indices (shared across batch for matched
         positive pairs between fake/real)
      3. projects each sampled (C_l,) vector via a per-layer 2-layer MLP
         (Linear -> ReLU -> Linear) to a common output dim `nc`
      4. L2-normalises along the channel axis

    Output: list of (B*num_patches, nc) tensors -- one per input feature map.
    The same `patch_ids` returned for the first (query) call must be passed
    to the second (key) call so positive pairs are matched.
    """

    def __init__(
        self,
        feature_dims: Sequence[int],
        nc: int = 256,
        init_gain: float = 0.02,
    ):
        super().__init__()
        self.nc = int(nc)
        self.l2norm = _L2Normalize()
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(int(d), self.nc),
                nn.ReLU(inplace=True),
                nn.Linear(self.nc, self.nc),
            )
            for d in feature_dims
        ])
        # Normal init with the CUT default gain (their `init_net` path).
        for mlp in self.mlps:
            for m in mlp:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=init_gain)
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        feats: List[torch.Tensor],
        num_patches: int = 256,
        patch_ids: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        assert len(feats) == len(self.mlps), (
            f"PatchSampleF got {len(feats)} feature maps, expected {len(self.mlps)}"
        )
        out_feats: List[torch.Tensor] = []
        out_ids: List[torch.Tensor] = []
        for layer_id, feat in enumerate(feats):
            B, C, H, W = feat.shape
            # (B, C, H, W) -> (B, H*W, C)
            feat_hw = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
            if num_patches > 0:
                if patch_ids is not None:
                    pid = patch_ids[layer_id].to(feat.device)
                else:
                    pid = torch.randperm(H * W, device=feat.device)[:min(num_patches, H * W)]
                # (B, num_patches, C) -> (B*num_patches, C)
                sampled = feat_hw[:, pid, :].reshape(B * pid.shape[0], C)
            else:
                pid = torch.empty(0, device=feat.device, dtype=torch.long)
                sampled = feat_hw.reshape(B * H * W, C)
            sampled = self.mlps[layer_id](sampled)
            sampled = self.l2norm(sampled)
            out_feats.append(sampled)
            out_ids.append(pid)
        return out_feats, out_ids


class PatchNCELoss(nn.Module):
    """InfoNCE on per-patch features (CUT, Park 2020).

    For each query patch ``q_i`` (from fake feature), there is one positive
    key ``k_i`` (the matched-position patch from real feature) and ``N-1``
    negatives (the other patches of the SAME image -- per-image negatives).

    The loss is the standard contrastive cross-entropy with temperature
    ``T`` (default 0.07 per CUT).  Key features are detached so gradient
    only flows into the query side (fake).
    """

    def __init__(self, temperature: float = 0.07, batch_size: int = 1):
        super().__init__()
        self.T = float(temperature)
        self.batch_size = int(batch_size)
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor) -> torch.Tensor:
        # feat_q, feat_k: (B*N, C) -- both L2-normalised by the sampler.
        num_patches, dim = feat_q.shape
        feat_k = feat_k.detach()

        # Positive logit: dot(q_i, k_i) -> (num_patches, 1).
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, dim),
            feat_k.view(num_patches, dim, 1),
        ).view(num_patches, 1)

        # Negative logits: per-image negatives.
        # Reshape to (B, N_per_image, C); compute (B, N, N) similarity matrix.
        N_per = num_patches // max(self.batch_size, 1)
        if N_per == 0 or num_patches % max(self.batch_size, 1) != 0:
            # Fallback: pool all negatives across the minibatch.
            B_eff, N_per = 1, num_patches
        else:
            B_eff = self.batch_size
        fq = feat_q.view(B_eff, N_per, dim)
        fk = feat_k.view(B_eff, N_per, dim)
        l_neg = torch.bmm(fq, fk.transpose(1, 2))               # (B, N, N)
        # Mask out the diagonal (q_i vs k_i) since that's already the positive.
        diag = torch.eye(N_per, device=feat_q.device, dtype=torch.bool).unsqueeze(0)
        l_neg = l_neg.masked_fill(diag, -1e4)
        l_neg = l_neg.reshape(B_eff * N_per, N_per)

        # Concatenate [pos, neg] -> (num_patches, 1 + N_per); target = 0 for all.
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return self.ce(logits, target)
