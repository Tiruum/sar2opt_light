"""LLW-Former generator (llwt-v0.1.x).

Single-tensor forward: ``gen(sar: (B,1,H,W)) -> rgb: (B,3,H,W) in [-1,1]``.

Composition (top to bottom; sizes assume 256x256 input):

  1. SARPhysicsFrontEnd       (raw + log1p + Sobel)              -> 3ch
  2. StemConv                 (3 -> C_base)                       -> Cch @ 256
  3. LLWTransform (L=3)       learnable lifting                    -> {L{l}_{band}}
                                  L1 @ 128, L2 @ 64, L3 @ 32 (C ch each band)
  4. PerSubbandEncoder per l  4 separate Swin-block stacks         -> shaped bands
  5. CrossBandAttention at L2, L3 (only — L1 is too many tokens)
  6. PCSG (Physics-Conditioned Subband Gating)
  7. LLWTransform.inverse     iDWT-symmetric reconstruction        -> Cch @ 256
  8. PostDec ConvNeXtV2-GRN   2 blocks                             -> Cch @ 256
  9. HeadRGB (Conv7x7 + tanh)                                       -> 3ch @ 256

Initialisation philosophy: at step 0 the model is **identity** on its own
embedding (zero-init residuals on every Swin attention out_proj, on every MLP
final linear, on every CBA proj_out, on PCSG gate, and on the last conv of
HeadRGB).  That way the loss at epoch 0 is determined purely by the
constants (mid-grey vs target) and not by random initialisation noise.

Imports SARPhysicsFrontEnd and ConvNeXtV2GRNBlock from sarformer_wb to keep
the codebase DRY; no edits made to those files.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sarformer_wb.gen import ConvNeXtV2GRNBlock, SARPhysicsFrontEnd
from src.models.llwt.lifting import LLWTransform


__all__ = [
    "LLWFormerGenerator",
    "WindowSwinBlock",
    "PerSubbandEncoder",
    "CrossBandAttention",
    "PCSG",
]


def _g(obj, key, default):
    """Polymorphic ``obj.get(key, default)`` works for ``DictConfig``,
    ``SimpleNamespace``, plain dicts, and ``None``.

    Needed because ``OmegaConf.DictConfig`` supports both ``__contains__`` and
    ``.get()`` whereas ``SimpleNamespace`` supports neither — and tests use
    ``SimpleNamespace`` to avoid a YAML round-trip.
    """
    if obj is None:
        return default
    if hasattr(obj, 'get') and callable(obj.get):
        try:
            return obj.get(key, default)
        except Exception:
            pass
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# window partition / reverse
# ---------------------------------------------------------------------------


def window_partition(x: torch.Tensor, win: int) -> torch.Tensor:
    """``(B,C,H,W) -> (B*nh*nw, win*win, C)`` non-overlapping windows.

    Requires H and W divisible by ``win``.
    """
    B, C, H, W = x.shape
    nh, nw = H // win, W // win
    x = x.view(B, C, nh, win, nw, win)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()      # B,nh,nw,win,win,C
    return x.view(B * nh * nw, win * win, C)


def window_reverse(windows: torch.Tensor, win: int, H: int, W: int) -> torch.Tensor:
    """Inverse of ``window_partition``."""
    nh, nw = H // win, W // win
    Bt = windows.shape[0]
    B = Bt // (nh * nw)
    C = windows.shape[-1]
    x = windows.view(B, nh, nw, win, win, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()      # B,C,nh,win,nw,win
    return x.view(B, C, H, W)


# ---------------------------------------------------------------------------
# Per-subband transformer block (windowed MSA + MLP with zero-init residuals)
# ---------------------------------------------------------------------------


class WindowSwinBlock(nn.Module):
    """Pre-LN windowed-attention block.

    Layout: ``x = x + attn(LN(x_w)); x = x + MLP(LN(x))``.

    Zero-init on attention's ``out_proj`` and on the MLP's final linear make
    the block a true identity at step 0 — the network learns *into* this
    block instead of having to fight random init noise out of the way.

    ``win`` must divide H and W of inputs.
    """
    def __init__(self, dim: int, heads: int, win: int, mlp_ratio: float = 4.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.win = int(win)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        # Zero-init the MultiheadAttention output projection.  PyTorch exposes
        # this as `out_proj`; the in-projection stays at its default init.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Auto-shrink window when the feature map is smaller than the
        # configured window size: at deep wavelet levels (L3 of a 32x32
        # input -> 4x4 subband) the configured win=8 is larger than the
        # spatial extent, so we collapse to a single full-attention window.
        win = min(self.win, H, W)
        if H % win != 0 or W % win != 0:
            raise ValueError(
                f"WindowSwinBlock(win={win}) needs H,W divisible by win; "
                f"got {(H, W)}"
            )
        # ---- attention sub-layer ------------------------------------------
        win_tokens = window_partition(x, win)                       # (Bt, T, C)
        attn_in = self.norm1(win_tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        win_tokens = win_tokens + attn_out
        # ---- MLP sub-layer ------------------------------------------------
        win_tokens = win_tokens + self.mlp(self.norm2(win_tokens))
        return window_reverse(win_tokens, win, H, W)


class PerSubbandStack(nn.Module):
    """``depth``-deep stack of ``WindowSwinBlock`` for a single subband."""
    def __init__(self, dim: int, heads: int, win: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            WindowSwinBlock(dim, heads, win) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class PerSubbandEncoder(nn.Module):
    """Four parallel Swin-block stacks, one per ``{LL, LH, HL, HH}`` band.

    When ``shared=True`` the four stacks share weights — the per-subband
    specialisation then comes only from the *input distribution* of each
    band (still a valid ablation).  ``shared=False`` is the default since it
    is the more novel configuration and the parameter cost is modest.
    """
    BANDS = ('LL', 'LH', 'HL', 'HH')

    def __init__(self, dim: int, heads: int, win: int, depth: int,
                 shared: bool = False):
        super().__init__()
        if shared:
            stack = PerSubbandStack(dim, heads, win, depth)
            self.stacks = nn.ModuleDict({b: stack for b in self.BANDS})
        else:
            self.stacks = nn.ModuleDict({
                b: PerSubbandStack(dim, heads, win, depth) for b in self.BANDS
            })

    def forward(self, bands: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {b: self.stacks[b](bands[b]) for b in self.BANDS}


# ---------------------------------------------------------------------------
# Cross-Band Attention
# ---------------------------------------------------------------------------


class CrossBandAttention(nn.Module):
    """Each band's tokens attend to the concatenated tokens of the other three.

    Identity at step 0 (zero-init on the output projection) so adversarial
    pressure does not feed garbage into the per-subband encoders before the
    Q/K projections have learned a useful subspace.

    Only used at low resolutions (32x32 and 64x64) — at 128x128 the
    ``3 * H * W`` key-value sequence per band would be > 49k tokens which is
    prohibitive memory.  Caller is responsible for the level gating.
    """
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, bands: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        # Pre-flatten each band once.
        flat = {}
        for k, v in bands.items():
            B, C, H, W = v.shape
            flat[k] = v.flatten(2).transpose(1, 2)              # (B, HW, C)
        for k, v in bands.items():
            B, C, H, W = v.shape
            q = self.norm(flat[k])
            others = [flat[kk] for kk in bands if kk != k]
            kv = torch.cat(others, dim=1)                       # (B, 3*HW, C)
            kv = self.norm(kv)
            attn, _ = self.attn(q, kv, kv, need_weights=False)
            updated = flat[k] + attn
            out[k] = updated.transpose(1, 2).reshape(B, C, H, W)
        return out


# ---------------------------------------------------------------------------
# Physics-Conditioned Subband Gating
# ---------------------------------------------------------------------------


class PCSG(nn.Module):
    """Speckle-prior-driven gating on HH (damp) and LL (boost).

    Gate ``g in [0,1]`` is predicted from a 7x7 local-variance map of the raw
    SAR input (Lee-filter style speckle cue).  At step 0 the gate convs are
    zero-init -> ``g = sigmoid(0) = 0.5`` everywhere, so HH gets multiplied
    by 0.5 and LL by 1.05 — almost identity, training will reshape.

    Applied at *every* level the caller asks for.
    """
    def __init__(self, gain_ll: float = 0.1, k: int = 7):
        super().__init__()
        self.gain_ll = float(gain_ll)
        self.k = int(k)
        self.gate = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 1, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _local_var(self, sar: torch.Tensor) -> torch.Tensor:
        k = self.k
        mu  = F.avg_pool2d(sar, k, stride=1, padding=k // 2)
        mu2 = F.avg_pool2d(sar * sar, k, stride=1, padding=k // 2)
        return (mu2 - mu * mu).clamp_min(1e-6)

    def forward(self, sar_full: torch.Tensor,
                bands: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        v = self._local_var(sar_full)                           # (B,1,H,W)
        g_full = torch.sigmoid(self.gate(v))                    # (B,1,H,W)
        out: Dict[str, torch.Tensor] = {}
        for k, b in bands.items():
            gk = F.interpolate(g_full, size=b.shape[-2:], mode='bilinear',
                               align_corners=False)
            if k == 'HH':
                out[k] = b * (1.0 - gk)
            elif k == 'LL':
                out[k] = b * (1.0 + self.gain_ll * gk)
            else:
                out[k] = b
        return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class LLWFormerGenerator(nn.Module):
    """The full LLW-Former generator.

    Hyperparameters are read from ``cfg.model.gen`` (omegaconf DictConfig).
    Smoke-test friendly: also accepts ``None`` for ``cfg``, in which case
    sensible defaults are filled in (used by tests).
    """

    def __init__(self, cfg: Optional[object] = None):
        super().__init__()
        gcfg = self._defaults() if cfg is None else cfg.model.gen
        self.image_size: int = int(_g(cfg.data, 'image_size', 256)) \
            if cfg is not None else 256

        # ------ hyperparameters --------------------------------------------
        C: int        = int(_g(gcfg, 'embed_dim', 96))
        levels: int   = int(_g(gcfg, 'lifting_levels', 3))
        lift_hidden: int = int(_g(gcfg, 'lifting_hidden', 32))
        pswe          = _g(gcfg, 'pswe', None)
        win: int      = int(_g(pswe, 'window_size', 8))
        heads: int    = int(_g(pswe, 'heads', 4))
        depth: int    = int(_g(pswe, 'depth', 2))
        shared: bool  = bool(_g(pswe, 'shared_blocks', False))
        cba           = _g(gcfg, 'cba_fus', None)
        cba_levels    = list(_g(cba, 'levels', [2, 3]))
        cba_heads     = int(_g(cba, 'heads', 4))
        pcsg_node     = _g(gcfg, 'pcsg', None)
        pcsg_enabled  = bool(_g(pcsg_node, 'enabled', True))
        pcsg_gain_ll  = float(_g(pcsg_node, 'gain_ll', 0.1))

        self.levels = levels
        self.cba_levels = sorted(set(cba_levels))

        # ------ 1. Physics front-end ---------------------------------------
        self.physics_front_end = SARPhysicsFrontEnd()
        self.stem = nn.Conv2d(3, C, kernel_size=3, padding=1, bias=True)
        # Initialise stem to keep dynamic range ~1 -> the lifting sees inputs
        # similar to the SARPhysicsFrontEnd outputs (~[-1, 2]).  Standard
        # kaiming init is fine; no zero-init here because the stem is the
        # *only* path information flows through.

        # ------ 2. Lifting transform ---------------------------------------
        self.llw = LLWTransform(levels=levels, channels=C, hidden=lift_hidden)

        # ------ 3. Per-subband encoders ------------------------------------
        self.psw = nn.ModuleList([
            PerSubbandEncoder(dim=C, heads=heads, win=win, depth=depth,
                              shared=shared)
            for _ in range(levels)
        ])

        # ------ 4. Cross-band attention (only at cheap levels) -------------
        # ModuleDict keyed by level number (as string) for omegaconf-safe ser.
        self.cba = nn.ModuleDict({
            str(l): CrossBandAttention(dim=C, heads=cba_heads)
            for l in self.cba_levels if 1 <= l <= levels
        })

        # ------ 5. PCSG ----------------------------------------------------
        self.pcsg_enabled = pcsg_enabled
        if pcsg_enabled:
            self.pcsg = PCSG(gain_ll=pcsg_gain_ll)
        else:
            self.pcsg = None

        # ------ 6. Post-decoder ConvNeXt blocks ----------------------------
        self.post_dec = nn.Sequential(
            ConvNeXtV2GRNBlock(C, drop_path=0.0, expand=4),
            ConvNeXtV2GRNBlock(C, drop_path=0.0, expand=4),
        )

        # ------ 7. RGB head ------------------------------------------------
        self.head_rgb = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(C, 3, kernel_size=7),
        )
        nn.init.zeros_(self.head_rgb[-1].weight)
        nn.init.zeros_(self.head_rgb[-1].bias)

    # ----- defaults for cfg-free instantiation (smoke tests) ---------------

    @staticmethod
    def _defaults():
        from types import SimpleNamespace
        return SimpleNamespace(
            embed_dim=96,
            lifting_levels=3,
            lifting_hidden=32,
            pswe=SimpleNamespace(window_size=8, heads=4, depth=2,
                                 shared_blocks=False),
            cba_fus=SimpleNamespace(levels=[2, 3], heads=4),
            pcsg=SimpleNamespace(enabled=True, gain_ll=0.1),
            get=lambda *a, **k: None,
        )

    # ----- helpers ---------------------------------------------------------

    def _split_per_level(self, coeffs: Dict[str, torch.Tensor]):
        """Group flat coeff dict by level: ``{l: {LL,LH,HL,HH}}``."""
        return {
            l: {b: coeffs[f'L{l}_{b}'] for b in ('LL', 'LH', 'HL', 'HH')}
            for l in range(1, self.levels + 1)
        }

    def _flatten_per_level(self, per_level: Dict[int, Dict[str, torch.Tensor]]):
        flat: Dict[str, torch.Tensor] = {}
        for l, bands in per_level.items():
            for b, t in bands.items():
                flat[f'L{l}_{b}'] = t
        return flat

    # ----- forward ---------------------------------------------------------

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        # 1. Physics front-end + stem
        three = self.physics_front_end(sar)                          # (B,3,H,W)
        feat = self.stem(three)                                      # (B,C,H,W)

        # 2. Lifting transform -> coeffs dict
        coeffs = self.llw(feat)
        per_level = self._split_per_level(coeffs)

        # 3. Per-subband Swin encoding at each level
        for l in range(1, self.levels + 1):
            per_level[l] = self.psw[l - 1](per_level[l])

        # 4. Cross-band attention at requested levels (only low-res)
        for l in self.cba_levels:
            if 1 <= l <= self.levels:
                per_level[l] = self.cba[str(l)](per_level[l])

        # 5. PCSG (apply at every level; gain controlled by config)
        if self.pcsg is not None:
            for l in range(1, self.levels + 1):
                per_level[l] = self.pcsg(sar, per_level[l])

        # 6. Inverse lifting transform -> Cch @ HxW
        coeffs_out = self._flatten_per_level(per_level)
        recon_feat = self.llw.inverse(coeffs_out)                    # (B,C,H,W)

        # 7. Post-decoder
        post = self.post_dec(recon_feat)
        logits = self.head_rgb(post)
        return torch.tanh(logits)
