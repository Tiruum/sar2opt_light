"""SAR-Conditional Diffusion Refinement (SAR-DR) for LLW-Former v0.5.x.

Lightweight UNet diffusion model that refines a frozen GAN generator's coarse
output via few-step DDIM sampling.  Designed to push residual FID below the
GAN-only plateau (~81 for v4 R2) by leveraging diffusion's distribution-
matching guarantees, without losing GAN inference speed.

Pipeline at inference:

    coarse = G(SAR)                                 # 1 forward, ~40 ms
    x_T = sqrt(alpha_bar_T) * coarse + sqrt(1 - alpha_bar_T) * eps   # warm init
    for t in DDIM_SCHEDULE:                         # 3 steps
        eps_pred = SAR_DR(SAR, coarse, x_t, t)
        x_prev   = ddim_step(x_t, eps_pred, t, t_prev)
    return x_0

Total inference cost = 1 G forward + 3 SAR-DR forwards.  Comparable to GAN
inference (still sub-second for a single image).

Public exports:

    * ``SARDRefiner``       — small UNet, predicts noise eps_t conditioned on
                              [SAR, coarse, x_t] + sinusoidal timestep emb.
    * ``LinearDiffusion``   — schedule + forward/reverse step utilities.
    * ``ddim_sample``       — 3-step (or N-step) deterministic DDIM sampler.

Run smoke from repo root::

    python -m src.models.llwt_v5.diffusion
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time embedding (sinusoidal, GoogleNet-style).
# ---------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, f"SinusoidalTimeEmbedding dim must be even, got {dim}"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float or long
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)            # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)   # (B, dim)


# ---------------------------------------------------------------------------
# Building blocks: time-conditioned ResBlock and a small self-attention block.
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Two-conv ResBlock with GroupNorm + SiLU + injected time embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        groups = min(groups, in_ch) if in_ch >= 1 else 1
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        out_groups = min(8, out_ch)
        self.norm2 = nn.GroupNorm(out_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over flattened spatial tokens."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)                                              # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)
        # reshape to (B, heads, head_dim, H*W)
        def split(t):
            return t.view(B, self.num_heads, self.head_dim, H * W)
        q, k, v = split(q), split(k), split(v)
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)                 # (B, heads, head_dim, N)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# SAR-DR UNet: 5 downsample + 5 upsample levels with skips.  Self-attention
# at the bottleneck (16x16) and the level above (32x32) for some texture
# context without blowing up VRAM.
# ---------------------------------------------------------------------------


class SARDRefiner(nn.Module):
    """Lightweight UNet for SAR-conditional diffusion refinement.

    Input concatenated along channel dim:

        SAR      (B, sar_ch,    H, W)
        coarse   (B, 3,         H, W)
        x_t      (B, 3,         H, W)

    Total: ``in_ch = sar_ch + 3 + 3`` (default = 7 for sar_ch=1).

    Output:

        eps_pred (B, 3, H, W)  — noise prediction at timestep t.
    """

    def __init__(
        self,
        sar_channels: int = 1,
        base_dim: int = 64,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 6, 8),   # (64, 128, 256, 384, 512)
        time_dim: int = 256,
        attn_levels: Tuple[int, ...] = (3, 4),          # apply attn at idx 3 (32x32) and 4 (16x16)
    ):
        super().__init__()
        self.sar_channels = sar_channels
        self.in_channels  = sar_channels + 3 + 3        # SAR + coarse + x_t

        # Time embedding -> shared MLP for all ResBlocks.
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        dims = [base_dim * m for m in ch_mult]
        self.attn_levels = set(attn_levels)

        # Initial projection: in_ch -> base_dim
        self.in_proj = nn.Conv2d(self.in_channels, base_dim, kernel_size=3, padding=1)

        # Encoder: ResBlock + (attn) + Downsample at each level.
        self.encoder = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_dim = base_dim
        for idx, dim in enumerate(dims):
            block = nn.ModuleList([
                ResBlock(prev_dim, dim, time_dim),
                ResBlock(dim, dim, time_dim),
            ])
            if idx in self.attn_levels:
                block.append(SelfAttention(dim))
            self.encoder.append(block)
            # Downsample after each level except the last (last is bottleneck)
            if idx < len(dims) - 1:
                self.downsamples.append(nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1))
            else:
                self.downsamples.append(nn.Identity())
            prev_dim = dim

        # Bottleneck = last encoder level already includes attn.

        # Decoder: mirror of encoder, with skip concatenation.
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for idx, dim in enumerate(reversed(dims)):
            real_idx = len(dims) - 1 - idx
            # Upsample BEFORE block at each level except the very first decoder block.
            if real_idx < len(dims) - 1:
                self.upsamples.append(
                    nn.ConvTranspose2d(prev_dim, dim, kernel_size=4, stride=2, padding=1)
                )
            else:
                self.upsamples.append(nn.Identity())
            # Input to block = upsampled feat + skip concat.
            block_in = dim * 2 if real_idx < len(dims) - 1 else dim
            block = nn.ModuleList([
                ResBlock(block_in, dim, time_dim),
                ResBlock(dim, dim, time_dim),
            ])
            if real_idx in self.attn_levels:
                block.append(SelfAttention(dim))
            self.decoder.append(block)
            prev_dim = dim

        # Output projection -> 3 channels (eps prediction).
        out_groups = min(8, base_dim)
        self.out_norm = nn.GroupNorm(out_groups, base_dim)
        self.out_conv = nn.Conv2d(base_dim, 3, kernel_size=3, padding=1)
        # Zero-init output so initial eps_pred = 0 (stable diffusion start).
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        sar: torch.Tensor,
        coarse: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise eps at timestep t.

        Args:
            sar:    (B, sar_ch, H, W)
            coarse: (B, 3, H, W)         — frozen G output for this SAR
            x_t:    (B, 3, H, W)         — noisy state at step t
            t:      (B,)                 — timestep tensor (int or float)

        Returns:
            eps_pred: (B, 3, H, W)
        """
        t_emb = self.time_embed(t)                                       # (B, time_dim)

        # Concatenate conditioning + noisy state.
        x = torch.cat([sar, coarse, x_t], dim=1)                          # (B, in_ch, H, W)
        h = self.in_proj(x)

        # Encoder forward, save skips.
        skips: List[torch.Tensor] = []
        for level_idx, (block, downsample) in enumerate(zip(self.encoder, self.downsamples)):
            for module in block:
                if isinstance(module, ResBlock):
                    h = module(h, t_emb)
                else:                                                     # SelfAttention
                    h = module(h)
            skips.append(h)
            h = downsample(h)

        # Decoder forward, pop skips in reverse.
        for level_idx, (block, upsample) in enumerate(zip(self.decoder, self.upsamples)):
            real_idx = len(self.encoder) - 1 - level_idx
            if not isinstance(upsample, nn.Identity):
                h = upsample(h)
                skip = skips[real_idx]
                # Spatial size match safety (stride mismatches can lose 1 pixel).
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
                h = torch.cat([h, skip], dim=1)
            for module in block:
                if isinstance(module, ResBlock):
                    h = module(h, t_emb)
                else:                                                     # SelfAttention
                    h = module(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# ---------------------------------------------------------------------------
# Diffusion schedule + DDIM sampler.
# ---------------------------------------------------------------------------


class LinearDiffusion(nn.Module):
    """Linear-beta diffusion schedule with forward q(x_t | x_0) helpers.

    Standard DDPM linear schedule: ``beta_t = linspace(beta_start, beta_end, T)``.
    Stores ``alphas``, ``alpha_bars``, ``sqrt_alpha_bars`` etc. as buffers so
    they move with the module via ``.to(device)``.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1.0e-4,
        beta_end:   float = 2.0e-2,
    ):
        super().__init__()
        self.num_train_timesteps = int(num_train_timesteps)

        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sqrt_alpha_bars', alpha_bars.sqrt())
        self.register_buffer('sqrt_one_minus_alpha_bars', (1.0 - alpha_bars).sqrt())

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise."""
        sqrt_ab    = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_1m_ab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        return sqrt_ab * x_0 + sqrt_1m_ab * noise


@torch.no_grad()
def ddim_sample(
    refiner: SARDRefiner,
    diffusion: LinearDiffusion,
    sar: torch.Tensor,
    coarse: torch.Tensor,
    num_steps: int = 3,
    eta: float = 0.0,
    init_strength: float = 1.0,
) -> torch.Tensor:
    """Deterministic DDIM sampling, warm-started from ``coarse`` + noise.

    ``init_strength`` in [0, 1] interpolates the starting point:
      * 1.0 = init from pure noise at ``t = T-1`` (vanilla DDPM start)
      * 0.0 = init from clean coarse + zero noise (degenerate, no refinement)
      Recommend 0.6-0.9 — leaves enough noise for refinement without erasing
      the GAN's structural prior.

    Returns refined x_0 estimate in [-1, 1] (tanh-clamped from the schedule).
    """
    B = sar.size(0)
    device = sar.device
    T = diffusion.num_train_timesteps
    # Build descending timestep schedule, evenly spaced.
    step_ratio = T // num_steps
    timesteps = torch.arange(T - 1, -1, -step_ratio, device=device)[:num_steps]
    timesteps = timesteps.long()

    # Warm-start x_T from coarse + noise at the first scheduled timestep.
    t_first = timesteps[0]
    sqrt_ab    = diffusion.sqrt_alpha_bars[t_first]
    sqrt_1m_ab = diffusion.sqrt_one_minus_alpha_bars[t_first]
    eps_init = torch.randn_like(coarse) * float(init_strength)
    x_t = sqrt_ab * coarse + sqrt_1m_ab * eps_init

    for i in range(num_steps):
        t = timesteps[i]
        t_batch = t.expand(B)
        eps_pred = refiner(sar, coarse, x_t, t_batch)

        alpha_bar_t = diffusion.alpha_bars[t]
        # x_0 prediction from eps prediction (DDIM)
        x_0_pred = (x_t - (1.0 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()
        x_0_pred = x_0_pred.clamp(-1.0, 1.0)

        if i < num_steps - 1:
            t_prev = timesteps[i + 1]
            alpha_bar_prev = diffusion.alpha_bars[t_prev]
            # DDIM update: x_{t-1} = sqrt(a_bar_prev) * x_0 + sqrt(1 - a_bar_prev) * eps
            sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t)).sqrt() * \
                    (1 - alpha_bar_t / alpha_bar_prev).sqrt()
            dir_xt = (1 - alpha_bar_prev - sigma ** 2).sqrt() * eps_pred
            noise  = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
            x_t = alpha_bar_prev.sqrt() * x_0_pred + dir_xt + sigma * noise
        else:
            x_t = x_0_pred

    return x_t


# ---------------------------------------------------------------------------
# Tier 1 unit smoke.  Verifies forward shapes, gradient flow, sampler
# determinism with fixed seed.  Run with::
#
#     python -m src.models.llwt_v5.diffusion
# ---------------------------------------------------------------------------


def _smoke_sardr() -> None:
    print("[sardr smoke] forward shape + grad + sampler")
    torch.manual_seed(0)

    B, H, W = 2, 64, 64                                                 # small for CPU speed
    sar    = torch.randn(B, 1, H, W)
    coarse = torch.tanh(torch.randn(B, 3, H, W))
    x_t    = torch.randn(B, 3, H, W)
    t      = torch.randint(0, 1000, (B,))

    refiner = SARDRefiner(sar_channels=1, base_dim=32, ch_mult=(1, 2, 4, 4))
    n_params = sum(p.numel() for p in refiner.parameters())
    print(f"  refiner params: {n_params/1e6:.2f}M  (test config; prod = ~7M with base_dim=64)")

    eps_pred = refiner(sar, coarse, x_t, t)
    assert eps_pred.shape == (B, 3, H, W), f"bad shape: {eps_pred.shape}"
    print(f"  [OK] forward shape: eps_pred = {tuple(eps_pred.shape)}")

    # Gradient backflow through eps_pred -> refiner weights.
    target = torch.randn_like(eps_pred)
    loss = F.mse_loss(eps_pred, target)
    loss.backward()
    grads = [p.grad for p in refiner.parameters() if p.grad is not None]
    assert len(grads) > 0, "no gradients computed"
    nz = sum(int(g.abs().sum() > 0) for g in grads)
    assert nz > 0, "all gradients zero (suspect dead branch)"
    print(f"  [OK] gradient flows: {nz}/{len(grads)} params received nonzero grad; loss = {loss.item():.4f}")

    # Diffusion schedule + q_sample shape.
    diffusion = LinearDiffusion(num_train_timesteps=1000)
    noise = torch.randn_like(coarse)
    t_b = torch.randint(0, 1000, (B,))
    x_t_q = diffusion.q_sample(coarse, t_b, noise)
    assert x_t_q.shape == coarse.shape
    print(f"  [OK] q_sample shape: {tuple(x_t_q.shape)}")

    # DDIM sampler determinism with eta=0.
    refiner.eval()
    torch.manual_seed(42)
    out1 = ddim_sample(refiner, diffusion, sar, coarse, num_steps=3, eta=0.0)
    torch.manual_seed(42)
    out2 = ddim_sample(refiner, diffusion, sar, coarse, num_steps=3, eta=0.0)
    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-6, f"DDIM eta=0 should be deterministic; max diff = {diff}"
    assert out1.shape == coarse.shape
    print(f"  [OK] DDIM eta=0 deterministic: max diff = {diff:.2e}, output shape = {tuple(out1.shape)}")

    # DDIM output range sanity (tanh-clamped to [-1, 1]).
    assert out1.min() >= -1.0 - 1e-4 and out1.max() <= 1.0 + 1e-4, \
        f"DDIM output outside [-1, 1]: min={out1.min()}, max={out1.max()}"
    print(f"  [OK] DDIM output in [-1, 1]: range [{out1.min():.3f}, {out1.max():.3f}]")

    print("[sardr smoke] PASS")


if __name__ == '__main__':
    _smoke_sardr()
