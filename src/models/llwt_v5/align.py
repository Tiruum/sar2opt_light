"""llwt_v5 Self-Aligning Wavelet GAN — deformation aligner + scattering-center detector.

DeformationAligner predicts a dense deformation field phi from the LL (low-frequency)
Haar band of (fake, opt) and warps the GT optical into the generator's geometry.  phi is
estimated at H/4 (deformation fields are low-frequency) and upsampled.  Zero-init on the
final conv => phi=0 => identity warp at step 0 (warm-start contract).

psc_detect produces a deterministic Point-Scattering-Center heatmap from despeckled SAR
(local maxima, top-K, gaussian-splatted).  No learnable params.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, radius: int) -> torch.Tensor:
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Separable gaussian blur. x: (B,C,H,W). Channel-wise (groups=C)."""
    radius = max(1, int(round(3 * sigma)))
    k = _gaussian_kernel1d(sigma, radius).to(x)
    C = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    x = F.conv2d(x, kx, padding=(0, radius), groups=C)
    x = F.conv2d(x, ky, padding=(radius, 0), groups=C)
    return x


class DeformationAligner(nn.Module):
    """Predict a deformation field phi (B,2,H/4,W/4) in pixel units from LL bands.

    Channel order of phi is [x, y] = [horizontal, vertical].  ``warp`` upsamples phi to the
    target resolution and converts pixel offsets to normalized grid coords.
    """

    def __init__(self, ll_channels: int = 3, max_disp_px: float = 8.0, hidden: int = 64):
        super().__init__()
        self.max_disp_px = float(max_disp_px)
        self.net = nn.Sequential(
            nn.Conv2d(2 * ll_channels, hidden, 3, stride=2, padding=1),  # LL H/2 -> H/4
            nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, 2, 3, padding=1),                          # phi @ H/4
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fake_ll: torch.Tensor, opt_ll: torch.Tensor) -> torch.Tensor:
        raw = self.net(torch.cat([fake_ll, opt_ll], dim=1))   # (B,2,H/4,W/4)
        return torch.tanh(raw) * self.max_disp_px             # pixel units

    @staticmethod
    def warp(img: torch.Tensor, phi_px: torch.Tensor) -> torch.Tensor:
        """Warp img (B,C,H,W) by phi_px (B,2,h,w) in pixels. align_corners=True so phi=0
        is an exact identity warp."""
        B, C, H, W = img.shape
        phi = F.interpolate(phi_px, size=(H, W), mode='bilinear', align_corners=True)
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=img.device),
            torch.linspace(-1.0, 1.0, W, device=img.device),
            indexing='ij',
        )
        base = torch.stack([xs, ys], dim=0).unsqueeze(0).to(img.dtype)   # (1,2,H,W) [x,y]
        scale = torch.tensor([2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)],
                             device=img.device, dtype=img.dtype).view(1, 2, 1, 1)
        grid = (base + phi * scale).permute(0, 2, 3, 1)                  # (B,H,W,2)
        return F.grid_sample(img, grid, mode='bilinear',
                             padding_mode='border', align_corners=True)


def _smoke_aligner() -> None:
    print("[align smoke] DeformationAligner identity-at-init + warp shape")
    torch.manual_seed(0)
    aligner = DeformationAligner(ll_channels=3, max_disp_px=8.0)
    fake_ll = torch.randn(2, 3, 128, 128)
    opt_ll = torch.randn(2, 3, 128, 128)
    phi = aligner(fake_ll, opt_ll)
    assert phi.shape == (2, 2, 64, 64), f"phi shape {tuple(phi.shape)}"
    assert phi.abs().max().item() < 1e-6, f"zero-init phi not zero: {phi.abs().max().item()}"
    opt = torch.randn(2, 3, 256, 256)
    warped = DeformationAligner.warp(opt, phi)
    assert warped.shape == opt.shape, f"warp shape {tuple(warped.shape)}"
    err = (warped - opt).abs().max().item()
    assert err < 1e-3, f"identity warp error {err}"  # float32 bilinear-sampling rounding tolerance; a real grid-convention bug gives O(1) error
    print(f"  [OK] phi zero-init, identity warp max err = {err:.2e}")
    print("[align smoke] PASS")


if __name__ == '__main__':
    _smoke_aligner()
