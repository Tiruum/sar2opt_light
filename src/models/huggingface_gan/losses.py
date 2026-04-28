import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    """LSGAN loss with label smoothing support.

    Handles both single logit tensors and tuples (for multi-scale discriminator).
    For tuples, computes the mean loss across all scales.
    """
    def __init__(self, real_smooth: float = 0.9, fake_smooth: float = 0.0):
        super().__init__()
        self.criterion   = nn.MSELoss()
        self.real_smooth = real_smooth
        self.fake_smooth = fake_smooth

    def _loss(self, logit: torch.Tensor, is_real: bool) -> torch.Tensor:
        """Compute MSE loss against label."""
        val = self.real_smooth if is_real else self.fake_smooth
        return self.criterion(logit, torch.full_like(logit, val))

    def forward(self, logits, is_real: bool) -> torch.Tensor:
        """Forward pass.

        Args:
            logits: Single tensor or tuple of tensors (for multi-scale discriminator)
            is_real: Whether target is real (True) or fake (False)

        Returns:
            Scalar loss value
        """
        if isinstance(logits, (list, tuple)):
            return sum(self._loss(l, is_real) for l in logits) / len(logits)
        return self._loss(logits, is_real)


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss between fake and real discriminator feature maps.

    Averages L1 loss across all feature layers. Real features are detached
    to prevent backprop through real images during generator training.
    """
    def forward(self, fake_feats: list, real_feats: list) -> torch.Tensor:
        """Forward pass.

        Args:
            fake_feats: List of feature tensors from discriminator on fake images
            real_feats: List of feature tensors from discriminator on real images

        Returns:
            Mean L1 loss across all layers
        """
        loss = sum(F.l1_loss(f, r.detach()) for f, r in zip(fake_feats, real_feats))
        return loss / len(fake_feats)


class FFTLoss(nn.Module):
    """FFT-domain loss for frequency constraint.

    Computes L1 loss on log-magnitude of the FFT to encourage frequency alignment.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pred: Predicted image
            target: Target image

        Returns:
            L1 loss in frequency domain
        """
        pred_mag   = torch.log1p(torch.abs(torch.fft.rfft2(pred,   norm='ortho')))
        target_mag = torch.log1p(torch.abs(torch.fft.rfft2(target, norm='ortho')))
        return F.l1_loss(pred_mag, target_mag)


class PerceptualLoss(nn.Module):
    """Perceptual loss using ConvNeXt V2 backbone.

    Extracts multi-scale features from a pre-trained ConvNeXt V2 backbone
    and computes L1 loss on them. Backbone is frozen and ImageNet-normalized.
    """
    def __init__(self, backbone_name: str = "facebook/convnextv2-tiny-22k-224"):
        super().__init__()
        from transformers import AutoBackbone
        self.backbone = AutoBackbone.from_pretrained(backbone_name, out_indices=(0, 1, 2))
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [-1, 1] to ImageNet normalization."""
        return ((x + 1) / 2 - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pred: Predicted image (in [-1, 1])
            target: Target image (in [-1, 1])

        Returns:
            Mean L1 loss across all feature levels
        """
        pf = self.backbone(pixel_values=self._norm(pred)).feature_maps
        tf = self.backbone(pixel_values=self._norm(target)).feature_maps
        return sum(F.l1_loss(p, t.detach()) for p, t in zip(pf, tf)) / len(pf)
