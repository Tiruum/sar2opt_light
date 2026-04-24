# models/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import typing

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real, real_label_smooth=1.0, fake_label_smooth=0.0):
        if target_is_real:
            return torch.full_like(prediction, real_label_smooth)
        else:
            return torch.full_like(prediction, fake_label_smooth)

    def forward(self, prediction, target_is_real, real_label_smooth=1.0, fake_label_smooth=0.0):
        target_tensor = self.get_target_tensor(
            prediction,
            target_is_real,
            real_label_smooth=real_label_smooth,
            fake_label_smooth=fake_label_smooth
        ).to(prediction.device)
        return self.loss(prediction, target_tensor)

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)
    
class FeatureMatchingLoss(nn.Module):
    """
    High-dimensional Feature Matching Loss (LFM) as described in CFRWD-GAN paper.
    Assumes feat_real and feat_fake are lists of feature maps from discriminator layers.
    LFM = sum_{i=0}^M (1 / (C_i * W_i * H_i)) * || D^i(Y) - D^i(G(X)) ||_1
    Which simplifies to sum mean(|D^i(Y) - D^i(G(X))|) over layers i.
    """
    def __init__(self):  # M+1, adjust based on discriminator layers
        super(FeatureMatchingLoss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')  # Mean L1 per layer

    def forward(self, feat_real, feat_fake):
        """
        feat_real: list of tensors [batch, C_i, W_i, H_i] from D on real
        feat_fake: list of tensors [batch, C_i, W_i, H_i] from D on fake
        """
        if len(feat_real) != len(feat_fake):
            raise ValueError(f"Длина feat_real ({len(feat_real)}) не совпадает с длиной feat_fake ({len(feat_fake)})")

        total_loss = 0.0
        num_layers = len(feat_real)
        for i in range(num_layers):
            # Normalization: 1/(C W H) * sum |diff| == mean(|diff|) since L1 mean is sum/N
            diff_loss = self.l1_loss(feat_real[i], feat_fake[i])
            total_loss += diff_loss
        return total_loss


class FFTLoss(nn.Module):
    """
    Frequency-domain L1 loss on log-magnitude spectrum.

    Для SAR→OPT модель должна воспроизводить и грубую структуру (низкие частоты),
    и тонкие края/текстуры (высокие частоты). Стандартная L1 в пространстве пикселей
    штрафует за любое отклонение, но не различает частоты — высокочастотные детали
    получают непропорционально малый сигнал (их амплитуда мала).

    Формула:
        L_fft = mean( |log(1 + |FFT(pred)|) - log(1 + |FFT(target)|)| )

    log(1 + |F|) сжимает динамический диапазон спектра (охватывает несколько порядков
    величины), давая сбалансированные градиенты как на низких, так и на высоких частотах.
    norm='ortho' нормирует на sqrt(N) → результат не зависит от разрешения изображения.

    Вычисление в float32: FFT нестабилен в bf16 из-за малой амплитуды ВЧ-коэффициентов.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # float32: FFT нестабилен в bf16
        pred_fft   = torch.fft.fft2(pred.float(),   norm='ortho')
        target_fft = torch.fft.fft2(target.float(), norm='ortho')

        pred_mag   = torch.log1p(pred_fft.abs())
        target_mag = torch.log1p(target_fft.abs())

        return torch.mean(torch.abs(pred_mag - target_mag))


class AdaptiveLoss(nn.Module):
    """
    Homoscedastic uncertainty weighting (Kendall et al., 2018).
    Авто-балансирует N компонент потерь через обучаемые лог-дисперсии eta.

    Формула:
        L_total = sum_i( L_i * exp(-eta_i) + eta_i )

    Механизм:
        - Если L_i велика → градиент увеличивает eta_i → exp(-eta_i) уменьшается
          → потеря i автоматически down-weightится.
        - Слагаемое eta_i является регуляризатором, препятствующим eta → +∞.
        - При инициализации eta=0: все потери имеют полный вес exp(0)=1.

    Параметры eta обучаются вместе с генератором (включены в optG).
    Логируйте `loss/eta_*` и `loss/w_*` = exp(-eta) для анализа баланса.
    """
    def __init__(self, n_losses: int):
        super().__init__()
        # eta=0 → effective weight = exp(-0) = 1.0 для всех компонент
        self.eta = nn.Parameter(torch.zeros(n_losses))

    def forward(self, losses: typing.List[torch.Tensor]) -> torch.Tensor:
        assert len(losses) == self.eta.numel(), \
            f"Expected {self.eta.numel()} losses, got {len(losses)}"
        return sum(L * torch.exp(-e) + e for L, e in zip(losses, self.eta))


class LPIPSLoss(nn.Module):
    """
    Perceptual loss — wraps a frozen AlexNet backbone.

    Accepts the backbone from an already-initialized torchmetrics LPIPS instance
    to avoid loading AlexNet twice into VRAM.

    Usage:
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False)
        lpips_loss = LPIPSLoss(backbone=lpips_metric.net)

    Input: B×3×H×W in [-1, 1] (tanh output — correct).
    Output: scalar mean LPIPS per batch (lower = better, 0 = identical).
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.net = backbone
        # Backbone is already frozen by torchmetrics; enforce it here too.
        for p in self.net.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # float32: AlexNet can be numerically unstable in bf16 for small activations
        return self.net(pred.float(), target.float()).mean()


class WaveletSupervisionLoss(nn.Module):
    """
    Wavelet-domain L1 supervision for the HFCF branch output.

    Computes 2-level Haar DWT of pred and target, then L1 on all 6 detail
    subbands (LH1, HL1, HH1, LH2, HL2, HH2). LL subbands are excluded:
    the HFCF branch discards LL2 in its own DWT, so supervising LL would
    penalise content the branch cannot model from its inputs.

    Architecturally matched: HFCF processes wavelet coefficients → supervise
    its output in the same domain. Unlike full-image L1, this does not force
    the HF-only branch to reconstruct LF structure.

    Input: B×3×H×W tensors in [-1, 1] (tanh output range).
    """
    def __init__(self):
        super().__init__()
        # HaarDown uses register_buffer — auto-moved with .to(device).
        # in_channels arg is unused in forward computation (works for any C).
        from src.models.cfrwd.gen import HaarDown
        self._haar = HaarDown(in_channels=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # float32: avoid bf16 precision loss in DWT (same rationale as FFTLoss)
        p = pred.float()
        t = target.float()

        # Level-1 decomposition
        ll1_p, lh1_p, hl1_p, hh1_p = self._haar(p)
        ll1_t, lh1_t, hl1_t, hh1_t = self._haar(t)

        # Level-2 decomposition on LL1
        _, lh2_p, hl2_p, hh2_p = self._haar(ll1_p)
        _, lh2_t, hl2_t, hh2_t = self._haar(ll1_t)

        # Mean L1 across all 6 detail subbands (equal weighting, no LL)
        return (
            F.l1_loss(lh1_p, lh1_t) + F.l1_loss(hl1_p, hl1_t) + F.l1_loss(hh1_p, hh1_t) +
            F.l1_loss(lh2_p, lh2_t) + F.l1_loss(hl2_p, hl2_t) + F.l1_loss(hh2_p, hh2_t)
        ) / 6.0
