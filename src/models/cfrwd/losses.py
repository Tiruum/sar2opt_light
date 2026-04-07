# models/losses.py

import torch
import torch.nn as nn
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

    Retained for ablation reference. Replaced by FocalFrequencyLoss in cfrwd-37
    (factory.py no longer registers this; re-import from losses.py for ablations).

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


class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss (Jiang et al., ICCV 2021).

    Replaces FFTLoss in AdaptiveLoss to prevent eta spiral.

    Problem with FFTLoss: uniform L1 on FFT magnitude → loss value converges
    to ~0.046 by epoch 5 → eta_fft → -3.07 → weight ×21.6 → FFT monopolizes
    AdaptiveLoss gradient → CFR handles all frequency reconstruction → HFCF redundant.

    Solution: per-frequency adaptive weight = |diff_f| / (|pred_f| + eps).
    Well-reconstructed frequencies get low weight; hard ones get high weight.
    The absolute loss value stays stable (normalization by prediction magnitude
    prevents collapse to near-zero), so eta equilibrium stays near 0 (weight ~1–4×).

    Input: B×C×H×W in [-1, 1] (tanh output — correct).
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = torch.fft.rfft2(pred.float(), norm='ortho')
        tgt_f  = torch.fft.rfft2(target.float(), norm='ortho')
        diff   = (pred_f - tgt_f).abs()
        with torch.no_grad():
            weight = diff / (pred_f.abs().detach() + 1e-8)
        return (weight * diff).mean()


class HFMaskedFFTLoss(nn.Module):
    """
    Frequency-band-selective supervision for the HFCF branch.

    The paper (Figure 10c) shows HFCF output is an edge map — it produces
    high-frequency optical features, not the full image. Supervising hfcf_out
    on the full optical image (pixel L1 or full FFT) is architecturally wrong
    because HFCF then competes with CFR on low-frequency content where CFR
    has more capacity. CFR wins → HFCF collapses.

    Penalizes only spatial frequencies above freq_threshold (fraction of Nyquist).
    freq_threshold=0.25 → supervise edges/textures only; coarse structure and
    color (below 25% Nyquist) are excluded — that is CFR's domain.

    Used as auxiliary loss: loss_hfcf_aux = HFMaskedFFTLoss()(hfcf_out, real_opt).

    Input: B×C×H×W in [-1, 1].
    """
    def __init__(self, freq_threshold: float = 0.25):
        super().__init__()
        self.freq_threshold = freq_threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = torch.fft.rfft2(pred.float(), norm='ortho')
        tgt_f  = torch.fft.rfft2(target.float(), norm='ortho')
        H = pred_f.shape[-2]
        W = pred.shape[-1]
        fy = torch.fft.fftfreq(H, device=pred.device).abs()   # H
        fx = torch.fft.rfftfreq(W, device=pred.device)         # W//2+1
        freq_mag = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()  # H×(W//2+1)
        hf_mask = (freq_mag > self.freq_threshold).float()
        return ((pred_f - tgt_f).abs() * hf_mask).mean()
