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
    Perceptual loss на основе замороженного AlexNet (LPIPS, Zhang et al. 2018).

    Используется ТОЛЬКО как тренировочный лосс — отдельный от torchmetrics-метрики,
    чтобы избежать аккумуляции состояния. Инициализируется из уже загруженного
    torchmetrics LPIPS (общий AlexNet не грузится дважды).

    Вход: B×3×H×W в диапазоне [-1, 1] (tanh — уже корректно).
    Выход: скаляр — среднее LPIPS по батчу (ниже = лучше, 0 = идентично).

    AlexNet выбран над VGG: в 8× быстрее, ~60 MB vs ~500 MB,
    качество градиентного сигнала достаточно для image translation.
    """
    def __init__(self):
        super().__init__()
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as _LPIPS
        # Извлекаем замороженный AlexNet из torchmetrics-обёртки.
        # Это гарантирует идентичные веса с val-метрикой без двойной загрузки.
        _metric = _LPIPS(net_type='alex', normalize=False)
        self.net = _metric.net
        # Заморозка: градиент течёт в fake_opt, но не обновляет AlexNet
        for p in self.net.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # float32: AlexNet нестабилен в bf16 для малых активаций
        return self.net(pred.float(), target.float()).mean()