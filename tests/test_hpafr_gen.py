"""
Smoke tests for HPAFRGenerator and supporting utilities.

Run with: pytest tests/test_hpafr_gen.py -v
"""
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope='module')
def gen(device):
    from src.models.hpafr.gen import HPAFRGenerator
    return HPAFRGenerator(in_channels=1).to(device).eval()


@pytest.fixture(scope='module')
def dis(device):
    from src.models.hpafr.dis import HPAFRPatchDis
    return HPAFRPatchDis(in_channels=4, ndf=64).to(device).eval()


# ---------------------------------------------------------------------------
# Generator shape tests
# ---------------------------------------------------------------------------

def test_gen_output_shape(gen, device):
    x = torch.randn(2, 1, 256, 256, device=device)
    with torch.no_grad():
        out = gen(x)
    assert out.shape == (2, 3, 256, 256), f'Unexpected output shape: {out.shape}'


def test_gen_output_range(gen, device):
    x = torch.randn(2, 1, 256, 256, device=device)
    with torch.no_grad():
        out = gen(x)
    assert out.min() >= -1.0 - 1e-5, 'Output below -1'
    assert out.max() <=  1.0 + 1e-5, 'Output above +1'


def test_gen_return_despeckle(gen, device):
    x = torch.randn(2, 1, 256, 256, device=device)
    with torch.no_grad():
        out, sar_clean = gen(x, return_despeckle=True)
    assert out.shape == (2, 3, 256, 256)
    assert sar_clean.shape == x.shape, f'sar_clean shape mismatch: {sar_clean.shape}'


def test_gen_param_count(gen):
    n = sum(p.numel() for p in gen.parameters())
    # Accept between 10M and 30M params
    assert 10_000_000 <= n <= 30_000_000, f'Param count out of expected range: {n:,}'


# ---------------------------------------------------------------------------
# Gradient flow test
# ---------------------------------------------------------------------------

def test_gen_gradient_flow(device):
    from src.models.hpafr.gen import HPAFRGenerator
    gen = HPAFRGenerator(in_channels=1).to(device).train()

    x = torch.randn(1, 1, 256, 256, device=device)
    out = gen(x)
    loss = out.mean()
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in gen.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, 'No gradients computed'
    assert all(torch.isfinite(torch.tensor(g)) for g in grad_norms), 'Non-finite gradients'


# ---------------------------------------------------------------------------
# Discriminator tests
# ---------------------------------------------------------------------------

def test_dis_output_shape(dis, device):
    sar  = torch.randn(2, 1, 256, 256, device=device)
    opt  = torch.randn(2, 3, 256, 256, device=device)
    pair = torch.cat([sar, opt], dim=1)
    with torch.no_grad():
        (large, small), feats = dis(pair)
    assert large.ndim == 4
    assert small.ndim == 4
    assert len(feats) == 8  # 2 branches × 4 LeakyReLU layers


def test_dis_features_length(dis, device):
    pair = torch.randn(2, 4, 256, 256, device=device)
    with torch.no_grad():
        (_, _), feats = dis(pair)
    assert len(feats) > 0


# ---------------------------------------------------------------------------
# R1 gradient penalty sanity check
# ---------------------------------------------------------------------------

def test_r1_penalty_finite(dis, device):
    """R1 gradient norm w.r.t. real_opt should be finite and O(1)."""
    real_sar = torch.randn(1, 1, 256, 256, device=device)
    real_opt = torch.randn(1, 3, 256, 256, device=device).requires_grad_(True)
    pair = torch.cat([real_sar, real_opt], dim=1)

    dis_train = dis.train()
    (d_large, d_small), _ = dis_train(pair)
    grads = torch.autograd.grad(
        outputs=[d_large.sum() + d_small.sum()],
        inputs=[real_opt],
        create_graph=False,
    )[0]
    r1 = (10.0 / 2.0) * grads.pow(2).sum([1, 2, 3]).mean()
    assert torch.isfinite(r1), f'R1 penalty is not finite: {r1}'
    # Typical range with random inputs: 0.01 – 1000; reject if wildly off
    assert r1.item() < 1e6, f'R1 penalty suspiciously large: {r1}'


# ---------------------------------------------------------------------------
# FocalFrequencyLoss
# ---------------------------------------------------------------------------

def test_ffl_zero_on_identical(device):
    from src.models.hpafr.losses import FocalFrequencyLoss
    ffl = FocalFrequencyLoss().to(device)
    x = torch.randn(2, 3, 64, 64, device=device)
    # Identical inputs → weight matrix all zeros → loss ≈ 0
    loss = ffl(x, x)
    assert loss.item() < 1e-6, f'FFL on identical inputs should be ~0, got {loss.item()}'


def test_ffl_positive_on_different(device):
    from src.models.hpafr.losses import FocalFrequencyLoss
    ffl = FocalFrequencyLoss().to(device)
    x = torch.randn(2, 3, 64, 64, device=device)
    y = torch.randn(2, 3, 64, 64, device=device)
    loss = ffl(x, y)
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# Lee filter
# ---------------------------------------------------------------------------

def test_lee_filter_range(device):
    from src.utils.lee_filter import lee_filter
    x = torch.rand(2, 1, 64, 64, device=device) * 2.0 - 1.0
    y = lee_filter(x)
    assert y.shape == x.shape
    assert y.min() >= -1.0 - 1e-5
    assert y.max() <=  1.0 + 1e-5


def test_lee_filter_smooths(device):
    from src.utils.lee_filter import lee_filter
    # Salt-and-pepper noise + smooth input; filter should bring range closer to centre
    x = torch.zeros(1, 1, 64, 64, device=device)
    x[0, 0, 16, 16] = 0.9   # single spike
    y = lee_filter(x)
    # Spike should be reduced (not amplified)
    assert y[0, 0, 16, 16].abs() <= 0.9


# ---------------------------------------------------------------------------
# pin_diverse_batch (mock dataset)
# ---------------------------------------------------------------------------

def test_pin_diverse_batch():
    from src.utils.visualize import pin_diverse_batch
    from unittest.mock import MagicMock
    import torch

    # Build a mock dataset with 3 classes, 4 items each
    items = []
    data  = []
    classes = ['agri', 'cities', 'forest']
    for cls in classes:
        for i in range(4):
            items.append((cls, f's1_{i}.png', f's2_{i}.png'))
            sar = torch.zeros(1, 64, 64)
            opt = torch.zeros(3, 64, 64)
            data.append((sar, opt))

    class FakeDataset:
        def __init__(self):
            self.items = items
        def __getitem__(self, idx):
            return data[idx]

    dm = MagicMock()
    dm.train_dataset = FakeDataset()

    sar_b, opt_b = pin_diverse_batch(dm, n_per_class=2, device='cpu', seed=0)
    # Should have 2 samples per class × 3 classes = 6 total
    assert sar_b.shape[0] == 6
    assert opt_b.shape[0] == 6


def test_pin_diverse_batch_sen12full():
    """pin_diverse_batch must handle SEN12Full 5-tuple items (season, s1_dir, s2_dir, s1_f, s2_f)."""
    from src.utils.visualize import pin_diverse_batch
    from unittest.mock import MagicMock

    items = []
    data  = []
    scenes = ['s1_5', 's1_45', 's1_52']
    for s1_d in scenes:
        for i in range(4):
            items.append(('ROIs1158_spring', s1_d, s1_d.replace('s1', 's2'), f'p{i:03d}.png', f'p{i:03d}.png'))
            data.append((torch.zeros(1, 64, 64), torch.zeros(3, 64, 64)))

    class FakeDataset:
        def __init__(self):
            self.items = items
        def __getitem__(self, idx):
            return data[idx]

    dm = MagicMock()
    dm.train_dataset = FakeDataset()

    sar_b, opt_b = pin_diverse_batch(dm, n_per_class=2, device='cpu', seed=0)
    assert sar_b.shape[0] == 6   # 2 per scene × 3 scenes
    assert opt_b.shape[0] == 6
