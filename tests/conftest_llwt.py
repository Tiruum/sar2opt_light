"""Shared fixtures for LLW-Former tests."""
import pytest
import torch


@pytest.fixture(scope='module')
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def rng_seed():
    torch.manual_seed(20260521)
    return 20260521
