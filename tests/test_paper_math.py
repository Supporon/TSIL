"""Independent numerical oracles for the equations in the original manuscript."""
import importlib.util

import numpy as np
import pytest
import torch

from src import temporal_predicates as tp


def descriptor(name, **kwargs):
    assert hasattr(tp, "get_temporal_phi"), "paper raw-descriptor API is missing"
    return tp.get_temporal_phi(name, **kwargs)


def loss_class():
    assert importlib.util.find_spec("src.tsil_loss"), "paper additive loss is missing"
    from src.tsil_loss import StatisticalInvariantLoss
    return StatisticalInvariantLoss


def test_momentum_uses_log_prices_and_keeps_all_horizons():
    close = torch.exp(torch.arange(41, dtype=torch.float64)[None, :] *
                      torch.tensor([[0.01], [0.02]], dtype=torch.float64))
    phi = descriptor("phi_momentum", close=close)
    expected = torch.tensor([[0.05, 0.10, 0.20], [0.10, 0.20, 0.40]], dtype=torch.float64)
    torch.testing.assert_close(phi, expected)


def test_volatility_is_population_std_of_returns():
    returns = torch.arange(80, dtype=torch.float64).reshape(2, 40) / 1000
    phi = descriptor("phi_volatility", returns=returns)
    expected = np.stack([returns.numpy()[:, -w:].std(axis=1, ddof=0)
                         for w in (10, 20, 40)], axis=1)
    np.testing.assert_allclose(phi.numpy(), expected, atol=1e-12)


def test_distribution_uses_four_return_moments():
    r = np.array([[0.01, -0.04, 0.07, -0.02], [0.03, 0.03, 0.03, 0.03]])
    mu = r.mean(axis=1, keepdims=True)
    sd = r.std(axis=1, keepdims=True)
    z = (r - mu) / (sd + 1e-8)
    expected = np.concatenate([mu, sd, (z ** 3).mean(1, keepdims=True),
                               (z ** 4).mean(1, keepdims=True) - 3], axis=1)
    phi = descriptor("phi_return_dist", returns=torch.tensor(r), distribution_window=4)
    np.testing.assert_allclose(phi.numpy(), expected, atol=1e-12)


def test_market_cycle_uses_time_fft_and_trading_calendar():
    q = torch.tensor([41., 42., 41.], dtype=torch.float64)
    r = torch.sin(2 * torch.pi * torch.arange(20, dtype=torch.float64) / 5)
    market = r.repeat(3, 1)
    phi = descriptor("phi_cycle", market_returns=market, trading_index=q)
    angle = 2 * torch.pi * torch.remainder(q, 5) / 5
    torch.testing.assert_close(phi, torch.stack([angle.sin(), angle.cos()], dim=1))
    torch.testing.assert_close(phi[0], phi[2])
    order = torch.tensor([2, 0, 1])
    shuffled = descriptor("phi_cycle", market_returns=market[order], trading_index=q[order])
    torch.testing.assert_close(shuffled, phi[order])


def test_phase_is_asset_specific_modulo_not_ceil():
    t = torch.arange(20, dtype=torch.float64)
    returns = torch.stack([torch.sin(2*torch.pi*t/5), torch.sin(2*torch.pi*t/10)])
    q = torch.tensor([41., 41.], dtype=torch.float64)
    phi = descriptor("phi_phase", returns=returns, trading_index=q, seq_len=7)
    expected = np.stack([np.arange(35, 42) % p / p for p in (5, 10)])
    np.testing.assert_allclose(phi.numpy(), expected, atol=1e-12)


def test_timestamp_is_one_hot_with_fixed_categories():
    # zero-based weekday and month; fixed widths 7 and 12, independent of batch
    attrs = torch.tensor([[0, 0], [4, 11], [0, 0]])
    phi = descriptor("phi_timestamp", timestamp_x=attrs)
    assert phi.shape == (3, 19)
    assert torch.equal(phi.sum(1), torch.tensor([2., 2., 2.]))
    assert phi[0, 0] == phi[0, 7] == phi[1, 4] == phi[1, 18] == 1
    assert torch.equal(phi[0], phi[2])


def test_constant_return_fft_excludes_dc_and_remains_finite():
    r = torch.zeros(2, 20)
    for name, args in [("phi_cycle", {"market_returns": r}), ("phi_phase", {"returns": r})]:
        phi = descriptor(name, trading_index=torch.tensor([40., 41.]), **args)
        assert torch.isfinite(phi).all()


@pytest.mark.parametrize("name,kwargs", [
    ("phi_momentum", {"close": torch.ones(2, 10)}),
    ("phi_momentum", {"close": torch.zeros(2, 41)}),
    ("phi_volatility", {"returns": torch.ones(2, 20)}),
    ("phi_cycle", {"market_returns": torch.ones(2, 20)}),
    ("typo", {"close": torch.ones(2, 41)}),
])
def test_invalid_or_missing_raw_inputs_are_not_silently_replaced(name, kwargs):
    with pytest.raises((ValueError, KeyError)):
        descriptor(name, **kwargs)


def test_additive_loss_and_analytic_gradient_match_paper():
    cls = loss_class()
    pred = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0.05, 0.02, -0.03], dtype=torch.float64)
    phis = {"a": torch.tensor([[1., 2.], [0., 1.], [2., -1.]], dtype=torch.float64,
                              requires_grad=True),
            "b": torch.ones(3, 1, dtype=torch.float64)}
    betas = {"a": 0.3, "b": 0.8}
    criterion = cls(betas)
    actual = criterion(pred, target, phis)
    e = (pred.detach() - target).numpy()
    P = np.zeros((3, 3))
    for key, raw in phis.items():
        phi = raw.detach().numpy()
        phi = phi / (np.linalg.norm(phi) + 1e-8)
        P += betas[key] * phi @ phi.T / 9
    expected = np.mean(e*e) + e @ P @ e
    assert actual["total"].item() == pytest.approx(expected, abs=1e-14)
    actual["total"].backward()
    np.testing.assert_allclose(pred.grad.numpy(), 2*e/3 + 2*P@e, atol=1e-14)
    assert phis["a"].grad is None
    assert list(criterion.parameters()) == []


def test_zero_beta_mse_singleton_and_zero_descriptor():
    cls = loss_class()
    pred = torch.tensor([2.], requires_grad=True)
    target = torch.tensor([1.])
    for beta, phi in [(0., torch.ones(1, 1)), (0.5, torch.zeros(1, 4))]:
        result = cls(beta)(pred, target, {"ones": phi})
        assert result["total"].item() == 1.
        assert result["P_loss"].item() == 0.


def test_loss_retains_base_mse_and_has_inverse_batch_squared_penalty():
    cls = loss_class()
    for size in (2, 8):
        result = cls(0.5)(torch.ones(size), torch.zeros(size), {"ones": torch.ones(size, 1)})
        assert result["V_loss"].item() == 1.
        assert result["P_loss"].item() == pytest.approx(0.5 / size)


def test_invalid_loss_shapes_and_weights_rejected():
    cls = loss_class()
    with pytest.raises(ValueError):
        cls(-1)
    with pytest.raises(ValueError):
        cls(0.5)(torch.ones(2), torch.ones(3), {"p": torch.ones(2, 1)})
