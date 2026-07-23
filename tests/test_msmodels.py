"""Tests for the combined main-sequence mode model."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from pbjam import MSmodels
from pbjam.MSmodels import Asyl021model


@pytest.fixture
def main_sequence_model():
    model = Asyl021model.__new__(Asyl021model)
    model.N_p = 3
    model.f = jnp.linspace(80.0, 120.0, 81)
    model.vis = {"V20": 0.5, "V10": 1.25}
    model._makeEmpties()
    return model


def test_asymptotic_main_sequence_radial_frequencies(main_sequence_model):
    frequencies, orders = main_sequence_model.asymptotic_nu_p(
        numax=100.0,
        dnu=10.0,
        eps_p=1.0,
        alpha_p=0.0,
    )

    assert np.array_equal(orders, np.array([8, 9, 10]))
    assert np.allclose(frequencies, [90.0, 100.0, 110.0])


def test_add1modes_uses_only_m_zero_at_zero_inclination(
    main_sequence_model,
    monkeypatch,
):
    model = main_sequence_model
    model.N_p = 1
    model.asymptotic_nu_p = lambda **kwargs: (
        jnp.array([100.0]),
        jnp.array([10]),
    )

    monkeypatch.setattr(
        MSmodels.jar,
        "envelope",
        lambda frequencies, **kwargs: jnp.array([2.0]),
    )
    monkeypatch.setattr(
        MSmodels.jar,
        "lor",
        lambda frequencies, centre, height, width: (
            jnp.ones_like(frequencies) * height
        ),
    )

    modes = model.add1Modes(
        d01=5.0,
        mode_width=0.1,
        nurot_e=0.5,
        inc=0.0,
    )

    assert np.allclose(modes, 2.0 * model.vis["V10"])


def test_add1modes_conserves_triplet_visibility_at_edge_on_inclination(
    main_sequence_model,
    monkeypatch,
):
    model = main_sequence_model
    model.N_p = 1
    model.asymptotic_nu_p = lambda **kwargs: (
        jnp.array([100.0]),
        jnp.array([10]),
    )

    monkeypatch.setattr(
        MSmodels.jar,
        "envelope",
        lambda frequencies, **kwargs: jnp.array([2.0]),
    )
    monkeypatch.setattr(
        MSmodels.jar,
        "lor",
        lambda frequencies, centre, height, width: (
            jnp.ones_like(frequencies) * height
        ),
    )

    modes = model.add1Modes(
        d01=5.0,
        mode_width=0.1,
        nurot_e=0.5,
        inc=np.pi / 2.0,
    )

    # The two m=+-1 components each carry half of the total visibility.
    assert np.allclose(modes, 2.0 * model.vis["V10"])


def test_model_combines_three_ridges_then_applies_background():
    model = Asyl021model.__new__(Asyl021model)
    model.add20Pairs = lambda **kwargs: (
        jnp.array([2.0, 3.0]),
        None,
        None,
    )
    model.add1Modes = lambda **kwargs: jnp.array([4.0, 5.0])
    model.background = lambda theta: jnp.array([10.0, 20.0])

    result = model.model({"unused": 1.0})

    assert np.array_equal(result, np.array([60.0, 160.0]))


def test_setup_dr_allows_missing_bp_rp(monkeypatch):
    calls = {}

    class DummyPCA:
        def __init__(self, obs, varLabels, fName, nSamples, selectLabels):
            calls["obs"] = obs
            calls["selectLabels"] = selectLabels

        def fit_weightedPCA(self, dimensions):
            calls["dimensions"] = dimensions

        def setLatentNormalPrior(self):
            return None

        def refinePriorByObservables(self, **kwargs):
            return None

    monkeypatch.setattr(MSmodels, "PCA", DummyPCA)
    model = Asyl021model.__new__(Asyl021model)
    model.obs = {
        "numax": (100.0, 5.0),
        "dnu": (10.0, 0.5),
        "teff": (5777.0, 50.0),
    }
    model.pcaLabels = ["numax", "dnu"]
    model.priorPath = "prior.csv"
    model.PCAsamples = 20
    model.PCAdims = 2
    model.selectivePrior = False

    model.setupDR()

    assert calls["selectLabels"] == ["numax", "dnu", "teff"]
    assert "bp_rp" not in calls["obs"]


def test_unpack_params_uses_prior_order_and_log_scaling():
    model = Asyl021model.__new__(Asyl021model)
    model.DR = SimpleNamespace(
        dimsR=1,
        inverse_transform=lambda theta: jnp.array([1.0]),
    )
    model.pcaLabels = ["numax"]
    model.addLabels = ["H3_power", "inc"]
    model.priors = {
        "theta_0": object(),
        "inc": object(),
        "H3_power": object(),
    }
    model.logpars = ["numax", "H3_power"]

    unpacked = model.unpackParams(jnp.array([0.0, 0.5, 2.0]))

    assert unpacked["numax"] == pytest.approx(10.0)
    assert unpacked["inc"] == pytest.approx(0.5)
    assert unpacked["H3_power"] == pytest.approx(100.0)


def test_parse_samples_returns_three_ridges_with_expected_offsets():
    model = Asyl021model.__new__(Asyl021model)
    model.N_p = 2
    model.N_p_range = jnp.arange(2)
    model.N_p_mid = jnp.floor(model.N_p / 2)
    model.ell = np.array([0, 0, 2, 2, 1, 1])
    model.emm = np.zeros(6)
    model.vis = {"V20": 0.5, "V10": 1.25}

    samples = {
        "numax": np.array([100.0, 101.0, 102.0]),
        "dnu": np.array([10.0, 10.0, 10.0]),
        "eps_p": np.array([1.0, 1.0, 1.0]),
        "alpha_p": np.array([0.0, 0.0, 0.0]),
        "env_height": np.array([2.0, 2.0, 2.0]),
        "env_width": np.array([20.0, 20.0, 20.0]),
        "mode_width": np.array([0.1, 0.2, 0.3]),
        "d02": np.array([1.0, 1.0, 1.0]),
        "d01": np.array([5.0, 5.0, 5.0]),
    }

    result = model.parseSamples(samples, Nmax=2)

    assert result["ell"].shape == (6,)
    assert result["enn"].shape == (6,)
    assert result["zeta"].shape == (6,)
    assert result["summary"]["freq"].shape == (2, 6)
    assert result["samples"]["freq"].shape == (2, 6)

    radial = result["samples"]["freq"][:, :2]
    quadrupole = result["samples"]["freq"][:, 2:4]
    dipole = result["samples"]["freq"][:, 4:6]

    assert np.allclose(quadrupole, radial - 1.0)
    assert np.allclose(dipole, radial + 5.0)
