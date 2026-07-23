"""Tests for radial and quadrupole mode-identification models."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from pbjam import l20models
from pbjam.l20models import Asyl20model


@pytest.fixture
def asymptotic_model():
    model = Asyl20model.__new__(Asyl20model)
    model.N_p = 5
    model.f = jnp.linspace(80.0, 120.0, 81)
    model.vis = {"V20": 0.5}
    model._makeEmpties()
    return model


def test_make_empties_builds_static_arrays(asymptotic_model):
    model = asymptotic_model

    assert np.array_equal(model.N_p_range, np.arange(5))
    assert model.N_p_mid == 2
    assert np.array_equal(model.ones_nu, np.ones_like(model.f))


def test_get_n_p_max(asymptotic_model):
    assert asymptotic_model._get_n_p_max(
        dnu=10.0,
        numax=105.0,
        eps=1.2,
    ) == pytest.approx(9.3)


def test_get_n_p_returns_consecutive_orders_around_numax(asymptotic_model):
    orders = asymptotic_model._get_n_p(10.2)

    assert np.array_equal(orders, np.array([9, 10, 11, 12, 13]))


def test_asymptotic_relation_without_curvature(asymptotic_model):
    frequencies, orders = asymptotic_model.asymptotic_nu_p(
        numax=105.0,
        dnu=10.0,
        eps_p=1.2,
        alpha_p=0.0,
    )

    assert np.array_equal(orders, np.array([8, 9, 10, 11, 12]))
    assert np.allclose(frequencies, 10.0 * (orders + 1.2))


def test_asymptotic_curvature_is_symmetric_about_nmax(asymptotic_model):
    frequencies, orders = asymptotic_model.asymptotic_nu_p(
        numax=110.0,
        dnu=10.0,
        eps_p=1.0,
        alpha_p=0.02,
    )

    nmax = asymptotic_model._get_n_p_max(10.0, 110.0, 1.0)
    linear = 10.0 * (orders + 1.0)
    curvature = frequencies - linear

    assert np.allclose(
        curvature,
        10.0 * 0.02 / 2.0 * (orders - nmax) ** 2,
    )


def test_add20pairs_includes_radial_and_visible_quadrupole_modes(
    asymptotic_model,
    monkeypatch,
):
    model = asymptotic_model
    model.N_p = 1
    model.ones_nu = jnp.ones_like(model.f)
    model.asymptotic_nu_p = lambda **kwargs: (
        jnp.array([100.0]),
        jnp.array([10]),
    )

    monkeypatch.setattr(
        l20models.jar,
        "envelope",
        lambda frequencies, **kwargs: jnp.array([2.0]),
    )
    monkeypatch.setattr(
        l20models.jar,
        "lor",
        lambda frequencies, centre, height, width: (
            jnp.ones_like(frequencies) * height
        ),
    )

    modes, frequencies, orders = model.add20Pairs(
        d02=1.0,
        mode_width=0.1,
        nurot_e=0.5,
        inc=0.0,
    )

    # At zero inclination, only m=0 is visible for l=2.
    assert np.allclose(modes, 1.0 + 2.0 + 2.0 * model.vis["V20"])
    assert np.array_equal(frequencies, np.array([100.0]))
    assert np.array_equal(orders, np.array([10]))


def test_model_multiplies_mode_snr_by_background():
    model = Asyl20model.__new__(Asyl20model)
    model.add20Pairs = lambda **kwargs: (
        jnp.array([2.0, 3.0]),
        None,
        None,
    )
    model.background = lambda theta: jnp.array([5.0, 7.0])

    result = model.model({"unused": 1.0})

    assert np.array_equal(result, np.array([10.0, 21.0]))


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

    monkeypatch.setattr(l20models, "PCA", DummyPCA)
    model = Asyl20model.__new__(Asyl20model)
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


def test_unpack_params_applies_log_scaling_in_sampling_order():
    model = Asyl20model.__new__(Asyl20model)
    model.DR = SimpleNamespace(
        dimsR=1,
        inverse_transform=lambda theta: jnp.array([2.0]),
    )
    model.pcaLabels = ["dnu"]
    model.addLabels = ["shot", "nurot_e"]
    model.priors = {
        "theta_0": object(),
        "nurot_e": object(),
        "shot": object(),
    }
    model.logpars = ["dnu", "shot"]

    unpacked = model.unpackParams(jnp.array([0.0, 3.0, 4.0]))

    assert unpacked["dnu"] == pytest.approx(100.0)
    assert unpacked["nurot_e"] == pytest.approx(3.0)
    assert unpacked["shot"] == pytest.approx(10000.0)


def test_parse_samples_returns_two_ridges_with_consistent_shapes():
    model = Asyl20model.__new__(Asyl20model)
    model.N_p = 2
    model.N_p_range = jnp.arange(2)
    model.N_p_mid = jnp.floor(model.N_p / 2)
    model.ell = np.array([0, 0, 2, 2])
    model.emm = np.zeros(4)
    model.vis = {"V20": 0.5}

    samples = {
        "numax": np.array([100.0, 101.0, 102.0]),
        "dnu": np.array([10.0, 10.0, 10.0]),
        "eps_p": np.array([1.0, 1.0, 1.0]),
        "alpha_p": np.array([0.0, 0.0, 0.0]),
        "env_height": np.array([2.0, 2.0, 2.0]),
        "env_width": np.array([20.0, 20.0, 20.0]),
        "mode_width": np.array([0.1, 0.2, 0.3]),
        "d02": np.array([1.0, 1.0, 1.0]),
    }

    result = model.parseSamples(samples, Nmax=2)

    assert result["ell"].shape == (4,)
    assert result["enn"].shape == (4,)
    assert result["zeta"].shape == (4,)
    assert result["summary"]["freq"].shape == (2, 4)
    assert result["summary"]["height"].shape == (2, 4)
    assert result["summary"]["width"].shape == (2, 4)
    assert result["samples"]["freq"].shape == (2, 4)
    assert result["samples"]["height"].shape == (2, 4)
    assert result["samples"]["width"].shape == (2, 4)
    assert np.allclose(
        result["samples"]["freq"][:, 2:],
        result["samples"]["freq"][:, :2] - 1.0,
    )
