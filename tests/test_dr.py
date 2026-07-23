"""Tests for PCA-based dimensionality reduction."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from pbjam import DR


class DeterministicRNG:
    def normal(self, loc, scale, size):
        return np.zeros(size)

    def uniform(self, low, high, size):
        return np.zeros(size)


class CentralRNG:
    """Return latent draws at the requested normal means."""

    def normal(self, loc, scale, size):
        return np.broadcast_to(np.asarray(loc), size).copy()

    def uniform(self, low, high, size):
        return np.zeros(size)


class NonFiniteRNG:
    def normal(self, loc, scale, size):
        return np.zeros(size)

    def uniform(self, low, high, size):
        return np.zeros(size)


def make_full_rank_pca(weights=None):
    pca = DR.PCA.__new__(DR.PCA)
    pca.dataF = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
        ]
    )
    pca.varLabels = ["x", "y"]
    pca.nSamples = len(pca.dataF)
    pca.weights = (
        jnp.ones(pca.nSamples)
        if weights is None
        else jnp.asarray(weights, dtype=float)
    )
    pca.mu = jnp.average(pca.dataF, axis=0, weights=pca.weights)
    pca.var = jnp.average(
        (pca.dataF - pca.mu) ** 2,
        axis=0,
        weights=pca.weights,
    )
    pca.std = jnp.sqrt(pca.var)
    return pca


def test_set_weights_supports_uniform_array_and_callable():
    pca = DR.PCA.__new__(DR.PCA)
    pca.nSamples = 3

    pca.setWeights(None, {})
    assert np.array_equal(pca.weights, np.ones(3))

    explicit = np.array([1.0, 2.0, 3.0])
    pca.setWeights(explicit, {})
    assert pca.weights is explicit

    def weight_function(model, offset):
        return np.arange(model.nSamples, dtype=float) + offset

    pca.setWeights(weight_function, {"offset": 2.0})
    assert np.array_equal(pca.weights, [2.0, 3.0, 4.0])


def test_read_prior_data_replaces_infinities_and_drops_required_rows(tmp_path):
    path = tmp_path / "prior.csv"
    pd.DataFrame(
        {
            "x": [1.0, np.inf, 3.0, np.nan],
            "numax": [10.0, 10.1, np.nan, 10.3],
        }
    ).to_csv(path, index=False)

    pca = DR.PCA.__new__(DR.PCA)
    pca.varLabels = ["x"]
    pca.selectLabels = ["numax"]
    pca.dropNansIn = "all"

    result = pca.readPriorData(path, ["x", "numax"])

    assert result.to_dict("records") == [{"x": 1.0, "numax": 10.0}]
    assert pca.dropNanLabels == ["x", "numax"]


def test_read_prior_data_can_only_require_selection_columns(tmp_path):
    path = tmp_path / "prior.csv"
    pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0],
            "numax": [10.0, 10.1, np.nan],
        }
    ).to_csv(path, index=False)

    pca = DR.PCA.__new__(DR.PCA)
    pca.varLabels = ["x"]
    pca.selectLabels = ["numax"]
    pca.dropNansIn = "selection"

    result = pca.readPriorData(path, ["x", "numax"])

    assert len(result) == 2
    assert np.isnan(result.loc[1, "x"])
    assert pca.dropNanLabels == ["numax"]


def test_find_nearest_returns_closest_viable_rows():
    pca = DR.PCA.__new__(DR.PCA)
    pca.selectLabels = ["numax", "dnu"]
    pca.obs = {
        "numax": (10.0, 0.1),
        "dnu": (5.0, 0.1),
    }
    prior = pd.DataFrame(
        {
            "numax": [9.85, 9.95, 10.05, 10.15, 10.5],
            "dnu": [4.85, 4.95, 5.05, 5.15, 5.5],
            "value": [1, 2, 3, 4, 5],
        }
    )

    result = pca.findNearest(prior, N=2)

    assert np.array_equal(result["value"], [2, 3])
    assert pca.nanFraction == 0.0
    assert pca.viableFraction == 1.0


def test_get_sample_returns_requested_variables_and_dimensions(monkeypatch):
    pca = DR.PCA.__new__(DR.PCA)
    pca.varLabels = ["x", "y"]
    pca.selectLabels = ["numax"]

    prior = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "numax": [10.0, 11.0],
        }
    )
    selected = prior.iloc[[1]].reset_index(drop=True)

    monkeypatch.setattr(pca, "readPriorData", lambda f_name, labels: prior)
    monkeypatch.setattr(pca, "findNearest", lambda data, count: selected)

    data, dimensions, sample_count = pca.getSample("unused.csv", 1)

    assert np.array_equal(data, np.array([[2.0, 4.0]]))
    assert dimensions == 2
    assert sample_count == 1
    assert pca.selectedSubset.equals(selected)


def test_scale_and_inverse_scale_round_trip():
    pca = make_full_rank_pca()
    values = jnp.array([[0.5, 1.5], [2.5, 0.5]])

    scaled = pca.scale(values)

    assert np.allclose(pca.inverse_scale(scaled), values)
    assert np.allclose(np.average(pca.scale(pca.dataF), axis=0), 0.0)


def test_weighted_covariance_matches_numpy():
    weights = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
    pca = make_full_rank_pca(weights)
    scaled = np.asarray(pca.scale(pca.dataF))

    expected = np.cov(
        scaled,
        rowvar=False,
        aweights=weights,
        ddof=1,
    )

    assert np.allclose(pca.covarianceMatrix(jnp.array(scaled)), expected)


def test_fit_weighted_pca_orders_components_and_round_trips():
    pca = make_full_rank_pca()

    pca.fit_weightedPCA(dim=2)
    reconstructed = pca.inverse_transform(pca.transform(pca.dataF))

    assert pca.dimsR == 2
    assert pca.dataR.shape == pca.dataF.shape
    assert np.allclose(reconstructed, pca.dataF, atol=1e-10)
    assert np.all(np.diff(np.asarray(pca.explained_variance_ratio)) <= 0.0)
    assert np.sum(pca.explained_variance_ratio) == pytest.approx(1.0)
    assert np.isfinite(pca.erank)


def test_fit_weighted_pca_caps_requested_dimensions():
    pca = make_full_rank_pca()

    pca.fit_weightedPCA(dim=20)

    assert pca.dimsR == len(pca.varLabels)
    assert len(pca.sortidx) == len(pca.varLabels)


def test_reduced_pca_returns_requested_latent_shape():
    pca = make_full_rank_pca()

    pca.fit_weightedPCA(dim=1)

    assert pca.dataR.shape == (pca.nSamples, 1)
    assert pca.inverse_transform(pca.dataR).shape == pca.dataF.shape


def test_set_latent_normal_prior_uses_sample_moments_and_safe_scale():
    pca = DR.PCA.__new__(DR.PCA)
    pca.dimsR = 2
    sample = np.array(
        [
            [1.0, 2.0],
            [1.0, 4.0],
            [1.0, 6.0],
        ]
    )

    pca.setLatentNormalPrior(sample)

    assert np.allclose(pca.latentPriorLoc, [1.0, 4.0])
    assert np.allclose(pca.latentPriorScale, [1.0, np.std([2.0, 4.0, 6.0])])
    assert len(pca.latentPriors) == 2
    assert len(pca.ppf) == len(pca.pdf) == len(pca.logpdf) == len(pca.cdf) == 2
    assert pca.ppf[0](0.5) == pytest.approx(1.0)


def test_refine_prior_requires_initial_latent_prior():
    pca = DR.PCA.__new__(DR.PCA)

    with pytest.raises(AttributeError, match="initial latent prior"):
        pca.refinePriorByObservables()


def test_refine_prior_returns_early_for_zero_dimensions():
    pca = DR.PCA.__new__(DR.PCA)
    pca.ppf = []
    pca.dimsR = 0

    assert pca.refinePriorByObservables() is None


def test_refine_prior_warns_when_no_observed_pca_labels():
    pca = DR.PCA.__new__(DR.PCA)
    pca.ppf = True
    pca.dimsR = 1
    pca.selectLabels = ["teff"]
    pca.varLabels = ["numax"]
    pca.obs = {"teff": (5800.0, 100.0)}

    with pytest.warns(UserWarning, match="no observed selection labels"):
        assert pca.refinePriorByObservables() is None


def test_refine_prior_accepts_central_draws_and_updates_samples():
    pca = DR.PCA.__new__(DR.PCA)
    pca.ppf = True
    pca.dimsR = 2
    pca.selectLabels = ["numax", "dnu"]
    pca.varLabels = ["numax", "dnu"]
    pca.obs = {
        "numax": (10.0, 1.0),
        "dnu": (5.0, 0.5),
    }
    pca.latentPriorLoc = np.array([10.0, 5.0])
    pca.latentPriorScale = np.array([1.0, 1.0])
    pca.inverse_transform = lambda latent: latent
    pca.transform = lambda physical: physical

    result = pca.refinePriorByObservables(
        N=4,
        minAccepted=3,
        sigmaInflation=1.0,
        rng=CentralRNG(),
    )

    assert result.shape == (4, 2)
    assert np.allclose(result, [[10.0, 5.0]] * 4)
    assert pca.selectivePriorInfo == {
        "draws": 4,
        "accepted": 4,
        "minAccepted": 3,
        "sigmaInflation": 1.0,
        "labels": ["numax", "dnu"],
    }
    assert list(pca.selectiveSubset.columns) == ["numax", "dnu"]
    assert np.allclose(pca.dataR, result)
    assert np.allclose(pca.latentPriorScale, [1.0, 1.0])


def test_refine_prior_raises_when_all_physical_draws_are_nonfinite():
    pca = DR.PCA.__new__(DR.PCA)
    pca.ppf = True
    pca.dimsR = 1
    pca.selectLabels = ["numax"]
    pca.varLabels = ["numax"]
    pca.obs = {"numax": (10.0, 1.0)}
    pca.latentPriorLoc = np.array([0.0])
    pca.latentPriorScale = np.array([1.0])
    pca.inverse_transform = lambda latent: jnp.full(latent.shape, jnp.nan)

    with pytest.raises(ValueError, match="no finite prior draws"):
        pca.refinePriorByObservables(
            N=2,
            minAccepted=1,
            rng=NonFiniteRNG(),
        )


def test_refine_prior_doubles_sigma_inflation_on_retry():
    pca = DR.PCA.__new__(DR.PCA)
    pca.ppf = True
    pca.dimsR = 1
    pca.selectLabels = ["numax"]
    pca.varLabels = ["numax"]
    pca.obs = {"numax": (10.0, 1.0)}
    pca.latentPriorLoc = np.array([0.0])
    pca.latentPriorScale = np.array([1.0])

    pca.inverse_transform = lambda latent: jnp.zeros((latent.shape[0], 1))
    pca.transform = lambda physical: physical
    pca.setLatentNormalPrior = lambda latent: setattr(pca, "prior_sample", latent)

    with pytest.warns(UserWarning, match="fewer than minAccepted"):
        pca.refinePriorByObservables(
            N=1,
            minAccepted=2,
            sigmaInflation=3,
            rng=DeterministicRNG(),
        )

    assert pca.selectivePriorInfo["draws"] == 2
    assert pca.selectivePriorInfo["accepted"] == 2
    assert pca.selectivePriorInfo["sigmaInflation"] == 6
