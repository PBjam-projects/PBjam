"""Tests for posterior-versus-prior validation."""

import numpy as np
import pytest
import scipy.stats as st

from pbjam import distributions
from pbjam.validation import validate


def make_validator(samples=None):
    if samples is None:
        quantiles = (np.arange(200) + 0.5) / 200
        samples = st.norm.ppf(quantiles)

    validator = validate(
        priors={"x": distributions.normal(loc=0.0, scale=1.0)},
        postSamples={"x": np.asarray(samples)},
    )
    validator.testSizeLim = 200
    return validator


def test_ks_test_returns_expected_result_structure():
    result = make_validator().KStest(threshold=0.05)

    assert set(result) == {"statistic", "pvalue", "significant"}
    assert result["statistic"].shape == (1,)
    assert result["pvalue"].shape == (1,)
    assert result["significant"].dtype == bool
    assert not result["significant"][0]


def test_width_ratio_flags_narrow_posterior():
    samples = np.linspace(-0.1, 0.1, 200)
    result = make_validator(samples).widthRatio(threshold=0.5)

    assert result["statistic"][0] < 0.5
    assert result["significant"][0]


def test_width_ratio_rejects_unsupported_prior():
    validator = validate(
        priors={"x": distributions.uniform(loc=0.0, scale=1.0)},
        postSamples={"x": np.linspace(0.0, 1.0, 20)},
    )

    with pytest.raises(TypeError, match="normal and beta"):
        validator.widthRatio()


def test_scipy_kde_filters_nonfinite_samples():
    validator = make_validator()
    kde = validator._getScipyKDE([0.0, 1.0, 2.0, np.nan, np.inf])

    density = kde(np.array([0.5, 1.5]))

    assert density.shape == (2,)
    assert np.all(np.isfinite(density))
    assert np.all(density > 0)


@pytest.mark.parametrize(
    "samples",
    [
        [1.0],
        [1.0, np.nan],
        [2.0, 2.0, 2.0],
    ],
)
def test_scipy_kde_rejects_degenerate_samples(samples):
    validator = make_validator()

    with pytest.raises(ValueError):
        validator._getScipyKDE(samples)


@pytest.mark.parametrize(
    ("prior", "scipy_name"),
    [
        (distributions.normal(loc=2.0, scale=3.0), "norm"),
        (distributions.beta(a=2.0, b=4.0, loc=-1.0, scale=2.0), "beta"),
    ],
)
def test_scipy_distribution_conversion(prior, scipy_name):
    scipy_prior = make_validator()._getScipyDistVersion(prior)

    assert scipy_prior.dist.name == scipy_name


def test_js_test_uses_scipy_kde(monkeypatch):
    validator = make_validator()

    monkeypatch.setattr(
        validator,
        "_generateJSNullSample",
        lambda prior, N, M: np.linspace(0.0, 1.0, N),
    )

    result = validator.JStest(threshold=0.05, N=20)

    assert np.isfinite(result["statistic"][0])
    assert 0.0 <= result["pvalue"][0] <= 1.0


def test_call_dispatches_selected_tests(monkeypatch):
    validator = make_validator()
    monkeypatch.setattr(validator, "KStest", lambda: {"name": "ks"})
    monkeypatch.setattr(validator, "widthRatio", lambda: {"name": "width"})

    result = validator(tests=["KStest", "widthratio", "unknown"])

    assert result == {
        "kstest": {"name": "ks"},
        "widthratio": {"name": "width"},
    }
