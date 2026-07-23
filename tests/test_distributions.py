"""Tests for PBjam probability distributions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from pbjam import distributions


def evaluate_scalar(func, values):
    """Evaluate a scalar-oriented JAX function over an array."""
    return np.array([func(float(value)) for value in values], dtype=float)


def test_beta_matches_scipy_inside_support():
    dist = distributions.beta(a=2.5, b=3.5, loc=1.0, scale=4.0)
    reference = stats.beta(a=2.5, b=3.5, loc=1.0, scale=4.0)
    x = np.array([1.4, 2.0, 3.0, 4.4])

    assert np.allclose(evaluate_scalar(dist.pdf, x), reference.pdf(x))
    assert np.allclose(evaluate_scalar(dist.logpdf, x), reference.logpdf(x))
    assert np.allclose(evaluate_scalar(dist.cdf, x), reference.cdf(x))


def test_beta_ppf_cdf_round_trip_and_boundaries():
    dist = distributions.beta(a=2.0, b=3.0, loc=-2.0, scale=5.0)
    q = jnp.array([0.0, 0.1, 0.5, 0.9, 1.0])

    x = dist.ppf(q)

    assert np.all(np.isfinite(x))
    assert np.allclose(dist.cdf(x), q, atol=2e-8)
    assert np.isclose(x[0], -2.0)
    assert np.isclose(x[-1], 3.0)
    assert dist.pdf(-2.1) == 0.0
    assert dist.pdf(3.1) == 0.0
    assert np.isneginf(dist.logpdf(-2.1))


def test_beta_pdf_is_normalized():
    dist = distributions.beta(a=3.0, b=4.0, loc=2.0, scale=3.0)
    x = np.linspace(2.0, 5.0, 2001)
    pdf = evaluate_scalar(dist.pdf, x)

    assert np.trapezoid(pdf, x) == pytest.approx(1.0, rel=2e-4)


def test_normal_matches_scipy_and_round_trips():
    dist = distributions.normal(loc=3.0, scale=1.5)
    reference = stats.norm(loc=3.0, scale=1.5)
    x = jnp.array([-1.0, 1.0, 3.0, 5.0, 7.0])
    q = jnp.array([0.01, 0.1, 0.5, 0.9, 0.99])

    assert np.allclose(dist.pdf(x), reference.pdf(np.asarray(x)))
    assert np.allclose(dist.logpdf(x), reference.logpdf(np.asarray(x)))
    assert np.allclose(dist.cdf(x), reference.cdf(np.asarray(x)))
    assert np.allclose(dist.cdf(dist.ppf(q)), q, atol=1e-10)
    assert dist.mean == pytest.approx(3.0)
    assert dist.median == pytest.approx(3.0)


def test_normal_unnormalized_density_peaks_at_one():
    dist = distributions.normal(loc=-1.0, scale=2.0)

    assert dist.pdf(-1.0, norm=False) == pytest.approx(1.0)
    assert dist.logpdf(-1.0, norm=False) == pytest.approx(0.0)


def test_uniform_matches_scipy_inside_support():
    dist = distributions.uniform(loc=-2.0, scale=5.0)
    reference = stats.uniform(loc=-2.0, scale=5.0)
    x = np.array([-1.5, 0.0, 1.5, 2.5])
    q = jnp.array([0.0, 0.2, 0.5, 0.8, 1.0])

    assert np.allclose(evaluate_scalar(dist.pdf, x), reference.pdf(x))
    assert np.allclose(evaluate_scalar(dist.logpdf, x), reference.logpdf(x))
    assert np.allclose(evaluate_scalar(dist.cdf, x), reference.cdf(x))
    assert np.allclose(dist.cdf(dist.ppf(q)), q)
    assert dist.mean == pytest.approx(0.5)
    assert dist.median == pytest.approx(0.5)


@pytest.mark.xfail(
    reason="uniform.cdf does not currently clip values outside its support.",
    strict=False,
)
def test_uniform_cdf_is_bounded_outside_support():
    dist = distributions.uniform(loc=-2.0, scale=5.0)

    assert dist.cdf(-3.0) == 0.0
    assert dist.cdf(4.0) == 1.0


def test_truncated_sine_is_normalized_and_round_trips():
    dist = distributions.truncsine()
    x = np.linspace(0.0, np.pi / 2.0, 2001)
    pdf = evaluate_scalar(dist.pdf, x)
    q = jnp.array([0.0, 0.1, 0.5, 0.9, 1.0])

    assert np.trapezoid(pdf, x) == pytest.approx(1.0, rel=2e-6)
    assert np.allclose(dist.cdf(dist.ppf(q)), q, atol=1e-10)
    assert dist.ppf(0.0) == pytest.approx(0.0)
    assert dist.ppf(1.0) == pytest.approx(np.pi / 2.0)


@pytest.mark.xfail(
    reason="truncsine.cdf does not currently clip values outside its support.",
    strict=False,
)
def test_truncated_sine_cdf_is_bounded_outside_support():
    dist = distributions.truncsine()

    assert dist.cdf(-0.1) == 0.0
    assert dist.cdf(np.pi) == 1.0


def test_randint_pdf_cdf_and_ppf_on_support():
    dist = distributions.randint(low=2, high=6)
    values = np.arange(2, 6)
    quantiles = np.array([0.1, 0.3, 0.6, 0.9])

    assert np.allclose(evaluate_scalar(dist.pdf, values), 0.25)
    assert np.allclose(evaluate_scalar(dist.cdf, values), [0.25, 0.5, 0.75, 1.0])
    assert np.array_equal(np.asarray(dist.ppf(quantiles)), [2.0, 3.0, 4.0, 5.0])


@pytest.mark.xfail(
    reason="randint.logpdf returns -N rather than -log(N).",
    strict=False,
)
def test_randint_logpdf_matches_probability_mass():
    dist = distributions.randint(low=2, high=6)

    assert dist.logpdf(3.0) == pytest.approx(np.log(0.25))


@pytest.mark.xfail(
    reason="randint currently estimates its mean by continuous integration.",
    strict=False,
)
def test_randint_summary_attributes():
    dist = distributions.randint(low=2, high=6)

    assert dist.mean == pytest.approx(3.5)
    assert dist.median == pytest.approx(3.0)


@pytest.mark.parametrize(
    "dist,x",
    [
        (distributions.beta(a=2.0, b=3.0), 0.4),
        (distributions.normal(loc=0.0, scale=2.0), 0.4),
        (distributions.uniform(loc=-1.0, scale=2.0), 0.4),
        (distributions.truncsine(), 0.4),
    ],
)
def test_distribution_density_methods_jit_compile(dist, x):
    pdf = jax.jit(lambda value: dist.pdf(value))
    logpdf = jax.jit(lambda value: dist.logpdf(value))

    assert np.isfinite(pdf(jnp.array(x)))
    assert np.isfinite(logpdf(jnp.array(x)))


def test_random_variates_use_the_inverse_cdf(monkeypatch):
    monkeypatch.setattr(distributions.np.random, "uniform", lambda low, high: 0.25)
    dist = distributions.normal(loc=2.0, scale=3.0)

    assert dist.rv() == pytest.approx(dist.ppf(0.25))


def test_generic_distribution_wrapper_exposes_supplied_functions():
    reference = stats.norm(loc=0.0, scale=1.0)
    dist = distributions.distribution(
        ppf=reference.ppf,
        pdf=reference.pdf,
        logpdf=reference.logpdf,
        cdf=reference.cdf,
    )

    assert dist.pdf(0.0) == pytest.approx(reference.pdf(0.0))
    assert dist.logpdf(0.0) == pytest.approx(reference.logpdf(0.0))
    assert dist.cdf(0.0) == pytest.approx(0.5)
    assert dist.ppf(0.5) == pytest.approx(0.0)
    assert dist.mean == pytest.approx(0.0, abs=1e-5)
    assert dist.median == pytest.approx(0.0)


def test_quantile_functions_reproduce_sample_marginals():
    rng = np.random.default_rng(1234)
    data = np.column_stack(
        [
            rng.normal(-1.0, 0.5, 300),
            rng.normal(3.0, 1.2, 300),
        ]
    )

    ppfs, pdfs, logpdfs, cdfs = distributions.getQuantileFuncs(
        data,
        cut=3,
        densityScale=4,
    )

    assert len(ppfs) == len(pdfs) == len(logpdfs) == len(cdfs) == 2

    for index in range(2):
        median = float(ppfs[index](0.5))
        assert median == pytest.approx(np.median(data[:, index]), abs=0.2)
        assert float(pdfs[index](median)) > 0.0
        assert np.isfinite(float(logpdfs[index](median)))

        cdf = cdfs[index]
        if callable(cdf):
            assert 0.35 < float(cdf(median)) < 0.65
        else:
            cdf = np.asarray(cdf)
            assert np.all(np.diff(cdf) >= 0.0)
            assert cdf[0] >= 0.0
            assert cdf[-1] <= 1.0


def test_make_dist_object_builds_one_wrapper_per_column():
    rng = np.random.default_rng(4321)
    data = np.column_stack(
        [
            rng.normal(0.0, 1.0, 200),
            rng.normal(5.0, 2.0, 200),
        ]
    )

    result = distributions.makeDistObject(data, cut=3, densityScale=3)

    assert len(result) == 2
    assert all(callable(dist.ppf) for dist in result)
    assert all(callable(dist.pdf) for dist in result)
    assert all(callable(dist.logpdf) for dist in result)
    assert np.all(np.isfinite([dist.median for dist in result]))
