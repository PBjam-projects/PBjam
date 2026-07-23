"""Tests for the sampler mixins."""

import numpy as np
import jax.numpy as jnp
import pytest

from pbjam import distributions, samplers


class DummyEmceeSampler(samplers.EmceeSampling):
    def __init__(self):
        self.priors = {
            "x": distributions.uniform(loc=0.0, scale=2.0),
            "y": distributions.normal(loc=0.0, scale=1.0),
        }
        self.ndims = len(self.priors)

    def lnlikelihood(self, theta):
        return -jnp.sum(jnp.asarray(theta) ** 2)


class DummyDynestySampler(samplers.DynestySampling):
    def __init__(self):
        self.priors = {
            "x": distributions.uniform(loc=1.0, scale=2.0),
            "y": distributions.uniform(loc=-1.0, scale=4.0),
        }
        self.ndims = len(self.priors)
        self.likelihoodScale = 1.0

    def lnlikelihood(self, theta, **kwargs):
        return -jnp.sum(jnp.asarray(theta) ** 2)


def test_emcee_lnprior_is_finite_inside_support_and_infinite_outside():
    sampler = DummyEmceeSampler()

    assert np.isfinite(sampler.lnprior(jnp.array([1.0, 0.0])))
    assert np.isneginf(sampler.lnprior(jnp.array([3.0, 0.0])))


def test_emcee_lnpost_adds_prior_and_likelihood():
    sampler = DummyEmceeSampler()
    theta = jnp.array([1.0, 0.5])

    expected = sampler.lnprior(theta) + sampler.lnlikelihood(theta)

    assert np.isclose(sampler.lnpost(theta), expected)


def test_emcee_init_samples_uses_prior_transforms(monkeypatch):
    sampler = DummyEmceeSampler()
    monkeypatch.setattr(
        np.random,
        "uniform",
        lambda low, high, size: np.full(size, 0.5),
    )

    samples = sampler.initSamples(nchains=3)

    assert samples.shape == (3, 2)
    assert np.allclose(samples[:, 0], 1.0)
    assert np.allclose(samples[:, 1], 0.0)


@pytest.mark.parametrize(
    ("avg_dtau", "conservative", "total_steps", "expected"),
    [
        (1e-4, False, 10, False),
        (1e-6, False, 10, True),
        (1e-6, True, 100, False),
        (1e-6, True, 1000, True),
    ],
)
def test_emcee_stop_check(avg_dtau, conservative, total_steps, expected):
    sampler = DummyEmceeSampler()

    result = sampler._stopCheck(
        tau=np.array([10.0]),
        avgDtau=avg_dtau,
        DtauLimit=1e-5,
        totalSteps=total_steps,
        conservative=conservative,
    )

    assert bool(result) is expected


def test_dynesty_prior_transform_preserves_prior_order():
    sampler = DummyDynestySampler()

    theta = sampler.ptform(jnp.array([0.5, 0.5]))

    assert np.allclose(theta, jnp.array([2.0, 1.0]))


def test_dynesty_scaled_likelihood_applies_multiplier():
    sampler = DummyDynestySampler()
    sampler.likelihoodScale = 2.5
    theta = jnp.array([1.0, 2.0])

    assert np.isclose(
        sampler._scaledLnlikelihood(theta),
        2.5 * sampler.lnlikelihood(theta),
    )


def test_dynesty_init_samples_returns_finite_live_points(monkeypatch):
    sampler = DummyDynestySampler()
    values = np.linspace(0.1, 0.9, 30).reshape(15, 2)
    monkeypatch.setattr(
        np.random,
        "uniform",
        lambda low, high, size: values.copy(),
    )

    u, v, logl = sampler.initSamples(
        ndims=sampler.ndims,
        nlive=5,
        nliveMult=3,
    )

    assert u.shape == (5, 2)
    assert v.shape == (5, 2)
    assert logl.shape == (5,)
    assert np.all(np.isfinite(logl))


@pytest.mark.integration
@pytest.mark.skip(reason="Full dynesty sampling is covered by integration tests.")
def test_dynesty_run_sampler():
    sampler = DummyDynestySampler()
    sampler.runSampler(minSamples=10, sampler_kwargs={"nlive": 5})
