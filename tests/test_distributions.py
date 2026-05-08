"""Tests for the distributions module."""

import numpy as np
import jax.numpy as jnp

from pbjam import distributions


def test_beta_ppf_uses_jax_compatible_clip():
    dist = distributions.beta(a=2, b=3)

    q = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    x = dist.ppf(q)

    assert np.all(np.isfinite(x))
    assert np.all(x >= 0.0)
    assert np.all(x <= 1.0)
    assert np.isclose(dist.ppf(0.0), 0.0)
    assert np.isclose(dist.ppf(1.0), 1.0)
