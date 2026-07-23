"""Regression tests for custom-prior parameter ordering (issue #295)."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from pbjam.l1models import Mixl1model
from pbjam.l20models import Asyl20model
from pbjam.MSmodels import Asyl021model


@pytest.mark.parametrize("model_cls", [Asyl20model, Asyl021model, Mixl1model])
def test_unpack_params_uses_prior_sampling_order(model_cls):
    """Non-PCA parameters must be labelled in the order sampled by ptform."""

    model = model_cls.__new__(model_cls)
    model.DR = SimpleNamespace(
        dimsR=2,
        inverse_transform=lambda theta: jnp.array([10.0, 20.0]),
    )
    model.pcaLabels = ["pca_a", "pca_b"]

    # This is intentionally different from the insertion order of priors.
    # The old implementation used addLabels and therefore swapped values.
    model.addLabels = ["H3_power", "nurot_e", "inc"]
    model.priors = {
        "theta_0": object(),
        "theta_1": object(),
        "nurot_e": object(),
        "inc": object(),
        "H3_power": object(),
    }
    model.logpars = []

    theta = jnp.array([0.1, 0.2, 30.0, 40.0, 50.0])
    unpacked = model.unpackParams(theta)

    assert unpacked["pca_a"] == pytest.approx(10.0)
    assert unpacked["pca_b"] == pytest.approx(20.0)
    assert unpacked["nurot_e"] == pytest.approx(30.0)
    assert unpacked["inc"] == pytest.approx(40.0)
    assert unpacked["H3_power"] == pytest.approx(50.0)
