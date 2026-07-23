"""Tests for the dimensionality reduction module."""

import numpy as np
import jax.numpy as jnp
import pytest

from pbjam import DR


class DeterministicRNG:
    def normal(self, loc, scale, size):
        return np.zeros(size)

    def uniform(self, low, high, size):
        return np.zeros(size)


def test_refine_prior_doubles_sigma_inflation_on_retry():
    pca = DR.PCA.__new__(DR.PCA)
    pca.ppf = True
    pca.dimsR = 1
    pca.selectLabels = ['numax']
    pca.varLabels = ['numax']
    pca.obs = {'numax': (10.0, 1.0)}
    pca.latentPriorLoc = np.array([0.0])
    pca.latentPriorScale = np.array([1.0])

    pca.inverse_transform = lambda latent: jnp.zeros((latent.shape[0], 1))
    pca.transform = lambda physical: physical
    pca.setLatentNormalPrior = lambda latent: setattr(pca, 'prior_sample', latent)

    with pytest.warns(UserWarning, match="fewer than minAccepted"):
        pca.refinePriorByObservables(
            N=1,
            minAccepted=2,
            sigmaInflation=3,
            rng=DeterministicRNG(),
        )

    assert pca.selectivePriorInfo['draws'] == 2
    assert pca.selectivePriorInfo['accepted'] == 2
    assert pca.selectivePriorInfo['sigmaInflation'] == 6
