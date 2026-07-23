"""Tests for the background module."""

import numpy as np

from pbjam import background


def test_harvey_profile_matches_definition(monkeypatch):
    monkeypatch.setattr(background.jar, "attenuation", lambda nu, nyquist: np.ones_like(nu))

    nu = np.array([0.0, 10.0, 20.0])
    model = background.bkgModel(nu, Nyquist=100.0)

    result = model.harvey(nu, a=12.0, b=4.0, c=2.0)
    expected = 12.0 / 4.0 / (1.0 + (nu / 4.0) ** 2)

    assert np.allclose(result, expected)


def test_background_combines_components_attenuation_and_shot_noise(monkeypatch):
    attenuation = np.array([1.0, 0.5, 0.25])
    monkeypatch.setattr(
        background.jar,
        "attenuation",
        lambda nu, nyquist: attenuation,
    )

    nu = np.array([1.0, 2.0, 3.0])
    model = background.bkgModel(nu, Nyquist=10.0)

    theta = {
        "H_power": 8.0,
        "H1_nu": 1.0,
        "H1_exp": 2.0,
        "H2_nu": 2.0,
        "H2_exp": 4.0,
        "H3_power": 3.0,
        "H3_nu": 3.0,
        "H3_exp": 2.0,
        "shot": 0.7,
    }

    expected_components = (
        model.harvey(nu, theta["H_power"], theta["H1_nu"], theta["H1_exp"])
        + model.harvey(nu, theta["H_power"], theta["H2_nu"], theta["H2_exp"])
        + model.harvey(nu, theta["H3_power"], theta["H3_nu"], theta["H3_exp"])
    )
    expected = expected_components * attenuation**2 + theta["shot"]

    assert np.allclose(model.eta, attenuation**2)
    assert np.allclose(model(theta), expected)
