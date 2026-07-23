"""Tests for dipole-mode model implementations."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from pbjam import l1models
from pbjam.l1models import Asyl1model, Mixl1model, RGBl1model, commonFuncs


@pytest.fixture
def common_model():
    model = commonFuncs()
    model.vis = {"V10": 1.25}
    model.obs = {
        "nu0_p": jnp.array([100.0, 110.0]),
        "env_height": (2.0, 0.1),
        "numax": (105.0, 1.0),
        "env_width": (20.0, 1.0),
    }
    return model


def test_modewidths_scale_with_p_mode_fraction_and_apply_floor(common_model):
    widths = common_model.modewidths(
        Gamma=2.0,
        zeta=jnp.array([0.0, 0.25, 1.0, 1.5]),
        fac=0.5,
    )

    assert np.allclose(widths[:2], [1.0, 0.75])
    assert widths[2] == pytest.approx(1e-6)
    assert widths[3] == pytest.approx(1e-6)


def test_heights_are_visibility_scaled_envelope(common_model):
    frequencies = jnp.array([95.0, 105.0, 115.0])

    heights = common_model.heights(frequencies)

    expected = common_model.vis["V10"] * l1models.jar.envelope(
        frequencies,
        2.0,
        105.0,
        20.0,
    )
    assert np.allclose(heights, expected)
    assert heights[1] > heights[0]
    assert heights[0] == pytest.approx(heights[2])


def test_g_mode_periods_have_constant_spacing(common_model):
    orders = jnp.array([10.0, 11.0, 12.0])
    frequencies = common_model.asymptotic_nu_g(
        orders,
        DPi1=80.0,
        eps_g=0.25,
    )

    periods_seconds = 1e6 / np.asarray(frequencies)

    assert np.allclose(np.diff(periods_seconds), 80.0)


def test_asymptotic_dipole_p_modes_are_radial_modes_plus_d01(common_model):
    assert np.array_equal(
        common_model.asymptotic_nu_p(5.0),
        np.array([105.0, 115.0]),
    )


@pytest.mark.parametrize(
    "multiplet",
    [
        np.array([99.0, 100.0, 101.0]),
        np.array([98.5, 100.0, 101.5]),
    ],
)
def test_symmetric_triplets_have_zero_asymmetry(common_model, multiplet):
    assert common_model.asymmetry(multiplet, m=1) == pytest.approx(0.0)
    assert common_model.asymmetry(multiplet, m=-1) == pytest.approx(0.0)


def test_asymmetry_rejects_invalid_azimuthal_order(common_model):
    with pytest.raises(ValueError, match="m must be"):
        common_model.asymmetry(np.array([99.0, 100.0, 101.0]), m=0)


def test_asyl1_frequencies_have_zero_mixing():
    model = Asyl1model.__new__(Asyl1model)
    model.obs = {"nu0_p": jnp.array([100.0, 110.0])}

    frequencies, zeta = model.nu1_frequencies({"d01": 5.0})

    assert np.array_equal(frequencies, np.array([105.0, 115.0]))
    assert np.array_equal(zeta, np.zeros(2))


def test_asyl1_unpack_params_follows_prior_order_and_expands_logs():
    model = Asyl1model.__new__(Asyl1model)
    model.priors = {
        "d01": object(),
        "nurot_e": object(),
        "inc": object(),
    }
    model.logpars = ["nurot_e"]

    unpacked = model.unpackParams(jnp.array([5.0, 1.0, 0.5]))

    assert unpacked["d01"] == pytest.approx(5.0)
    assert unpacked["nurot_e"] == pytest.approx(10.0)
    assert unpacked["inc"] == pytest.approx(0.5)


def test_asyl1_model_at_zero_inclination_contains_only_central_components(
    monkeypatch,
):
    model = Asyl1model.__new__(Asyl1model)
    model.f = jnp.array([90.0, 100.0])
    model.ones_nu = jnp.ones_like(model.f)
    model.obs = {
        "nu0_p": jnp.array([100.0, 110.0]),
        "mode_width": (0.1, 0.01),
        "env_height": (2.0, 0.1),
        "numax": (105.0, 1.0),
        "env_width": (20.0, 1.0),
    }
    model.vis = {"V10": 1.0}

    monkeypatch.setattr(
        l1models.jar,
        "lor",
        lambda frequencies, centre, height, width: (
            jnp.ones_like(frequencies) * height
        ),
    )

    result = model.model(
        {"d01": 5.0, "nurot_e": 0.5, "inc": 0.0}
    )

    expected_heights = model.heights(jnp.array([105.0, 115.0]))
    assert np.allclose(result, 1.0 + expected_heights.sum())


def test_mix_make_empties_builds_block_matrices():
    model = Mixl1model.__new__(Mixl1model)
    model.f = jnp.array([1.0, 2.0])
    model.N_p = 2
    model.N_g = 3

    model.makeEmpties()

    assert model.ones_block.shape == (2, 3)
    assert model.zeros_block.shape == (2, 3)
    assert model.eye_N_p.shape == (2, 2)
    assert model.eye_N_g.shape == (3, 3)
    assert model.D_gamma.shape == (5, 5)
    assert np.array_equal(np.diag(model.D_gamma), [0, 0, 1, 1, 1])


def test_mix_generate_matrices_are_symmetric():
    model = Mixl1model.__new__(Mixl1model)
    model.f = jnp.array([1.0])
    model.N_p = 2
    model.N_g = 2
    model.makeEmpties()

    L, D = model.generate_matrices(
        nu_p=jnp.array([100.0, 120.0]),
        nu_g=jnp.array([80.0, 110.0]),
        p_L=0.01,
        p_D=0.02,
    )

    assert L.shape == (4, 4)
    assert D.shape == (4, 4)
    assert np.allclose(L, L.T)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 1.0)


def test_mix_zero_coupling_recovers_pure_frequencies_and_mixing_labels():
    model = Mixl1model.__new__(Mixl1model)
    model.f = jnp.array([1.0])
    model.N_p = 2
    model.N_g = 2
    model.makeEmpties()

    L, D = model.generate_matrices(
        nu_p=jnp.array([100.0, 120.0]),
        nu_g=jnp.array([80.0, 110.0]),
        p_L=0.0,
        p_D=0.0,
    )
    frequencies, zeta = model.new_modes(L, D)

    assert np.allclose(frequencies, [80.0, 100.0, 110.0, 120.0])
    assert np.allclose(zeta, [1.0, 0.0, 1.0, 0.0])


def test_generalized_eigenproblem_satisfies_matrix_equation():
    model = Mixl1model.__new__(Mixl1model)
    A = jnp.array([[2.0, 0.0], [0.0, 6.0]])
    B = jnp.array([[1.0, 0.0], [0.0, 2.0]])

    eigenvalues, eigenvectors = model.generalized_eig(A, B)

    for i in range(2):
        assert np.allclose(
            A @ eigenvectors[:, i],
            eigenvalues[i] * (B @ eigenvectors[:, i]),
        )


def test_mix_rotation_interpolates_between_envelope_and_core():
    model = Mixl1model.__new__(Mixl1model)

    splitting = model.rotation(
        zeta=jnp.array([0.0, 0.25, 1.0]),
        nurot_c=4.0,
        nurot_e=1.0,
    )

    assert np.allclose(splitting, [1.0, 1.75, 4.0])


def test_mix_nu1_frequencies_passes_asymptotic_modes_through_coupling():
    model = Mixl1model.__new__(Mixl1model)
    model.obs = {"nu0_p": jnp.array([100.0, 110.0])}
    model.n_g = jnp.array([10.0, 11.0])
    model.generate_matrices = lambda nu_p, nu_g, p_L, p_D: (
        (nu_p, nu_g),
        (p_L, p_D),
    )
    model.new_modes = lambda L, D: (
        jnp.concatenate(L),
        jnp.array([D[0], D[0], D[1], D[1]]),
    )

    frequencies, zeta = model.nu1_frequencies(
        {
            "d01": 5.0,
            "DPi1": 80.0,
            "eps_g": 0.25,
            "p_L": 0.2,
            "p_D": 0.8,
        }
    )

    expected_g = model.asymptotic_nu_g(
        model.n_g,
        80.0,
        0.25,
    )
    assert np.allclose(frequencies[:2], [105.0, 115.0])
    assert np.allclose(frequencies[2:], expected_g)
    assert np.allclose(zeta, [0.2, 0.2, 0.8, 0.8])


def test_rgb_nearest_returns_closest_candidates():
    model = RGBl1model.__new__(RGBl1model)

    result = model.nearest(
        jnp.array([1.0, 5.0, 9.0]),
        jnp.array([0.0, 6.0, 10.0]),
    )

    assert np.array_equal(result, np.array([0.0, 6.0, 10.0]))


def test_halley_iteration_moves_toward_quadratic_root():
    model = RGBl1model.__new__(RGBl1model)
    x = 1.5
    y = x**2 - 2.0
    yp = 2.0 * x
    ypp = 2.0

    updated = model.halley_iteration(x, y, yp, ypp)

    assert abs(updated - np.sqrt(2.0)) < abs(x - np.sqrt(2.0))


def test_rgb_couple_without_iterations_returns_p_and_g_modes():
    model = RGBl1model.__new__(RGBl1model)
    model.rootiter = 0
    model.obs = {"dnu": (10.0, 1.0)}

    result = model.couple(
        nu_p=jnp.array([100.0, 110.0]),
        nu_g=jnp.array([90.0, 95.0]),
        q_p=0.1,
        q_g=0.1,
        DPi1=80e-6,
    )

    assert np.array_equal(result, np.array([100.0, 110.0, 90.0, 95.0]))


def test_rgb_simple_frequency_model_applies_mixing_weighted_rotation():
    model = RGBl1model.__new__(RGBl1model)
    model.n_g = jnp.array([10.0])
    model.obs = {"nu0_p": jnp.array([100.0])}
    model.asymptotic_nu_g = lambda n_g, DPi1, eps_g: jnp.array([90.0])
    model.couple = lambda *args, **kwargs: jnp.array([95.0, 105.0])
    model.zeta_p = lambda *args, **kwargs: jnp.array([0.0, 1.0])

    frequencies, zeta = model.simpleFrequencies(
        {
            "DPi1": 80.0,
            "eps_g": 0.25,
            "d01": 5.0,
            "q": 0.1,
            "dnu": 10.0,
            "nurot_c": 4.0,
            "nurot_e": 1.0,
        }
    )

    expected_splitting = np.array([1.0, 2.0])
    assert frequencies.shape == (3, 2)
    assert np.allclose(frequencies[1], [95.0, 105.0])
    assert np.allclose(frequencies[0], frequencies[1] - expected_splitting)
    assert np.allclose(frequencies[2], frequencies[1] + expected_splitting)
    assert zeta.shape == (3, 2)
    assert np.allclose(zeta[0], [0.0, 1.0])
