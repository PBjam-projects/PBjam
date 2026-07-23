"""Tests for plotting numerical helpers."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from pbjam import plotting


def test_smooth_power_preserves_shape_and_smooths_a_spike():
    freq = np.linspace(0.0, 10.0, 101)
    power = np.zeros_like(freq)
    power[50] = 1.0

    smoothed = plotting.smooth_power(
        freq,
        power,
        smooth_filter_width=0.3,
    )

    assert smoothed.shape == power.shape
    assert np.all(np.isfinite(smoothed))
    assert 0.0 < smoothed[50] < 1.0
    assert np.isclose(smoothed[49], smoothed[51])


def test_echelle_returns_consistent_grid_and_repeated_rows():
    freq = np.linspace(0.0, 60.0, 601)
    power = 1.0 + freq

    x, y, z = plotting.echelle(
        freq,
        power,
        dnu=10.0,
        fmin=10.0,
        fmax=50.0,
        sampling=1.0,
    )

    assert x.shape == (101,)
    assert y.shape == (10,)
    assert z.shape == (8, 100)
    assert np.allclose(z[0], z[1])
    assert np.allclose(z[2], z[3])
    assert x[0] == 0.0
    assert x[-1] == 10.0
    assert y[0] == 10.0
    assert y[-1] == 50.0


def test_echellify_freqs_applies_modulo_and_preserves_frequency():
    frequencies = np.array([100.0, 107.0, 114.0])

    x, y = plotting._echellify_freqs(
        frequencies,
        dnu=10.0,
        offset=0.25,
    )

    assert np.allclose(x, [7.5, 4.5, 1.5])
    assert np.array_equal(y, frequencies)


def test_as_finite_array_flattens_and_filters():
    values = [[1.0, np.nan], [np.inf, 2.0]]

    result = plotting._asFiniteArray(values)

    assert np.array_equal(result, np.array([1.0, 2.0]))


def test_echelle_lower_limit_is_clamped_to_available_frequency():
    assert plotting._getEchelleYlim(
        f=np.array([80.0, 90.0, 100.0]),
        N_p=7,
        numax=100.0,
        dnu=10.0,
    ) == 80.0

    assert plotting._getEchelleYlim(
        f=np.array([]),
        N_p=7,
        numax=100.0,
        dnu=10.0,
    ) == 60.0


def test_set_sample_to_plot_uses_prior_transform(monkeypatch):
    model = SimpleNamespace(
        ndims=2,
        priors={"a": object(), "b": object()},
        ptform=lambda u: np.array([u[0] + 1.0, u[1] + 2.0]),
        unpackSamples=lambda samples: {
            "a": samples[:, 0],
            "b": samples[:, 1],
        },
    )
    monkeypatch.setattr(
        plotting.np.random,
        "uniform",
        lambda low, high, size: np.array([0.25, 0.75]),
    )

    result = plotting._setSampleToPlot(
        model,
        N=3,
        unpacked=True,
        stage="prior",
    )

    assert np.allclose(result["a"], 1.25)
    assert np.allclose(result["b"], 2.75)


def test_plot_echelle_reuses_supplied_axis():
    freq = np.linspace(10.0, 60.0, 501)
    power = np.ones_like(freq)
    fig, ax = plt.subplots()

    returned = plotting.plot_echelle(
        freq,
        power,
        numax=35.0,
        dnu=10.0,
        ax=ax,
        sampling=1.0,
    )

    assert returned is ax
    assert len(ax.images) == 1
    plt.close(fig)
