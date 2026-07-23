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


def make_ms_modeid():
    """Create a lightweight modeID-like object with an active MS model."""

    modeid_class = type("modeID", (plotting.plotting,), {})
    obj = modeid_class()
    obj.f = np.linspace(10.0, 60.0, 101)
    obj.s = np.ones_like(obj.f) * 2.0
    obj.sel = (obj.f >= 20.0) & (obj.f <= 50.0)
    obj.N_p = 3
    obj.obs = {
        "numax": (35.0, 1.0),
        "dnu": (10.0, 0.5),
    }

    class DummyMSModel:
        ndims = 1
        samples = np.ones((4, 1))
        pcaLabels = ["dnu"]
        addLabels = ["d01", "d02"]
        logpars = []
        priors = {"theta_0": object()}

        def getMedianModel(self):
            return np.ones(np.sum(obj.sel)) * 2.0

        def ptform(self, u):
            return np.asarray(u)

        def unpackParams(self, theta):
            return {
                "numax": 35.0,
                "dnu": 10.0,
                "eps_p": 1.0,
                "alpha_p": 0.0,
                "d01": 5.0,
                "d02": 2.0,
            }

        def asymptotic_nu_p(self, **kwargs):
            return np.array([25.0, 35.0, 45.0]), np.array([2, 3, 4])

        def model(self, theta):
            return np.ones(np.sum(obj.sel)) * 2.0

        def unpackSamples(self, samples):
            return {"theta_0": np.asarray(samples)[:, 0]}

    obj.MSmodel = DummyMSModel()
    obj.MSSamples = np.ones((4, 1))
    obj.MSresult = {
        "ell": np.array([0, 1, 2]),
        "emm": np.zeros(3),
        "summary": {
            "dnu": np.array([10.0, 0.5]),
            "numax": np.array([35.0, 1.0]),
            "eps_p": np.array([1.0, 0.1]),
            "freq": np.array([[30.0, 35.0, 38.0], [0.1, 0.1, 0.1]]),
        },
        "samples": {
            "dnu": np.full(4, 10.0),
            "freq": np.tile([30.0, 35.0, 38.0], (4, 1)),
        },
    }
    obj.result = obj.MSresult

    return obj


def test_active_ms_model_detection_ignores_stale_two_stage_models():
    obj = make_ms_modeid()

    assert plotting._usesMSModel(obj)

    obj.l20model = object()
    obj.l1model = object()
    assert plotting._usesMSModel(obj)

    obj.result = {"different": "result"}
    assert not plotting._usesMSModel(obj)


def test_ms_prior_echelle_plots_all_three_ridges(monkeypatch):
    obj = make_ms_modeid()
    fig, ax = plt.subplots()

    monkeypatch.setattr(plotting, "_baseEchelle", lambda *args, **kwargs: (fig, ax))
    monkeypatch.setattr(plotting.jax, "jit", lambda function: function)
    monkeypatch.setattr(
        plotting.np.random,
        "uniform",
        lambda low, high, size: np.full(size, 0.5),
    )

    returned_fig, returned_ax = plotting._ModeIDClassPriorEchelle(
        obj,
        Nsamples=2,
        scale=1 / 300,
        colors=plotting.ellColors,
    )

    assert returned_fig is fig
    assert returned_ax is ax
    assert len(ax.collections) == 9  # six ridge samples plus three legend entries
    assert ax.get_xlim() == (0.0, 10.0)
    plt.close(fig)


def test_ms_model_base_frames_include_residual_panel():
    obj = make_ms_modeid()

    fig, axes = plotting._makeBaseFrames(obj)

    assert len(axes) == 3
    residual = axes[2].lines[0].get_ydata()
    assert np.allclose(residual[5:-5], 1.0)
    assert axes[2].get_ylabel() == "Residual"
    plt.close(fig)


def test_ms_prior_and_posterior_spectrum_plot_model_samples(monkeypatch):
    obj = make_ms_modeid()
    monkeypatch.setattr(plotting.jax, "jit", lambda function: function)
    monkeypatch.setattr(
        plotting.np.random,
        "uniform",
        lambda low, high, size: np.full(size, 0.5),
    )
    monkeypatch.setattr(
        plotting.np.random,
        "randint",
        lambda low, high=None, size=None: np.zeros(size, dtype=int),
    )

    prior_fig, prior_axes = plotting._ModeIDClassPriorSpectrum(obj, N=2)
    posterior_fig, posterior_axes = plotting._ModeIDClassPostSpectrum(obj, N=2)

    assert len(prior_axes[0].lines) >= 3
    assert len(prior_axes[1].lines) >= 3
    assert len(posterior_axes[0].lines) >= 4
    assert len(posterior_axes[1].lines) >= 3

    plt.close(prior_fig)
    plt.close(posterior_fig)


def test_ms_model_dispatches_corner_and_reference(monkeypatch):
    obj = make_ms_modeid()
    corner_calls = []
    reference_calls = []

    def fake_corner(owner, model, unpacked, count, **kwargs):
        corner_calls.append((owner, model, unpacked, count))
        return "corner-figure", "corner-axes"

    def fake_reference(model):
        reference_calls.append(model)
        return "reference-figure", "reference-axes"

    monkeypatch.setattr(plotting, "_ModeIDClassPriorCorner", fake_corner)
    monkeypatch.setattr(plotting, "_ModeIDPriorReference", fake_reference)

    figures, axes = obj.corner(stage="prior", N=12)
    reference_figures, reference_axes = obj.reference(stage="prior")

    assert figures == ["corner-figure"]
    assert axes == ["corner-axes"]
    assert corner_calls == [(obj, obj.MSmodel, False, 12)]
    assert reference_figures == ["reference-figure"]
    assert reference_axes == ["reference-axes"]
    assert reference_calls == [obj.MSmodel]


def test_ms_public_echelle_and_spectrum_dispatch(monkeypatch):
    obj = make_ms_modeid()
    fig, ax = plt.subplots()
    spectrum_fig, spectrum_axes = plt.subplots(2, 1)

    monkeypatch.setattr(
        plotting,
        "_ModeIDClassPriorEchelle",
        lambda owner, **kwargs: (fig, ax),
    )
    monkeypatch.setattr(
        plotting,
        "_ModeIDClassPriorSpectrum",
        lambda owner, count, **kwargs: (spectrum_fig, spectrum_axes),
    )

    returned_fig, returned_ax = obj.echelle(stage="prior", ID="MS target")
    returned_spectrum_fig, returned_spectrum_axes = obj.spectrum(
        stage="prior",
        ID="MS target",
    )

    assert returned_fig is fig
    assert returned_ax is ax
    assert ax.get_title() == "MS target"
    assert returned_spectrum_fig is spectrum_fig
    assert returned_spectrum_axes is spectrum_axes
    assert spectrum_axes[0].get_title() == "MS target"

    plt.close(fig)
    plt.close(spectrum_fig)
