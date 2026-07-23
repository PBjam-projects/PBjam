"""Tests for mode-identification orchestration."""

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import importlib

modeid_module = importlib.import_module("pbjam.modeID")


@pytest.fixture
def obs():
    return {
        "numax": (100.0, 5.0),
        "dnu": (10.0, 0.5),
        "teff": (5777.0, 50.0),
        "bp_rp": (0.8, 0.05),
    }


@pytest.fixture
def mode_identifier(monkeypatch, obs):
    monkeypatch.setattr(modeid_module.IO, "_getPriorPath", lambda: "/tmp/prior.csv")
    return modeid_module.modeID(
        f=np.arange(0.0, 201.0),
        s=np.ones(201),
        obs=obs,
        N_p=7,
    )


def make_result(ells, start=0.0, n_samples=4, scalar=1.0):
    ells = np.asarray(ells)
    n_modes = len(ells)
    freq = np.arange(start, start + n_modes, dtype=float)
    samples = np.tile(freq, (n_samples, 1))
    return {
        "ell": ells,
        "enn": np.arange(n_modes),
        "emm": np.zeros(n_modes),
        "zeta": np.linspace(0.0, 1.0, n_modes),
        "summary": {
            "freq": np.vstack((freq, np.full(n_modes, 0.1))),
            "height": np.vstack((freq + 1.0, np.full(n_modes, 0.2))),
            "width": np.vstack((freq + 2.0, np.full(n_modes, 0.3))),
            "rotAsym": np.vstack((np.zeros(n_modes), np.full(n_modes, 0.01))),
            "numax": np.array([scalar, 0.1]),
            "dnu": np.array([scalar + 1.0, 0.2]),
        },
        "samples": {
            "freq": samples,
            "height": samples + 1.0,
            "width": samples + 2.0,
            "rotAsym": np.zeros_like(samples),
            "numax": np.full(n_samples, scalar),
            "dnu": np.full(n_samples, scalar + 1.0),
        },
    }


def test_init_sets_default_window_and_prior_path(mode_identifier):
    assert mode_identifier.freqLimits == [20.0, 180.0]
    assert mode_identifier.priorPath == "/tmp/prior.csv"
    assert np.array_equal(np.where(mode_identifier.sel)[0], np.arange(21, 180))
    assert mode_identifier.Nyquist == 200.0


@pytest.mark.parametrize(
    ("teff", "dnu", "expected"),
    [
        (6000.0, 70.0, "ms"),
        (5800.0, 50.0, "sg"),
        (5000.0, 20.0, "rgb"),
    ],
)
def test_select_model(teff, dnu, expected):
    obj = modeid_module.modeID.__new__(modeid_module.modeID)
    obj.obs = {"teff": (teff, 50.0), "dnu": (dnu, 1.0)}
    assert obj.selectModel() == expected


def test_unpack_prior_kwargs_combines_flat_and_nested_values(mode_identifier):
    prior_kwargs = {
        "selectivePrior": False,
        "PCAsamples": 50,
        "l20": {"PCAsamples": 100, "PCAdims": 4},
        "runl1model": {"PCAdims": 8},
    }

    assert mode_identifier._unpackPriorKwargs(prior_kwargs, "l20") == {
        "selectivePrior": False,
        "PCAsamples": 100,
        "PCAdims": 4,
    }
    assert mode_identifier._unpackPriorKwargs(prior_kwargs, "l1") == {
        "selectivePrior": False,
        "PCAsamples": 50,
        "PCAdims": 8,
    }
    assert mode_identifier._unpackPriorKwargs(None, "ms") == {}


class DummyModel:
    calls = []
    parsed_result = None

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.likelihoodScale = None
        type(self).calls.append(self)

    def runSampler(self, **kwargs):
        self.sampler_kwargs = kwargs
        return np.array([[1.0], [2.0]])

    def unpackSamples(self, samples):
        self.unpacked_input = samples
        return {"theta": samples[:, 0]}

    def parseSamples(self, samples):
        self.parsed_input = samples
        return self.parsed_result


@pytest.mark.parametrize(
    ("method_name", "model_attr", "class_attr", "result_attr", "samples_attr"),
    [
        ("runl20model", "l20model", "Asyl20model", "l20result", "l20Samples"),
        ("runMSmodel", "MSmodel", "Asyl021model", "MSresult", "MSSamples"),
    ],
)
def test_primary_model_stages_forward_sampler_and_prior_options(
    monkeypatch,
    mode_identifier,
    method_name,
    model_attr,
    class_attr,
    result_attr,
    samples_attr,
):
    DummyModel.calls = []
    DummyModel.parsed_result = make_result([0, 2], scalar=10.0)
    monkeypatch.setattr(modeid_module, class_attr, DummyModel)
    mode_identifier.mergeResults = lambda **kwargs: {"merged": kwargs}

    result = getattr(mode_identifier, method_name)(
        progress=False,
        dynamic=True,
        minSamples=12,
        sampler_kwargs={"nlive": 20},
        logl_kwargs={"scale": 2},
        loglikelihoodMultiplier=3,
        PCAsamples=25,
        PCAdims=4,
        selectivePrior=False,
        selectivePriorN=200,
        selectivePriorMin=10,
        selectivePriorSigma=2.5,
        selectivePriorSeed=7,
    )

    model = getattr(mode_identifier, model_attr)
    assert model is DummyModel.calls[-1]
    assert np.array_equal(model.args[0], mode_identifier.f[mode_identifier.sel])
    assert np.array_equal(model.args[1], mode_identifier.s[mode_identifier.sel])
    assert model.args[2] is mode_identifier.obs
    assert model.args[3] is mode_identifier.addPriors
    assert model.likelihoodScale == 3.0
    assert model.kwargs["priorPath"] == "/tmp/prior.csv"
    assert model.kwargs["selectivePrior"] is False
    assert model.kwargs["selectivePriorN"] == 200
    assert model.sampler_kwargs == {
        "progress": False,
        "dynamic": True,
        "minSamples": 12,
        "logl_kwargs": {"scale": 2},
        "sampler_kwargs": {"nlive": 20},
    }
    assert getattr(mode_identifier, result_attr) is result
    assert np.array_equal(getattr(mode_identifier, samples_attr), np.array([[1.0], [2.0]]))


@pytest.mark.parametrize(
    ("model_name", "class_attr"),
    [("ms", "Asyl1model"), ("sg", "Mixl1model"), ("rgb", "RGBl1model")],
)
def test_run_l1_model_selects_class_and_builds_residual(
    monkeypatch,
    mode_identifier,
    model_name,
    class_attr,
):
    DummyModel.calls = []
    DummyModel.parsed_result = make_result([1, 1], start=105.0, scalar=20.0)
    monkeypatch.setattr(modeid_module, class_attr, DummyModel)

    l20_result = make_result([2, 0, 2, 0], start=98.0, scalar=10.0)
    l20_result["summary"].update({
        "env_height": np.array([3.0, 0.2]),
        "env_width": np.array([20.0, 1.0]),
        "mode_width": np.array([0.2, 0.02]),
        "teff": np.array([5777.0, 50.0]),
        "bp_rp": np.array([0.8, 0.05]),
    })
    mode_identifier.l20result = l20_result
    mode_identifier.l20model = SimpleNamespace(
        getMedianModel=lambda: np.full(np.sum(mode_identifier.sel), 2.0)
    )
    mode_identifier.mergeResults = lambda **kwargs: {"merged": kwargs}

    result = mode_identifier.runl1model(
        model=model_name,
        progress=False,
        dynamic=True,
        minSamples=8,
        sampler_kwargs={"nlive": 10},
        logl_kwargs={"foo": "bar"},
        loglikelihoodMultiplier=2.5,
        PCAsamples=30,
        PCAdims=5,
        selectivePrior=False,
        selectivePriorSeed=4,
    )

    model = DummyModel.calls[-1]
    assert np.allclose(mode_identifier.l20residual, 0.5)
    assert np.array_equal(model.args[0], mode_identifier.f[mode_identifier.sel])
    assert np.allclose(model.args[1], 0.5)
    summary = model.args[2]
    assert np.array_equal(summary["n_p"], np.array([1, 3]))
    assert np.array_equal(summary["nu0_p"], np.array([99.0, 101.0]))
    assert model.likelihoodScale == 2.5
    assert result is mode_identifier.l1result

    if model_name == "sg":
        assert model.kwargs["selectivePrior"] is False
        assert model.kwargs["selectivePriorSeed"] == 4
    elif model_name == "rgb":
        assert model.kwargs["rootiter"] == 15
        assert model.kwargs["modelChoice"] == "simple"


def test_run_l1_model_rejects_unknown_model(mode_identifier):
    mode_identifier.l20model = SimpleNamespace(
        getMedianModel=lambda: np.ones(np.sum(mode_identifier.sel))
    )
    result = make_result([0], scalar=1.0)
    result["summary"].update({
        key: np.array([1.0, 0.1])
        for key in ["env_height", "env_width", "mode_width", "teff", "bp_rp"]
    })
    mode_identifier.l20result = result

    with pytest.raises(ValueError, match="invalid"):
        mode_identifier.runl1model(model="unknown")


def test_call_runs_main_sequence_stage_only(mode_identifier):
    calls = []
    mode_identifier.runMSmodel = lambda *args, **kwargs: calls.append(("ms", args, kwargs))
    mode_identifier.runl20model = lambda *args, **kwargs: calls.append(("l20", args, kwargs))
    mode_identifier.runl1model = lambda *args, **kwargs: calls.append(("l1", args, kwargs))

    mode_identifier(
        model="ms",
        progress=False,
        dynamic=True,
        sampler_kwargs={"nlive": 25},
        prior_kwargs={"PCAsamples": 40, "ms": {"PCAdims": 3}},
    )

    assert [call[0] for call in calls] == ["ms"]
    assert calls[0][2]["PCAsamples"] == 40
    assert calls[0][2]["PCAdims"] == 3


@pytest.mark.parametrize("model", ["sg", "rgb"])
def test_call_runs_two_stage_workflow_and_stage_specific_priors(mode_identifier, model):
    calls = []
    mode_identifier.runl20model = lambda *args, **kwargs: calls.append(("l20", kwargs))
    mode_identifier.runl1model = lambda *args, **kwargs: calls.append(("l1", kwargs))

    mode_identifier(
        model=model,
        prior_kwargs={
            "selectivePrior": False,
            "l20": {"PCAdims": 4},
            "l1": {"PCAdims": 6},
        },
    )

    assert [call[0] for call in calls] == ["l20", "l1"]
    assert calls[0][1]["PCAdims"] == 4
    assert calls[1][1]["PCAdims"] == 6
    assert calls[1][1]["model"] == model


def test_call_rejects_invalid_model(mode_identifier):
    with pytest.raises(ValueError, match="invalid"):
        mode_identifier(model="not-a-model")


def test_merge_results_uses_common_sample_count_and_l20_scalar_values(mode_identifier):
    l1 = make_result([1, 1], start=105.0, n_samples=3, scalar=20.0)
    l20 = make_result([2, 0], start=98.0, n_samples=5, scalar=10.0)

    merged = mode_identifier.mergeResults(l20result=l20, l1result=l1, N=4)

    assert np.array_equal(merged["ell"], np.array([1, 1, 2, 0]))
    assert merged["samples"]["freq"].shape == (3, 4)
    assert merged["summary"]["freq"].shape == (2, 4)
    assert np.array_equal(merged["summary"]["numax"], l20["summary"]["numax"])
    assert np.array_equal(merged["samples"]["numax"], l20["samples"]["numax"])


def test_store_result_writes_pickle_and_scalar_summary_csv(mode_identifier, tmp_path):
    result = make_result([0, 2], scalar=10.0)
    mode_identifier.result = result

    mode_identifier.storeResult(result, path=tmp_path, ID="target")

    pkl_path = tmp_path / "target_modeIDresult.pkl"
    csv_path = tmp_path / "target_modeIDresult.csv"
    assert pkl_path.exists()
    assert csv_path.exists()

    with pkl_path.open("rb") as handle:
        stored = pickle.load(handle)
    assert np.array_equal(stored["ell"], result["ell"])

    table = pd.read_csv(csv_path)
    assert set(table["name"]) == {"rotAsym", "numax", "dnu"}
