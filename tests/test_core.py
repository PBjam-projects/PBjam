"""Tests for the core module."""

import numpy as np
import pytest

from pbjam import core


def valid_obs():
    return {
        'numax': (100.0, 5.0),
        'dnu': (10.0, 0.5),
        'teff': (5777.0, 50.0),
    }


def test_session_array_timeseries_builds_psd_for_each_target(monkeypatch):
    calls = []

    class DummyPSD:
        def __init__(self, ID, **kwargs):
            calls.append(ID)
            self.freq = np.array([1.0, 2.0])
            self.powerdensity = np.array([3.0, 4.0])

        def __call__(self):
            return None

    monkeypatch.setattr(core.IO, 'psd', DummyPSD)
    monkeypatch.setattr(core, 'star', lambda *args, **kwargs: {'args': args, 'kwargs': kwargs})

    timeseries = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
    ])

    sess = core.session(['target-a', 'target-b'], valid_obs(), timeseries=timeseries)

    assert calls == ['target-a', 'target-b']
    assert np.allclose(sess.inputs['target-a']['f'], [1.0, 2.0])
    assert np.allclose(sess.inputs['target-b']['s'], [3.0, 4.0])


def test_validate_obs_raises_explicit_exceptions():
    with pytest.raises(TypeError, match='Entries in obs'):
        core._validateObs({'numax': 100.0, 'dnu': (10.0, 0.5), 'teff': (5777.0, 50.0)}, 'target')

    with pytest.raises(ValueError, match='Entries in obs'):
        core._validateObs({'numax': (100.0,), 'dnu': (10.0, 0.5), 'teff': (5777.0, 50.0)}, 'target')

    with pytest.raises(ValueError, match='Missing teff'):
        core._validateObs({'numax': (100.0, 5.0), 'dnu': (10.0, 0.5)}, 'target')


def test_session_rejects_invalid_obs_type():
    with pytest.raises(TypeError, match='obs argument must be a dictionary'):
        core.session('target', obs=[], spectrum=np.array([[1.0], [2.0]]))


def test_convert_to_list_handles_supported_inputs():
    assert core._convertToList("target") == ["target"]
    assert core._convertToList(("a", "b")) == ["a", "b"]
    assert core._convertToList(np.array(["a", "b"])) == ["a", "b"]
    values = ["a", "b"]
    assert core._convertToList(values) is values

    with pytest.raises(TypeError, match="Unsupported type"):
        core._convertToList({"a", "b"})


def test_session_array_spectrum_is_shared_across_targets(monkeypatch):
    created = []
    monkeypatch.setattr(core, "star", lambda *args, **kwargs: created.append((args, kwargs)) or kwargs)
    spectrum = np.array([[1.0, 2.0], [3.0, 4.0]])

    sess = core.session(["a", "b"], valid_obs(), spectrum=spectrum)

    assert len(sess.stars) == 2
    assert all(np.array_equal(sess.inputs[name]["f"], spectrum[0]) for name in ["a", "b"])
    assert all(np.array_equal(sess.inputs[name]["s"], spectrum[1]) for name in ["a", "b"])


def test_session_rejects_mismatched_spectrum_targets():
    spectrum = {"a": np.array([[1.0], [2.0]])}

    with pytest.raises(ValueError, match="targets in spectrum"):
        core.session(["a", "b"], valid_obs(), spectrum=spectrum)


def test_session_call_supports_shared_kwargs():
    calls = []

    class DummyStar:
        def __init__(self, name):
            self.name = name

        def __call__(self, mode_kwargs, peak_kwargs):
            calls.append((self.name, mode_kwargs, peak_kwargs))

    sess = core.session.__new__(core.session)
    sess.inputs = {"a": {}, "b": {}}
    sess.stars = [DummyStar("a"), DummyStar("b")]

    sess(modeID_kwargs={"model": "sg"}, peakbag_kwargs={"slice": True})

    assert calls == [
        ("a", {"model": "sg"}, {"slice": True}),
        ("b", {"model": "sg"}, {"slice": True}),
    ]


def test_session_call_supports_target_specific_kwargs():
    calls = []

    class DummyStar:
        def __init__(self, name):
            self.name = name

        def __call__(self, mode_kwargs, peak_kwargs):
            calls.append((self.name, mode_kwargs, peak_kwargs))

    sess = core.session.__new__(core.session)
    sess.inputs = {"a": {}, "b": {}}
    sess.stars = [DummyStar("a"), DummyStar("b")]
    mode_kwargs = {"a": {"model": "ms"}, "b": {"model": "rgb"}}
    peak_kwargs = {"a": {"slice": False}, "b": {"slice": True}}

    sess(modeID_kwargs=mode_kwargs, peakbag_kwargs=peak_kwargs)

    assert calls == [
        ("a", mode_kwargs["a"], peak_kwargs["a"]),
        ("b", mode_kwargs["b"], peak_kwargs["b"]),
    ]


def test_star_run_mode_id_forwards_attributes_and_default_prior(monkeypatch, tmp_path):
    calls = []

    class DummyModeID:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.result = {"summary": {}}
            calls.append(("init", kwargs))

        def __call__(self, **kwargs):
            calls.append(("call", kwargs))

    monkeypatch.setattr(core, "modeID", DummyModeID)
    monkeypatch.setattr(core.IO, "_getPriorPath", lambda: "/tmp/prior.csv")
    monkeypatch.setattr(core.IO, "_setOutpath", lambda name, outpath: str(tmp_path))

    star = core.star(
        "target",
        f=np.array([1.0, 2.0]),
        s=np.array([3.0, 4.0]),
        obs=valid_obs(),
    )
    star.runModeID({"model": "sg"})

    assert star.priorpath == "/tmp/prior.csv"
    assert calls[0][1]["name"] == "target"
    assert calls[0][1]["model"] == "sg"
    assert calls[0][1]["priorpath"] == "/tmp/prior.csv"
    assert calls[1][1] == calls[0][1]


def test_star_run_peakbag_uses_mode_id_results_and_rv(monkeypatch, tmp_path):
    calls = []

    class DummyPeakbag:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.result = {"done": True}
            calls.append(("init", kwargs))

        def __call__(self, **kwargs):
            calls.append(("call", kwargs))

    monkeypatch.setattr(core, "peakbag", DummyPeakbag)
    monkeypatch.setattr(core.IO, "_setOutpath", lambda name, outpath: str(tmp_path))

    obs = valid_obs()
    obs["RV"] = (20.0, 1.0)
    star = core.star(
        "target",
        f=np.array([1.0, 2.0]),
        s=np.array([3.0, 4.0]),
        obs=obs,
    )
    star.modeID = type("Mode", (), {
        "result": {
            "ell": np.array([0, 1]),
            "summary": {
                "freq": np.array([[100.0, 105.0], [0.1, 0.1]]),
                "height": np.ones((2, 2)),
                "width": np.ones((2, 2)) * 0.2,
            },
        }
    })()

    star.runPeakbag({"slice": True})

    kwargs = calls[0][1]
    assert np.array_equal(kwargs["ell"], np.array([0, 1]))
    assert np.array_equal(kwargs["freq"], star.modeID.result["summary"]["freq"])
    assert kwargs["RV"] == (20.0, 1.0)
    assert kwargs["slice"] is True
    assert calls[1][1] == kwargs


def test_star_call_runs_both_stages_and_returns_results():
    star = core.star.__new__(core.star)
    calls = []

    def run_mode(kwargs):
        calls.append(("mode", kwargs))
        star.modeID = type("Mode", (), {"result": {"mode": 1}})()

    def run_peak(kwargs):
        calls.append(("peak", kwargs))
        star.peakbag = type("Peak", (), {"result": {"peak": 2}})()

    star.runModeID = run_mode
    star.runPeakbag = run_peak

    result = star({"model": "sg"}, {"slice": True})

    assert calls == [
        ("mode", {"model": "sg"}),
        ("peak", {"slice": True}),
    ]
    assert result == ({"mode": 1}, {"peak": 2})
