"""Tests for time-series and power-spectrum input/output helpers."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pbjam import IO


def test_time_series_filters_invalid_and_user_masked_samples():
    time = np.arange(6.0)
    flux = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    flux_err = np.ones(6)
    bad_idx = np.array([False, False, False, True, False, False])

    ts = IO.timeSeries(
        "target",
        lk_kwargs={},
        time=time,
        flux=flux,
        flux_err=flux_err,
        badIdx=bad_idx,
    )

    assert np.array_equal(ts.time, np.array([0.0, 1.0, 4.0, 5.0]))
    assert np.array_equal(ts.flux, np.array([1.0, 2.0, 5.0, 6.0]))
    assert np.array_equal(ts.flux_err, np.ones(4))
    assert ts.NT == 4
    assert ts.dT == pytest.approx(5.0)
    assert ts.dt == pytest.approx(1.0)
    assert ts.dutyCycle == pytest.approx(0.8)


def test_get_bad_index_accepts_missing_flux_errors():
    ts = IO.timeSeries.__new__(IO.timeSeries)

    ts._getBadIndex(
        time=np.array([0.0, 1.0, np.nan]),
        flux=np.array([1.0, np.inf, 3.0]),
        flux_err=None,
        badIdx=None,
    )

    assert np.array_equal(ts.badIdx, np.array([False, True, True]))


def test_get_duty_cycle_uses_supplied_cadence():
    ts = IO.timeSeries.__new__(IO.timeSeries)
    ts.time = np.array([0.0, 1.0, 3.0, 4.0])

    assert ts._getDutyCycle(cadence=1.0) == pytest.approx(1.0)


def test_get_time_series_window_function_fills_internal_gaps():
    spectrum = IO.psd.__new__(IO.psd)
    spectrum.TS = SimpleNamespace(
        time=np.array([0.0, 1.0, 3.0]),
        dt=1.0,
    )

    time, window = spectrum.getTSWindowFunction()

    assert np.array_equal(time, np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.array_equal(window, np.array([1.0, 1.0, 0.0, 1.0]))


def test_get_norm_matches_parseval_for_unweighted_data():
    spectrum = IO.psd.__new__(IO.psd)
    spectrum.ls = SimpleNamespace(
        t=np.arange(4.0),
        y=np.array([1.0, 2.0, 3.0, 4.0]),
        dy=None,
    )

    spectrum._getNorm(np.array([1.0, 1.0]))

    expected_mean_square = np.mean(
        (spectrum.ls.y - spectrum.ls.y.mean()) ** 2
    )
    assert spectrum.normfactor == pytest.approx(expected_mean_square / 2.0)


def test_get_norm_handles_weighted_data():
    spectrum = IO.psd.__new__(IO.psd)
    spectrum.ls = SimpleNamespace(
        t=np.arange(3.0),
        y=np.array([1.0, 2.0, 4.0]),
        dy=np.array([1.0, 2.0, 1.0]),
    )

    spectrum._getNorm(np.array([2.0, 1.0]))

    centered = spectrum.ls.y - spectrum.ls.y.mean()
    expected = np.sum((centered / spectrum.ls.dy) ** 2)
    expected /= np.sum((1.0 / spectrum.ls.dy) ** 2)
    assert spectrum.normfactor == pytest.approx(expected / 3.0)


def test_get_ts_uses_lightkurve_download_pipeline(monkeypatch):
    calls = {}

    class Values:
        def __init__(self, value):
            self.value = np.asarray(value)

    class DummyLightCurve:
        time = Values([0.0, 1.0, 2.0])
        flux = Values([1.0, 0.9, 1.1])
        flux_err = Values([0.1, 0.1, 0.2])

    class DummyCollection:
        def stitch(self):
            calls["stitched"] = True
            return DummyLightCurve()

    class DummySearch:
        def download_all(self, download_dir=None):
            calls["download_dir"] = download_dir
            return DummyCollection()

    def search_lightcurve(target, **kwargs):
        calls["target"] = target
        calls["kwargs"] = kwargs
        return DummySearch()

    ts = IO.timeSeries.__new__(IO.timeSeries)
    ts.ID = "KIC 123"
    ts.lk_kwargs = {"mission": "Kepler"}
    ts.downloadDir = "/tmp/lightkurve"
    ts.cleanLC = lambda lc, threshold: lc

    monkeypatch.setattr(IO.time, "sleep", lambda _: None)
    monkeypatch.setattr(IO.np.random, "uniform", lambda low, high: 0.0)
    monkeypatch.setattr(IO.lk, "search_lightcurve", search_lightcurve)

    time, flux, flux_err = ts._getTS(outlierRejection=4)

    assert calls == {
        "target": "KIC 123",
        "kwargs": {"mission": "Kepler"},
        "download_dir": "/tmp/lightkurve",
        "stitched": True,
    }
    assert np.allclose(time, [0.0, 1.0, 2.0])
    assert np.allclose(flux, [1.0, 0.9, 1.1])
    assert np.allclose(flux_err, [0.1, 0.1, 0.2])


def test_set_outpath_creates_target_directory(tmp_path):
    path = IO._setOutpath("KIC123", str(tmp_path))

    assert Path(path) == tmp_path / "KIC123"
    assert Path(path).is_dir()


def test_get_outpath_appends_filename_and_rejects_invalid_input(tmp_path):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    obj = SimpleNamespace(path=str(target_dir))

    assert IO._getOutpath(obj, None) == str(target_dir)
    assert IO._getOutpath(obj, "result.csv") == str(target_dir / "result.csv")

    with pytest.raises(ValueError, match="Unrecognized input"):
        IO._getOutpath(obj, 42)


def test_default_prior_path_points_to_packaged_csv():
    assert IO._getPriorPath().endswith("pbjam/data/prior_data.csv")
