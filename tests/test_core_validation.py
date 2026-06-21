"""Tests for high-level session input validation."""

import numpy as np
import pytest

from pbjam import core


def valid_obs():
    return {
        'numax': (1000.0, 50.0),
        'dnu': (50.0, 1.0),
        'teff': (5777.0, 100.0),
    }


def test_session_accepts_array_like_spectrum(tmp_path):
    spectrum = [
        np.linspace(100.0, 200.0, 10),
        np.ones(10),
    ]

    sess = core.session('target', valid_obs(), spectrum=spectrum, outpath=tmp_path)

    assert len(sess.stars) == 1
    assert sess.stars[0].name == 'target'


def test_session_requires_obs_dictionary(tmp_path):
    spectrum = np.ones((2, 10))

    with pytest.raises(TypeError, match='obs argument must be a dictionary'):
        core.session('target', None, spectrum=spectrum, outpath=tmp_path)


def test_session_reports_missing_required_obs_key(tmp_path):
    obs = valid_obs()
    obs.pop('teff')
    spectrum = np.ones((2, 10))

    with pytest.raises(ValueError, match='Missing teff'):
        core.session('target', obs, spectrum=spectrum, outpath=tmp_path)


def test_session_rejects_non_positive_obs_uncertainty(tmp_path):
    obs = valid_obs()
    obs['dnu'] = (50.0, 0.0)
    spectrum = np.ones((2, 10))

    with pytest.raises(ValueError, match='must be positive'):
        core.session('target', obs, spectrum=spectrum, outpath=tmp_path)


def test_session_requires_complete_per_target_obs(tmp_path):
    obs = {'a': valid_obs()}
    spectrum = {'a': np.ones((2, 10)), 'b': np.ones((2, 10))}

    with pytest.raises(ValueError, match='targets in obs must match'):
        core.session(['a', 'b'], obs, spectrum=spectrum, outpath=tmp_path)


def test_session_reports_spectrum_target_mismatch(tmp_path):
    spectrum = {'a': np.ones((2, 10)), 'extra': np.ones((2, 10))}

    with pytest.raises(ValueError, match='targets in spectrum must match'):
        core.session(['a', 'b'], valid_obs(), spectrum=spectrum, outpath=tmp_path)


def test_session_reports_bad_spectrum_shape(tmp_path):
    spectrum = np.ones((3, 10))

    with pytest.raises(ValueError, match='spectrum must have shape'):
        core.session('target', valid_obs(), spectrum=spectrum, outpath=tmp_path)


def test_session_requires_download_kwargs_without_local_data(tmp_path):
    with pytest.raises(ValueError, match='To download data lk_kwargs must include'):
        core.session('target', valid_obs(), outpath=tmp_path)


def test_session_uses_array_timeseries_for_each_target(monkeypatch, tmp_path):
    calls = []

    class FakePsd:
        def __init__(self, ID, **kwargs):
            calls.append((ID, kwargs))
            self.freq = np.linspace(100.0, 200.0, 10)
            self.powerdensity = np.ones(10)

        def __call__(self):
            return None

    monkeypatch.setattr(core.IO, 'psd', FakePsd)

    timeseries = np.vstack([
        np.linspace(0.0, 1.0, 10),
        np.ones(10),
    ])

    sess = core.session(['a', 'b'], valid_obs(), timeseries=timeseries, outpath=tmp_path)

    assert [star.name for star in sess.stars] == ['a', 'b']
    assert [call[0] for call in calls] == ['a', 'b']
