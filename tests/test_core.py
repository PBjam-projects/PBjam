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
