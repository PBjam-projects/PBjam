"""Tests for the jar module"""

from pbjam import jar
import numpy as np
import numpy.testing as npt #import assert_almost_equal, assert_array_equal
import jax.numpy as jnp
import pytest

def test_visibility():
    for inc in range(0, 90, 10):
        inc = inc / 180 * np.pi
        for l in np.arange(0, 4):
            for m in np.arange(2*l+1) - l:

                if l==0: 
                    if m==0:
                        assert np.isclose(jar.visibility(l, m, inc), 1)
                    else:
                        assert np.isnan(jar.visibility(l, m, inc))
                        
                elif l==1: 
                    if m==0:
                        assert np.isclose(jar.visibility(l, m, inc), np.cos(inc)**2)
                    elif m==1:
                        assert np.isclose(jar.visibility(l, m, inc), 0.5 * np.sin(inc)**2)
                    elif m==-1:
                        assert np.isclose(jar.visibility(l, m, inc), 0.5 * np.sin(inc)**2)
                    else: 
                        assert np.isnan(jar.visibility(l, m, inc))
                        
                elif l==2:
                    if m==0:
                        assert np.isclose(jar.visibility(l, m, inc), 1/4 * (3*np.cos(inc)**2 - 1)**2)
                    elif m==1:
                        assert np.isclose(jar.visibility(l, m, inc), 3/8 * np.sin(2*inc)**2)
                    elif m==-1:
                        assert np.isclose(jar.visibility(l, m, inc), 3/8 * np.sin(2*inc)**2)
                    elif m==2:
                        assert np.isclose(jar.visibility(l, m, inc), 3/8 * np.sin(inc)**4)
                    elif m==-2:
                        assert np.isclose(jar.visibility(l, m, inc), 3/8 * np.sin(inc)**4)
                    else:
                        assert np.isnan(jar.visibility(l, m, inc))
                        
                elif l==3:
                    if m==0:
                        assert np.isclose(jar.visibility(l, m, inc), 1/64 * (5 * np.cos(3 * inc) + 3 * np.cos(inc))**2)
                    elif m==1:
                        assert np.isclose(jar.visibility(l, m, inc), 3/64 * (5 * np.cos(2 * inc) + 3)**2 * np.sin(inc)**2)
                    elif m==-1:
                        assert np.isclose(jar.visibility(l, m, inc), 3/64 * (5 * np.cos(2 * inc) + 3)**2 * np.sin(inc)**2)
                    elif m==2:
                        assert np.isclose(jar.visibility(l, m, inc), 15/8 * np.cos(inc)**2 * np.sin(inc)**4)
                    elif m==-2:
                        assert np.isclose(jar.visibility(l, m, inc), 15/8 * np.cos(inc)**2 * np.sin(inc)**4)
                    elif m==3:
                        assert np.isclose(jar.visibility(l, m, inc), 5/16 * np.sin(inc)**6)
                    elif m==-3:
                        assert np.isclose(jar.visibility(l, m, inc), 5/16 * np.sin(inc)**6)
                    else:
                        assert np.isnan(jar.visibility(l, m, inc))
                        
                else:
                    assert np.isnan(jar.visibility(l, m, inc))

@pytest.fixture
def generalModelFuncsClass():
    """Create an instance of the generalModelFuncs class class for testing.
    
    This class in only ever inherited so should not take any args or kwargs.
    
    """

    return jar.generalModelFuncs()

def test_modeUpdoot():
    sample = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    result = {
        'summary': {'freq': np.empty((2, 0))},
        'samples': {'freq': np.empty((sample.shape[0], 0))},
    }

    jar.modeUpdoot(result, sample, 'freq', Nmodes=2)

    assert result['summary']['freq'].shape == (2, 2)
    assert result['samples']['freq'].shape == sample.shape
    assert np.allclose(result['summary']['freq'][0], [2.0, 5.0])
    assert np.allclose(result['samples']['freq'], sample)

def test_smryStats():
    stats = jar.smryStats(np.array([1.0, 2.0, 3.0]))

    assert stats.shape == (2,)
    assert stats[0] == 2.0
    assert stats[1] > 0

def test_envelope():
    nu = jnp.array([90.0, 100.0, 110.0])
    env = jar.envelope(nu, env_height=3.0, numax=100.0, env_width=10.0)

    assert type(env) is type(nu)
    assert env[1] == 6.0
    assert env[0] == env[2]
    assert env[0] < env[1]

def test_attenuation():
    """Some tests for attenuation function in jar"""

    f = jnp.linspace(0, 1000, 100)

    eta = jar.attenuation(f, f[-1])

    assert all(np.isreal(eta))

    assert len(eta) == len(f)

    assert type(eta) is type(f)

    assert jar.attenuation(0, 1) == 1.0

def test_lor():
    """Some tests for lorentzian function in jar"""

    f = jnp.linspace(0, 100, 1000)

    L = jar.lor(f, 50, 1, 1)

    assert all(L > 0)

    L = jar.lor(f, 50, -1, 1)

    assert all(L < 0)

    assert len(L) == len(f)

    assert type(L) is type(f)

    assert jar.lor(50, 50, 1, 1) == 1.0

def test_getCurvePercentiles():
    """Some tests for getCurvePercentiles

    Should return an array of values corresponding to points along x

    """

    x = np.linspace(0, 1, 100)

    y = np.linspace(0, 1, 100)

    p = jar.getCurvePercentiles(x, y)

    assert all(np.isreal(p))

    assert all(p > 0)

    assert all(p < 1)

    assert np.allclose(
        p, [0.15151515, 0.4040404, 0.71717172, 0.91919192, 0.98989899]
    )

    assert np.allclose(
        jar.getCurvePercentiles(x, y, percentiles=[0.4]),
        np.array([0.63636364]),
    )

def test_jaxInterp1D():
    """Tests for the jaxinterp1d wrapper

    It's a callable wrapper function for jax.numpy.interp.

    """
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)

    j = jar.jaxInterp1D(x, y)

    assert j(0.5) == 0.5

    assert type(j(0.5)) is type(jnp.array([0.5]))

def test_to_log10():

    assert len(jar.to_log10(1, 1)) == 2

    assert np.allclose(jar.to_log10(1, 1), [0.0, 1 / np.log(10)])

    assert jar.to_log10(1, -1) == [1, -1]

def test_normal():

    x = jnp.linspace(-100, 100, 1000)
    mu = 0
    sigma = 10

    assert type(jar.normal(x, mu, sigma)) == type(x)

    assert len(jar.normal(x, mu, sigma)) == len(x)

    assert all(jar.normal(x, mu, sigma) > 0)

    assert jar.normal(mu, mu, sigma) == 1 / jnp.sqrt(2 * jnp.pi * sigma**2)

def test_gaussian():

    x = jnp.linspace(-100, 100, 1000)
    mu = 0
    sigma = 10
    A = 5

    assert type(jar.gaussian(x, A, mu, sigma)) == type(x)

    assert len(jar.gaussian(x, A, mu, sigma)) == len(x)

    assert all(jar.gaussian(x, A, mu, sigma) > 0)

    assert all(jar.gaussian(x, -A, mu, sigma) < 0)

    assert jar.gaussian(mu, A, mu, sigma) == A


class DummyObsLikelihoodModel(jar.generalModelFuncs):
    def __init__(self):
        self.obs = {
            'numax': (100.0, 10.0),
            'dnu': (10.0, 1.0),
            'teff': (5800.0, 100.0),
            'bp_rp': (0.8, 0.1),
        }
        self.s = jnp.array([0.0])
        self.setAddObs(keys=['numax', 'dnu', 'teff', 'bp_rp', 'missing'])

    def unpackParams(self, theta):
        return {
            'numax': theta[0],
            'dnu': theta[1],
            'teff': theta[2],
            'bp_rp': theta[3],
        }

    def model(self, thetaU):
        return jnp.array([1.0])


def test_additional_observables_are_used_in_full_likelihood():
    model = DummyObsLikelihoodModel()
    theta = jnp.array([100.0, 10.0, 5800.0, 0.8])

    expected = sum(obs.logpdf(theta[i]) for i, obs in enumerate(model.addObs.values()))

    assert set(model.addObs) == {'numax', 'dnu', 'teff', 'bp_rp'}
    assert np.isclose(model.lnlikelihood(theta), expected)


def test_setAddObs_skips_missing_and_invalid_constraints():
    model = jar.generalModelFuncs()
    model.obs = {
        'teff': (5800.0, 100.0),
        'bp_rp': (0.8, 0.0),
    }

    model.setAddObs(keys=['numax', 'teff', 'bp_rp'])

    assert set(model.addObs) == {'teff'}
