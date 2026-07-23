"""Tests for deterministic peakbagging helpers."""

import numpy as np
import pytest

from pbjam.peakbagging import basePeakbag, jointRotInc, peakbag


def bare_peakbag():
    obj = peakbag.__new__(peakbag)
    obj.ell = np.array([2, 0, 1, 2, 0, 1, 2, 0])
    obj.freq = np.array([
        [98.0, 100.0, 105.0, 108.0, 110.0, 115.0, 118.0, 120.0],
        [0.2, 0.2, 0.3, 0.2, 0.2, 0.3, 0.2, 0.2],
    ])
    obj.height = np.ones_like(obj.freq)
    obj.width = np.full_like(obj.freq, 0.1)
    obj.zeta = np.linspace(0.0, 0.7, obj.freq.shape[1])
    obj.rotAsym = np.zeros_like(obj.freq)
    obj.RV = None
    obj.dnu = None
    obj.d02 = None
    return obj


def test_set_dnu_prefers_radial_modes():
    obj = bare_peakbag()

    dnu = obj._setDnu()

    assert dnu[0] == pytest.approx(10.0)
    assert np.isnan(dnu[1])


def test_set_d02_estimates_radial_quadrupole_separation():
    obj = bare_peakbag()
    obj.dnu = np.array([10.0, np.nan])

    d02 = obj._setd02()

    assert d02[0] == pytest.approx(2.0)
    assert np.isnan(d02[1])


@pytest.mark.parametrize(
    ("attribute", "value", "expected"),
    [
        ("dnu", 10.0, 10.0),
        ("d02", 1.5, 1.5),
    ],
)
def test_scalar_separations_are_converted_to_value_error_pairs(
    attribute,
    value,
    expected,
):
    obj = bare_peakbag()
    setattr(obj, attribute, value)

    result = obj._setDnu() if attribute == "dnu" else obj._setd02()

    assert result[0] == expected
    assert np.isnan(result[1])


def test_check_defaults_populates_optional_mode_properties():
    obj = bare_peakbag()
    obj.height = None
    obj.width = None
    obj.zeta = None
    obj.rotAsym = None

    obj._checkDefaults()

    assert np.all(obj.height == 1.0)
    assert np.all(obj.width == 0.1)
    assert np.all(obj.zeta == 0.0)
    assert np.all(obj.rotAsym == 0.0)
    assert obj.RV == (0, 0)


def test_pick_modes_trims_all_mode_arrays_consistently():
    obj = bare_peakbag()
    obj.freqLimits = [(99.0, 111.0)]

    selected = obj._pickModes()

    assert np.array_equal(
        selected,
        np.array([False, True, True, True, True, False, False, False]),
    )
    assert np.array_equal(obj.ell, np.array([0, 1, 2, 0]))
    assert np.allclose(obj.freq[0], [100.0, 105.0, 108.0, 110.0])
    assert obj.height.shape == obj.freq.shape
    assert obj.width.shape == obj.freq.shape
    assert obj.rotAsym.shape == obj.freq.shape
    assert obj.zeta.shape == (4,)


def test_build_spectrum_mask_uses_explicit_intervals():
    obj = bare_peakbag()
    obj.f = np.arange(90.0, 131.0)
    obj.snr = np.ones_like(obj.f)
    obj.freqLimits = [(95.0, 100.0), (115.0, 120.0)]
    obj.dnu = np.array([10.0, np.nan])
    obj.d02 = np.array([2.0, np.nan])
    obj.dipoleMask = False

    mask = obj._buildSpectrumMask()

    expected = ((95.0 <= obj.f) & (obj.f <= 100.0))
    expected |= ((115.0 <= obj.f) & (obj.f <= 120.0))
    assert np.array_equal(mask, expected)


def test_small_difference_check_rejects_cuts_near_modes():
    obj = bare_peakbag()

    result = obj._checkSmallDiffs(
        cuts=np.array([90.0, 99.5, 105.0, 130.0]),
        nu=np.array([100.0, 110.0]),
        Gamma=1.0,
    )

    assert np.array_equal(result, np.array([True, False, True, True]))


def test_small_separation_check_preserves_mode_pairs():
    obj = bare_peakbag()

    result = obj._checkNoSmallSep(
        C=np.array([95.0, 99.0, 105.0, 115.0]),
        nu=np.array([98.0, 100.0, 108.0, 110.0]),
        ells=np.array([2, 0, 2, 0]),
    )

    assert np.array_equal(result, np.array([True, False, True, True]))


def test_kmeans_groups_well_separated_modes():
    obj = bare_peakbag()

    centroids, labels, count = obj._kmeans(
        nu=np.array([0.0, 1.0, 9.0, 10.0]),
        ells=np.array([0, 1, 0, 1]),
        centroids=[0.5, 9.5],
    )

    assert count == 2
    assert np.allclose(centroids, [0.5, 9.5])
    assert np.array_equal(labels, np.array([0, 0, 1, 1]))


def test_determine_cuts_places_boundaries_between_clusters():
    obj = bare_peakbag()

    cuts = obj._determineCuts(
        dnu=12.0,
        nu=np.array([100.0, 102.0, 110.0, 112.0]),
        labels=np.array([0, 0, 1, 1]),
    )

    assert np.allclose(cuts, [96.0, 106.0, 116.0])


def test_slice_helper_includes_both_bounds():
    obj = bare_peakbag()

    assert np.array_equal(
        obj._slc(np.array([0.0, 1.0, 2.0, 3.0]), 1.0, 2.0),
        np.array([False, True, True, False]),
    )


def test_slice_spectrum_accepts_manual_intervals():
    obj = bare_peakbag()
    obj.freqLimits = [(95.0, 105.0), (105.0, 115.0)]

    assert obj.sliceSpectrum() == obj.freqLimits


def test_slice_spectrum_rejects_reversed_manual_interval():
    obj = bare_peakbag()
    obj.freqLimits = [(105.0, 95.0), (105.0, 115.0)]

    with pytest.raises(AssertionError, match="freqLimits"):
        obj.sliceSpectrum()


def test_base_peakbag_frequency_range_is_open_interval():
    obj = basePeakbag.__new__(basePeakbag)
    obj.f = np.array([90.0, 100.0, 110.0, 120.0])
    obj.freqLimits = [100.0, 120.0]

    assert np.array_equal(
        obj.setFreqRange(),
        np.array([False, False, True, False]),
    )


def test_base_peakbag_chi_square_rejects_invalid_models():
    obj = basePeakbag.__new__(basePeakbag)
    obj.s = np.array([2.0, 4.0])
    obj.sel = np.array([True, True])

    assert np.isfinite(obj.chi_sqr(np.array([1.0, 2.0])))

    with np.errstate(divide="ignore", invalid="ignore"):
        assert np.isneginf(obj.chi_sqr(np.array([1.0, 0.0])))
        assert np.isneginf(obj.chi_sqr(np.array([1.0, np.nan])))


def test_doppler_correction_is_unity_without_rv_uncertainty():
    obj = basePeakbag.__new__(basePeakbag)
    obj.RV = (25.0, 0.0)

    assert np.array_equal(obj.dopplerRVCorrection(4), np.ones(4))


def test_joint_rotation_kde_uses_scipy_density_estimator():
    rng = np.random.default_rng(42)

    class DummyInstance:
        def __init__(self):
            self.samples = rng.normal(size=(100, 3))
            self.priors = {
                "nurot_e": object(),
                "nurot_c": object(),
                "inc": object(),
            }

        def unpackSamples(self, samples):
            return {
                "nurot_e": samples[:, 0],
                "nurot_c": samples[:, 1],
                "inc": samples[:, 2],
            }

    pb = type("Peakbag", (), {"pbInstances": [DummyInstance(), DummyInstance()]})()

    model = jointRotInc(pb, NKDE=50, bw=0.1)

    assert len(model.kdes) == 2
    assert all(np.isfinite(kde.pdf(np.zeros(3))).all() for kde in model.kdes)
