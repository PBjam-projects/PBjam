"""Smoke tests for package imports and the public API."""

import importlib

import pytest

import pbjam


MODULES = [
    "background",
    "core",
    "distributions",
    "DR",
    "IO",
    "jar",
    "l1models",
    "l20models",
    "MSmodels",
    "modeID",
    "peakbagging",
    "plotting",
    "samplers",
    "validation",
    "version",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_package_modules_import(module_name):
    module = importlib.import_module(f"pbjam.{module_name}")

    assert module.__name__ == f"pbjam.{module_name}"


def test_public_api_exports_main_workflow_classes():
    assert callable(pbjam.session)
    assert callable(pbjam.star)
    assert callable(pbjam.modeID)
    assert callable(pbjam.basePeakbag)
    assert isinstance(pbjam.__version__, str)
    assert pbjam.__version__


def test_importing_pbjam_does_not_change_numpy_error_policy():
    import numpy as np

    before = np.geterr()
    importlib.reload(pbjam)
    after = np.geterr()

    assert after == before
