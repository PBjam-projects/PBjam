from importlib.metadata import version

import pbjam


def test_runtime_version_matches_package_metadata():
    assert pbjam.__version__ == version("pbjam")
