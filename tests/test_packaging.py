"""Repository-level packaging and metadata tests."""

from importlib.metadata import version
from pathlib import Path
import os
import subprocess
import sys
import tomllib

import pbjam
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_pyproject():
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_pyproject_is_valid_toml():
    metadata = load_pyproject()

    assert metadata["project"]["name"] == "pbjam"


def test_project_metadata_matches_runtime_package():
    project = load_pyproject()["project"]

    assert project["version"] == pbjam.__version__
    assert project["version"] == version("pbjam")
    assert project["requires-python"].startswith(">=")
    assert project["readme"] == "README.rst"


def test_project_urls_are_valid_strings():
    urls = load_pyproject()["project"]["urls"]

    assert set(urls) >= {"Homepage", "Issues", "Documentation", "Repository"}
    assert all(isinstance(value, str) for value in urls.values())
    assert urls["Issues"].endswith("/PBjam/issues")
    assert urls["Repository"].endswith("/PBjam")


def test_required_repository_files_exist():
    for relative_path in [
        "README.rst",
        "LICENSE",
        "pbjam/data/parameters.json",
        "pbjam/data/prior_data.csv",
    ]:
        assert (REPO_ROOT / relative_path).is_file(), relative_path


def test_packaged_data_files_are_declared():
    package_data = load_pyproject()["tool"]["setuptools"]["package-data"]

    assert "pbjam" in package_data
    assert "data/*.csv" in package_data["pbjam"]
    assert "data/*.json" in package_data["pbjam"]


def test_statsmodels_dependency_has_been_removed():
    dependencies = load_pyproject()["project"]["dependencies"]

    assert not any(
        dependency.lower().startswith("statsmodels")
        for dependency in dependencies
    )

    source_files = list((REPO_ROOT / "pbjam").glob("*.py"))
    assert not any(
        "import statsmodels" in source_file.read_text().lower()
        for source_file in source_files
    )


@pytest.mark.packaging
def test_wheel_builds_without_resolving_dependencies(tmp_path):
    command = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        str(tmp_path),
        str(REPO_ROOT),
    ]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = list(tmp_path.glob("pbjam-*.whl"))
    assert len(wheels) == 1


@pytest.mark.integration
@pytest.mark.packaging
def test_editable_install_in_clean_environment(tmp_path):
    if os.environ.get("PBJAM_RUN_INSTALL_TESTS") != "1":
        pytest.skip("Set PBJAM_RUN_INSTALL_TESTS=1 to run the editable-install test.")

    environment = tmp_path / "venv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "venv",
            "--system-site-packages",
            str(environment),
        ],
        check=True,
    )

    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    completed = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "-e",
            str(REPO_ROOT),
        ],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    smoke = subprocess.run(
        [
            str(python),
            "-c",
            "import pbjam; print(pbjam.__version__)",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert smoke.returncode == 0, smoke.stdout + smoke.stderr
