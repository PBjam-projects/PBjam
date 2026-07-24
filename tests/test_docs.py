"""Repository-level documentation tests."""

from pathlib import Path
import importlib.util
import runpy
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs" / "source"

def test_notebook_parser_is_registered_by_nbsphinx():
    namespace = runpy.run_path(str(DOCS_ROOT / "conf.py"))

    assert "nbsphinx" in namespace["extensions"]

    source_suffix = namespace.get("source_suffix", {})
    assert source_suffix.get(".ipynb") != "nbsphinx"

def iter_toctree_targets(path):
    lines = path.read_text().splitlines()
    inside = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith(".. toctree::"):
            inside = True
            continue

        if not inside:
            continue

        if not stripped:
            continue

        if line == line.lstrip():
            inside = False
            continue

        if stripped.startswith(":"):
            continue

        yield stripped


def resolve_doc_target(source, target):
    target = target.split("<")[-1].rstrip(">").strip()
    if "://" in target:
        return None

    candidate = source.parent / target
    if candidate.suffix:
        return candidate

    for suffix in [".rst", ".ipynb"]:
        with_suffix = candidate.with_suffix(suffix)
        if with_suffix.exists():
            return with_suffix

    return candidate.with_suffix(".rst")


def test_sphinx_configuration_executes():
    namespace = runpy.run_path(str(DOCS_ROOT / "conf.py"))

    assert namespace["project"] == "PBjam"
    assert namespace["root_doc"] == "index"
    assert "sphinx.ext.autodoc" in namespace["extensions"]
    assert "sphinx.ext.napoleon" in namespace["extensions"]


def test_all_toctree_targets_exist():
    missing = []

    for source in DOCS_ROOT.rglob("*.rst"):
        for target in iter_toctree_targets(source):
            resolved = resolve_doc_target(source, target)
            if resolved is not None and not resolved.exists():
                missing.append(f"{source.relative_to(REPO_ROOT)} -> {target}")

    assert not missing, "Missing toctree targets:\n" + "\n".join(missing)


def test_api_pages_cover_package_modules():
    expected = {
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
    }
    documented = {
        path.stem
        for path in DOCS_ROOT.glob("*.rst")
        if ".. automodule:: pbjam." in path.read_text()
    }

    assert expected <= documented


def test_readthedocs_configuration_targets_sphinx_docs():
    yaml = pytest.importorskip("yaml")
    config_path = REPO_ROOT / ".readthedocs.yaml"

    assert config_path.is_file()
    config = yaml.safe_load(config_path.read_text())

    assert config["version"] == 2
    assert config["sphinx"]["configuration"] == "docs/source/conf.py"

    installs = config["python"]["install"]
    assert any(
        entry.get("path") == "."
        and "docs" in entry.get("extra_requirements", [])
        for entry in installs
    )


@pytest.mark.docs
def test_sphinx_html_build_has_no_warnings(tmp_path):
    pytest.importorskip("sphinx")
    pytest.importorskip("nbsphinx")
    pytest.importorskip("sphinx_rtd_theme")

    output = tmp_path / "html"
    doctrees = tmp_path / "doctrees"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            "html",
            "-d",
            str(doctrees),
            str(DOCS_ROOT),
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (output / "index.html").is_file()
