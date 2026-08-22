"""Repo-root results layout: `results/<stage>/…` and `docs/design/`."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
DOCS_DESIGN = REPO_ROOT / "docs" / "design"


def results_dir(stage: str) -> Path:
    """Return `results/<stage>/` (does not create it)."""
    return RESULTS_ROOT / stage


def docs_design(*parts: str) -> Path:
    """Return `docs/design/` or a file under it."""
    return DOCS_DESIGN.joinpath(*parts) if parts else DOCS_DESIGN
