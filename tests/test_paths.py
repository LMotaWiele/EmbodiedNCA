"""Smoke for embodiednca.paths — results root is repo-relative."""

from embodiednca.paths import REPO_ROOT, RESULTS_ROOT, docs_design, results_dir


def test_results_dir_is_under_repo_results():
    assert results_dir("v0") == RESULTS_ROOT / "v0"
    assert RESULTS_ROOT == REPO_ROOT / "results"
    assert (REPO_ROOT / "pyproject.toml").is_file()


def test_docs_design_helper():
    assert docs_design("v0.md") == REPO_ROOT / "docs" / "design" / "v0.md"
