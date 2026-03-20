from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_uses_supported_metadata_and_dependency_groups():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())

    project = data["project"]
    assert project["description"] != "Add your description here"
    assert project["requires-python"] == ">=3.11,<3.15"
    assert "dependency-groups" in data
    assert (
        "tool" not in data
        or "uv" not in data["tool"]
        or "dev-dependencies" not in data["tool"]["uv"]
    )

    runtime_deps = set(project["dependencies"])
    assert not any(dep.startswith("disutils") for dep in runtime_deps)
    assert not any(dep.startswith("setuptools") for dep in runtime_deps)
    assert not any(dep.startswith("hyperopt") for dep in runtime_deps)


def test_gitignore_covers_generated_env_and_build_artifacts():
    gitignore = (ROOT / ".gitignore").read_text()

    for expected in [
        ".venv/",
        ".tmp-venv/",
        ".uv-cache/",
        ".pytest_cache/",
        ".ruff_cache/",
        "dist/",
        "*.egg-info/",
    ]:
        assert expected in gitignore


def test_generated_egg_info_is_not_present_in_src_tree():
    assert not (ROOT / "src" / "sk_stepwise.egg-info").exists()


def test_readme_documents_uv_contributor_workflow():
    readme = (ROOT / "README.md").read_text()

    assert "uv sync" in readme
    assert "uv run pytest" in readme
    assert "uv run pytest -q tests/test_readme_doctest.py" in readme


def test_github_actions_workflow_covers_supported_python_versions() -> None:
    workflow = (ROOT / ".github" / "workflows" / "basic.yml").read_text()

    for version in ['"3.11"', '"3.12"', '"3.13"', '"3.14"']:
        assert version in workflow

    assert "uv sync" in workflow
    assert "uv run pytest -q" in workflow


def test_mainline_docs_and_source_do_not_depend_on_hyperopt() -> None:
    readme = (ROOT / "README.md").read_text()
    source_files = [
        path.read_text() for path in (ROOT / "src" / "sk_stepwise").glob("*.py")
    ]

    assert "import hyperopt" not in readme
    assert "hp.choice" not in readme
    assert "hp.uniform" not in readme
    assert "hp.quniform" not in readme

    combined_source = "\n".join(source_files)
    assert "import hyperopt" not in combined_source
    assert "from hyperopt" not in combined_source
