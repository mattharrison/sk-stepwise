from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ENV = {
    "UV_CACHE_DIR": str(ROOT / ".uv-cache"),
    "UV_PROJECT_ENVIRONMENT": str(ROOT / ".tmp-venv"),
}


def _run_uv(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", *args],
        cwd=ROOT,
        env={**os.environ, **ENV},
        capture_output=True,
        text=True,
    )


def test_ruff_check_passes() -> None:
    result = _run_uv("ruff", "check", ".")
    assert result.returncode == 0, result.stdout + result.stderr


def test_mypy_passes() -> None:
    result = _run_uv("mypy", "src")
    assert result.returncode == 0, result.stdout + result.stderr
