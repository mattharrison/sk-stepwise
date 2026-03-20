from __future__ import annotations

from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _run(
    *args: str, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd or ROOT,
        env={**os.environ, **(env or {})},
        text=True,
        capture_output=True,
    )


def test_built_wheel_installs_and_imports_in_clean_env(tmp_path: Path) -> None:
    cache_dir = ROOT / ".uv-cache"
    build_env = {
        "UV_CACHE_DIR": str(cache_dir),
        "UV_PROJECT_ENVIRONMENT": str(ROOT / ".tmp-venv"),
    }
    build = _run("uv", "build", cwd=ROOT, env=build_env)
    assert build.returncode == 0, build.stdout + build.stderr

    wheel_dir = ROOT / "dist"
    wheels = sorted(wheel_dir.glob("sk_stepwise-*.whl"))
    assert wheels, "No wheel produced by uv build"
    wheel = wheels[-1]

    smoke_env_dir = tmp_path / "release-smoke-venv"
    smoke_env = {
        "UV_CACHE_DIR": str(cache_dir),
        "UV_PROJECT_ENVIRONMENT": str(smoke_env_dir),
    }

    sync = _run("uv", "venv", str(smoke_env_dir), cwd=ROOT, env=smoke_env)
    assert sync.returncode == 0, sync.stdout + sync.stderr

    install = _run("uv", "pip", "install", str(wheel), cwd=ROOT, env=smoke_env)
    assert install.returncode == 0, install.stdout + install.stderr

    smoke = _run(
        "uv",
        "run",
        "python",
        "-c",
        "import sk_stepwise as sw; assert hasattr(sw, 'StepwiseOptunaSearchCV'); assert hasattr(sw, 'Int')",
        cwd=ROOT,
        env=smoke_env,
    )
    assert smoke.returncode == 0, smoke.stdout + smoke.stderr
