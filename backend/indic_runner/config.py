import os
from pathlib import Path

# Location of the optional .env file, searched upward from the working
# directory so the CLI works from anywhere inside the project.
ENV_FILENAME = ".env"


def _parse_env(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines, ignoring comments, blanks and `export` prefixes."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def find_env_file(start: Path | None = None) -> Path | None:
    """Locate the nearest .env, walking up from `start` to the filesystem root."""
    current = (start or Path.cwd()).resolve()
    for directory in [current, *current.parents]:
        candidate = directory / ENV_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_env(start: Path | None = None, override: bool = False) -> dict[str, str]:
    """Load .env into os.environ and return what it set.

    Real environment variables win by default, so an explicitly exported
    HF_TOKEN is never silently replaced by a stale file. Empty values are
    skipped so a blank placeholder cannot mask a working token.
    """
    path = find_env_file(start)
    if path is None:
        return {}
    try:
        parsed = _parse_env(path.read_text(encoding="utf-8"))
    except OSError:
        return {}

    applied = {}
    for key, value in parsed.items():
        if not value:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


# Load before BASE_DIR is computed so .env may also set INDIC_RUNNER_HOME.
load_env()

# Single Source of Truth for all internal operations
BASE_DIR = Path(os.getenv("INDIC_RUNNER_HOME", Path.home() / ".indic-runner"))

DIRS = {
    "bin": BASE_DIR / "bin",             # Static binaries (e.g., llama-server)
    "envs": BASE_DIR / "envs",           # Isolated Python virtualenvs
    "models": BASE_DIR / "models",       # Compiled inference-ready weights
    "manifests": BASE_DIR / "manifests", # JSON execution contracts
    "runs": BASE_DIR / "runs",           # Per-run partial rows and final results
}


# Environment variables that may carry a Hugging Face token, in the order
# huggingface_hub itself checks them.
HF_TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


def hf_token() -> str | None:
    """The configured Hugging Face token, if any.

    `.env` is loaded at import, so a token placed there is visible here with
    no further wiring. Needed for the registry's gated models, and it raises
    rate limits for the open ones.
    """
    for name in HF_TOKEN_VARS:
        token = os.environ.get(name)
        if token:
            return token
    return None
