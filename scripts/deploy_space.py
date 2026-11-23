"""Create or update the Hugging Face Space that hosts the live demo.

Usage: uv run python scripts/deploy_space.py [space_name]

- Reads the HF write token from HF_TOKEN (or HF_API_KEY) and the Groq keys from settings.
- Creates a Docker Space (if missing), stores the Groq keys as an encrypted Space SECRET
  (never uploaded as a file, never printed) and non-secret settings as Space VARIABLES.
- Uploads the repository files the image needs, with a Space-specific README (the YAML
  header tells Hugging Face to build the Dockerfile and expose port 7860).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from carepilot.config import ROOT, get_settings

UPLOAD_PATTERNS = [
    "Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock",
    "src/**", "app/**", "data/**", "models/**", "index/**", "reports/**",
]  # fmt: skip
IGNORE_PATTERNS = ["**/__pycache__/**", "**/*.pyc", ".env", "zdnd/**", "CLAUDE.md"]

SPACE_README = """---
title: CarePilot
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Diabetes & heart risk, explained with cited NIH/WHO docs
---

# CarePilot (live demo)

Diabetes and heart-disease risk prediction, explained per patient by a citation-grounded
RAG pipeline that abstains when the documents do not support an answer.

Demonstration prototype, not medical advice. Source code and full documentation: see the
GitHub repository linked in the project README.
"""


def main() -> None:
    token = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")
    if not token:
        sys.exit("Set HF_TOKEN (a Hugging Face WRITE token) in .env first.")
    settings = get_settings()
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{sys.argv[1] if len(sys.argv) > 1 else 'carepilot'}"

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    if settings.groq_api_keys:
        api.add_space_secret(repo_id, "GROQ_API_KEY", ",".join(settings.groq_api_keys))
    api.add_space_variable(repo_id, "LLM_PROVIDER", "auto")
    api.add_space_variable(repo_id, "GROQ_MODEL", settings.groq_model)

    with tempfile.TemporaryDirectory() as tmp:
        readme = Path(tmp) / "README.md"
        readme.write_text(SPACE_README, encoding="utf-8")
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md", repo_id=repo_id,
                        repo_type="space")  # fmt: skip
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=repo_id,
        repo_type="space",
        allow_patterns=UPLOAD_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Deploy CarePilot",
    )
    print(f"Space: https://huggingface.co/spaces/{repo_id}")
    print(f"App:   https://{repo_id.replace('/', '-').lower()}.hf.space")
    print(f"Groq keys stored as secret: {len(settings.groq_api_keys)} (values not shown)")


if __name__ == "__main__":
    main()
