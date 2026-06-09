"""Tests for LoRA download URL resolution."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfysprites_assets.download_utils import (
    civitai_authenticated_url,
    download_candidates,
)


def test_download_candidates_primary_and_version_id():
    info = {
        "download_url": "https://example.com/a.safetensors",
        "version_id": 99,
    }
    urls = download_candidates(info)
    assert urls[0] == "https://example.com/a.safetensors"
    assert any("civitai" in u and "99" in u for u in urls)


def test_civitai_authenticated_url_appends_token_query():
    url = civitai_authenticated_url(
        "https://civitai.com/api/download/models/2883731",
        "secret-token",
    )
    assert url == "https://civitai.com/api/download/models/2883731?token=secret-token"


def test_civitai_authenticated_url_appends_token_ampersand():
    url = civitai_authenticated_url(
        "https://civitai.red/api/download/models/1?type=Model",
        "secret-token",
    )
    assert url.endswith("&token=secret-token")
    assert "type=Model" in url
