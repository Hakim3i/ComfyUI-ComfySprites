"""Download LoRA files into ComfyUI ``models/loras/``."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .download_utils import (
    civitai_authenticated_url,
    download_candidates,
    is_civitai_url,
    is_huggingface_url,
)
from .paths import checkpoints_dir, controlnet_dir, loras_dir

_LOG = "[ComfySprites LoRA]"
_CN_LOG = "[ComfySprites ControlNet]"
_CKPT_LOG = "[ComfySprites Checkpoint]"


def _format_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def _format_download_error(exc: Exception | None) -> str:
    if exc is None:
        return "unknown error"
    if isinstance(exc, urllib.error.HTTPError):
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:240]
        except Exception:
            pass
        detail = f"HTTP {exc.code} {exc.reason}"
        if body:
            detail = f"{detail}: {body}"
        return detail
    return str(exc)


def _download_file(
    url: str,
    target: Path,
    *,
    civitai_token: str,
    hf_token: str,
    label: str,
    log: str = _LOG,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    headers = {"User-Agent": "comfyui-comfysprites/1.0"}
    request_url = url
    if is_civitai_url(url):
        token = (civitai_token or "").strip()
        if token:
            # Query token only — do not send Authorization on Civitai URLs. urllib
            # forwards that header to the S3/R2 redirect target and S3 returns 400.
            request_url = civitai_authenticated_url(url, token)
    elif is_huggingface_url(url):
        token = (hf_token or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(request_url, headers=headers)
    print(f"{log} DOWNLOAD START")
    print(f"{log}   Asset: {label}")
    print(f"{log}   URL:  {url}")
    print(f"{log}   Save: {target}")
    with urllib.request.urlopen(req, timeout=600) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        if total:
            print(f"{log}   Size: {_format_mb(total)}")
        downloaded = 0
        with open(tmp, "wb") as handle:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = min(100, downloaded * 100 // total)
                    print(
                        f"{log}   Progress: {_format_mb(downloaded)} / "
                        f"{_format_mb(total)} ({pct}%)",
                        end="\r",
                        flush=True,
                    )
        print()
    tmp.replace(target)
    print(f"{log} DOWNLOAD OK -> {target.name}")


def ensure_lora_file(
    info: dict[str, Any],
    *,
    civitai_token: str = "",
    hf_token: str = "",
) -> Path:
    """Ensure ``info['filename']`` exists under ``models/loras/``; download if missing.

    Uses only the token strings passed in (no ``os.environ``). Raises on failure.
    """
    filename = info.get("filename")
    name = info.get("name") or filename or "?"
    if not filename or not str(filename).strip():
        raise RuntimeError(f"{_LOG} catalog entry missing filename ({name!r})")
    filename = str(filename).strip()
    target = loras_dir() / filename
    if target.is_file():
        print(f"{_LOG} {filename}: on disk ({_format_mb(target.stat().st_size)})")
        return target

    urls = download_candidates(info)
    if not urls:
        raise RuntimeError(
            f"{_LOG} {filename!r}: missing download_url and version_id; "
            "add a download URL on the LoRA row in ComfySprites."
        )

    civitai_token = (civitai_token or "").strip()
    hf_token = (hf_token or "").strip()
    last_error: Exception | None = None
    for idx, url in enumerate(urls):
        src = "huggingface" if is_huggingface_url(url) else "civitai"
        try:
            if idx:
                print(f"{_LOG} retrying with {src} mirror ({idx + 1}/{len(urls)})")
            _download_file(
                url,
                target,
                civitai_token=civitai_token,
                hf_token=hf_token,
                label=str(name),
                log=_LOG,
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            print(f"{_LOG} DOWNLOAD FAILED: {_format_download_error(exc)} ({url})")

    if not target.is_file():
        raise RuntimeError(
            f"{_LOG} could not download {filename!r}: {_format_download_error(last_error)}"
        )
    return target


def ensure_loras_from_json(
    loras_json: str,
    *,
    civitai_token: str = "",
    hf_token: str = "",
) -> list[str]:
    """Parse a JSON list of LoRA dicts and ensure each file exists. Returns filenames."""
    import json

    raw = (loras_json or "").strip()
    if not raw:
        return []
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{_LOG} invalid loras_json: {exc}") from exc
    if not isinstance(entries, list):
        raise RuntimeError(f"{_LOG} loras_json must be a JSON array")
    applied: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path = ensure_lora_file(
            entry,
            civitai_token=civitai_token,
            hf_token=hf_token,
        )
        applied.append(path.name)
    return applied


def ensure_controlnet_file(
    info: dict[str, Any],
    *,
    civitai_token: str = "",
    hf_token: str = "",
) -> Path:
    filename = info.get("filename")
    if not filename or not str(filename).strip():
        raise RuntimeError(f"{_CN_LOG} catalog entry missing filename")
    filename = str(filename).strip()
    target = controlnet_dir() / filename
    if target.is_file():
        print(f"{_CN_LOG} {filename}: on disk ({_format_mb(target.stat().st_size)})")
        return target
    urls = download_candidates(info)
    if not urls:
        raise RuntimeError(f"{_CN_LOG} {filename!r}: missing download_url")
    civitai_token = (civitai_token or "").strip()
    hf_token = (hf_token or "").strip()
    last_error: Exception | None = None
    for idx, url in enumerate(urls):
        src = "huggingface" if is_huggingface_url(url) else "civitai"
        try:
            if idx:
                print(f"{_CN_LOG} retrying with {src} mirror ({idx + 1}/{len(urls)})")
            _download_file(
                url,
                target,
                civitai_token=civitai_token,
                hf_token=hf_token,
                label=filename,
                log=_CN_LOG,
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            print(f"{_CN_LOG} DOWNLOAD FAILED: {_format_download_error(exc)} ({url})")
    if not target.is_file():
        raise RuntimeError(
            f"{_CN_LOG} could not download {filename!r}: {_format_download_error(last_error)}"
        )
    return target


def ensure_controlnets_from_json(
    controlnets_json: str,
    *,
    civitai_token: str = "",
    hf_token: str = "",
) -> list[str]:
    import json

    raw = (controlnets_json or "").strip()
    if not raw:
        return []
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{_CN_LOG} invalid controlnets_json: {exc}") from exc
    if not isinstance(entries, list):
        raise RuntimeError(f"{_CN_LOG} controlnets_json must be a JSON array")
    applied: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path = ensure_controlnet_file(
            entry,
            civitai_token=civitai_token,
            hf_token=hf_token,
        )
        applied.append(path.name)
    return applied


def ensure_checkpoint_file(
    info: dict[str, Any],
    *,
    civitai_token: str = "",
    hf_token: str = "",
) -> Path:
    """Ensure ``info['filename']`` exists under ``models/checkpoints/``."""
    filename = info.get("filename")
    name = info.get("name") or filename or "?"
    if not filename or not str(filename).strip():
        raise RuntimeError(f"{_CKPT_LOG} catalog entry missing filename ({name!r})")
    filename = str(filename).strip()
    target = checkpoints_dir() / filename
    if target.is_file():
        print(f"{_CKPT_LOG} {filename}: on disk ({_format_mb(target.stat().st_size)})")
        return target

    urls = download_candidates(info)
    if not urls:
        raise RuntimeError(
            f"{_CKPT_LOG} {filename!r}: missing download_url and version_id; "
            "add a download URL on the Style row in ComfySprites."
        )

    civitai_token = (civitai_token or "").strip()
    hf_token = (hf_token or "").strip()
    last_error: Exception | None = None
    for idx, url in enumerate(urls):
        src = "huggingface" if is_huggingface_url(url) else "civitai"
        try:
            if idx:
                print(f"{_CKPT_LOG} retrying with {src} mirror ({idx + 1}/{len(urls)})")
            _download_file(
                url,
                target,
                civitai_token=civitai_token,
                hf_token=hf_token,
                label=str(name),
                log=_CKPT_LOG,
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            print(f"{_CKPT_LOG} DOWNLOAD FAILED: {_format_download_error(exc)} ({url})")

    if not target.is_file():
        raise RuntimeError(
            f"{_CKPT_LOG} could not download {filename!r}: {_format_download_error(last_error)}"
        )
    return target


def ensure_checkpoints_from_json(
    checkpoints_json: str,
    *,
    civitai_token: str = "",
    hf_token: str = "",
) -> list[str]:
    """Parse a JSON list of checkpoint dicts and ensure each file exists."""
    import json

    raw = (checkpoints_json or "").strip()
    if not raw:
        return []
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{_CKPT_LOG} invalid checkpoints_json: {exc}") from exc
    if not isinstance(entries, list):
        raise RuntimeError(f"{_CKPT_LOG} checkpoints_json must be a JSON array")
    applied: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path = ensure_checkpoint_file(
            entry,
            civitai_token=civitai_token,
            hf_token=hf_token,
        )
        applied.append(path.name)
    return applied


def ensure_all_assets(
    *,
    checkpoints_json: str = "",
    loras_json: str = "",
    controlnets_json: str = "",
    civitai_token: str = "",
    hf_token: str = "",
) -> dict[str, list[str]]:
    """Download every asset listed in the three JSON manifests."""
    civitai_token = (civitai_token or "").strip()
    hf_token = (hf_token or "").strip()
    return {
        "checkpoints": ensure_checkpoints_from_json(
            checkpoints_json,
            civitai_token=civitai_token,
            hf_token=hf_token,
        ),
        "loras": ensure_loras_from_json(
            loras_json,
            civitai_token=civitai_token,
            hf_token=hf_token,
        ),
        "controlnets": ensure_controlnets_from_json(
            controlnets_json,
            civitai_token=civitai_token,
            hf_token=hf_token,
        ),
    }


def lora_entry_for_name(
    loras_json: str,
    lora_name: str,
) -> dict[str, Any] | None:
    """Return the manifest row matching *lora_name*, if any."""
    import json

    name = (lora_name or "").strip()
    if not name:
        return None
    raw = (loras_json or "").strip()
    if not raw:
        return None
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(entries, list):
        return None
    key = name.lower()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        filename = str(entry.get("filename") or "").strip()
        if filename.lower() == key:
            return entry
    return None


def checkpoint_entry_for_name(
    checkpoints_json: str,
    ckpt_name: str,
) -> dict[str, Any] | None:
    """Return the manifest row matching *ckpt_name*, if any."""
    import json

    name = (ckpt_name or "").strip()
    if not name:
        return None
    raw = (checkpoints_json or "").strip()
    if not raw:
        return None
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(entries, list):
        return None
    key = name.lower()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        filename = str(entry.get("filename") or "").strip()
        if filename.lower() == key:
            return entry
    return None
