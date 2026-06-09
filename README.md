# ComfyUI-ComfySprites

LoRA download support for **ComfySprites** Make and Video Lab. The ComfySprites webapp injects ensure nodes into API workflows before `POST /prompt`; ComfyUI downloads missing files into `models/loras/`.

- **Editor:** https://github.com/Hakim3i/ComfySprites
- **This pack:** https://github.com/Hakim3i/ComfyUI-ComfySprites

## Nodes

| Node | Role |
|------|------|
| **ComfySprites Ensure Checkpoint Loader** | Download SDXL checkpoints from `checkpoints_json`, then load `MODEL` + `CLIP` + `VAE` |
| **ComfySprites Ensure SDXL LoRAs** | Download SDXL LoRAs from `loras_json`, pass `MODEL` + `CLIP` through |
| **ComfySprites Ensure LTX LoRAs** | Download LTX LoRAs from `loras_json`, pass `MODEL` through |
| **ComfySprites Export Image** | Strip metadata and compress stills (WebP/JPEG/PNG) before ComfySprites download |
| **ComfySprites Export Audio** | Resample / downmix audio before video mux |
| **ComfySprites Export Video** | Metadata-free H.264 MP4 mux for Video Lab (replaces VHS combine in ComfySprites workflows) |

### Inputs (injected by ComfySprites webapp)

| Input | Description |
|-------|-------------|
| `checkpoints_json` | JSON array of checkpoint rows from Style (`filename`, `download_url`, `version_id`, …) |
| `loras_json` | JSON array of LoRA rows (`filename`, `download_url`, `version_id`, …) |
| `civitai_token` | From ComfySprites **Settings** (workspace `.env`) |
| `hf_token` | From ComfySprites **Settings** |

Ensure nodes **do not** read `CIVITAI_TOKEN` / `HF_TOKEN` from the ComfyUI process environment.

## Install

Copy or symlink this folder into ComfyUI:

```
ComfyUI/custom_nodes/ComfyUI-ComfySprites
```

Restart ComfyUI. Photo / Video Lab queueing requires these node types on the ComfyUI host.

**Video export** needs `ffmpeg` on the ComfyUI host `PATH` (or `imageio-ffmpeg` in the ComfyUI Python env).

## v1 scope

- **SDXL checkpoints + LoRAs + ControlNets** — auto-download when Style / catalog rows include `download_url` or `version_id`
- VAE and upscale weights must already exist on the ComfyUI host

## Docs

- [INTEGRATION.md](INTEGRATION.md) — how ComfySprites webapp, ComfyUI, and this pack fit together (Photo / Video Lab flow, tokens, webapp modules)

## Files

| Path | Role |
|------|------|
| `comfysprites_assets/download.py` | Civitai / HF download into `models/loras/`, `models/checkpoints/`, `models/controlnet/` |
| `nodes.py` | ComfyUI node registrations |
| `comfysprites_export/` | Strip metadata + compress image/audio/video helpers |
