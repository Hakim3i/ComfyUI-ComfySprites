# ComfyUI-ComfySprites

Asset download + SDXL load support for **ComfySprites** Make and Video Lab.

- **Editor:** https://github.com/Hakim3i/ComfySprites
- **This pack:** https://github.com/Hakim3i/ComfyUI-ComfySprites

## Photo Studio flow

```
ComfySprites Downloader  →  ComfySprites SDXL Loader  →  ComfySprites LoRA Loader (chain)  →  …
```

1. **Downloader** — one node; downloads all checkpoints, LoRAs, and ControlNets from JSON manifests + API keys; outputs inference `ckpt_name`.
2. **SDXL Loader** — loads `ckpt_name` (wire `assets_ready` from Downloader for ordering).
3. **LoRA Loader** — apply one LoRA per node; chain for style → character → partner → act.

## Nodes

| Node | Role |
|------|------|
| **ComfySprites Downloader** | Download all assets; output inference checkpoint filename |
| **ComfySprites SDXL Loader** | Load checkpoint MODEL + CLIP + VAE (after Downloader) |
| **ComfySprites LoRA Loader** | Apply one LoRA (chain for stacks) |
| **ComfySprites Ensure LTX LoRAs** | Video Lab LTX LoRA bulk download (passthrough) |
| **ComfySprites Export Image / Audio / Video** | Metadata-free export helpers |

### Downloader inputs (injected by ComfySprites webapp)

| Input | Description |
|-------|-------------|
| `ckpt_name` | Inference checkpoint filename |
| `checkpoints_json` | All checkpoint rows (inference + refine) |
| `loras_json` | All SDXL LoRA rows |
| `controlnets_json` | ControlNet weight rows |
| `civitai_token` / `hf_token` | From ComfySprites Settings |

Ensure nodes **do not** read tokens from the ComfyUI process environment.

## Install

```
ComfyUI/custom_nodes/ComfyUI-ComfySprites
```

Restart ComfyUI after updating.

## Docs

- [INTEGRATION.md](INTEGRATION.md)
