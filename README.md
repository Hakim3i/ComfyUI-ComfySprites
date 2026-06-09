# ComfyUI-ComfySprites

Asset download + export nodes for **ComfySprites** Make.

- **Editor:** https://github.com/Hakim3i/ComfySprites
- **This pack:** https://github.com/Hakim3i/ComfyUI-ComfySprites

## Make flow (two-phase)

1. **Webapp preflight** — compare required models vs `GET /models/*` on ComfyUI.
2. **Download workflow** — if anything is missing, queue a single `ComfySpritesDownloader` node.
3. **Make workflow** — standard ComfyUI `CheckpointLoaderSimple`, `LoraLoader`, `ControlNetLoader`, etc.

## Nodes

| Node | Role |
|------|------|
| **ComfySprites Downloader** | Download checkpoints, LoRAs, ControlNets from JSON manifests |
| **ComfySprites Ensure LTX LoRAs** | Bulk LTX LoRA download (legacy) |
| **ComfySprites Export Image / Audio / Video** | Metadata-free export helpers |

### Downloader inputs (injected by ComfySprites webapp)

| Input | Description |
|-------|-------------|
| `ckpt_name` | Inference checkpoint filename (required by node) |
| `checkpoints_json` | Missing checkpoint rows |
| `loras_json` | Missing LoRA rows |
| `controlnets_json` | Missing ControlNet rows |
| `civitai_token` / `hf_token` | From ComfySprites Settings |

Ensure nodes **do not** read tokens from the ComfyUI process environment.

## Install

```
ComfyUI/custom_nodes/ComfyUI-ComfySprites
```

Restart ComfyUI after updating.

## Docs

- [INTEGRATION.md](INTEGRATION.md)
