# ComfySprites ↔ ComfyUI integration architecture

## Photo Studio asset flow

```
ComfySprites webapp (inject_assets.py)
    │
    ├─ Patch Downloader: ckpt_name + checkpoints_json + loras_json + controlnets_json + tokens
    ├─ Patch SDXL Loader: ckpt_name ← Downloader output; assets_ready ← Downloader output
    └─ Patch LoRA chain: ComfySpritesLoraLoader nodes (download already done)
            │
            ▼
    POST /prompt  →  ComfyUI executes Downloader first, then loaders
```

| Node | Class |
|------|--------|
| Downloader | `ComfySpritesDownloader` |
| Inference / refine checkpoint | `ComfySpritesSDXLLoader` |
| LoRA stack | `ComfySpritesLoraLoader` (chained) |
| Export still | `ComfySpritesExportImage` |

**v1:** One Downloader per Photo Studio graph. VAE / upscale weights are not auto-downloaded.

Video Lab still uses **ComfySprites Ensure LTX LoRAs** until migrated to the same Downloader pattern.

## Configuration

| Variable | Consumer | Purpose |
|----------|----------|---------|
| `COMFYUI_BASE_URL` | ComfySprites server | ComfyUI base URL |
| `CIVITAI_TOKEN` / `HF_TOKEN` | Injected on Downloader | Settings; not ComfyUI host env |

## Webapp modules

| Module | Role |
|--------|------|
| `webapp/comfyui/workflow.py` | Photo Studio patch + compose |
| `webapp/comfyui/asset_manifest.py` | Manifest builders |
| `webapp/comfyui/inject_assets.py` | Patch Downloader + SDXL Loader |
| `webapp/comfyui/lora_loader_chain.py` | Chain `ComfySpritesLoraLoader` nodes |
