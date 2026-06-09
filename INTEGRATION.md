# ComfySprites ↔ ComfyUI integration architecture

## Make asset flow (two-phase)

```
ComfySprites webapp
    │
    ├─ GET /models/checkpoints|loras|controlnet — compare vs build manifest
    ├─ If missing: POST /prompt asset_download workflow (ComfySpritesDownloader only)
    ├─ Wait for download completion
    └─ POST /prompt Make workflow (stock CheckpointLoader / LoraLoader / ControlNetLoader)
```

| Phase | ComfyUI graph |
|-------|----------------|
| Asset download | `ComfySpritesDownloader` only ([`download_workflow.py`](../webapp/comfyui/download_workflow.py)) |
| Make generation | Standard ComfyUI loaders + KSampler + export nodes |

| Node pack | Class |
|-----------|--------|
| Downloader | `ComfySpritesDownloader` |
| Export still | `ComfySpritesExportImage` |

## Configuration

| Variable | Consumer | Purpose |
|----------|----------|---------|
| `COMFYUI_BASE_URL` | ComfySprites server | ComfyUI base URL |
| `CIVITAI_TOKEN` / `HF_TOKEN` | Downloader workflow | Settings; injected on download prompt |

## Webapp modules

| Module | Role |
|--------|------|
| `webapp/comfyui/asset_inventory.py` | Required vs installed asset check |
| `webapp/comfyui/download_workflow.py` | Build download-only workflow |
| `webapp/comfyui/generate.py` | Queue download then Make |
| `webapp/comfyui/workflow.py` | Compose Make workflow (no downloader node) |
| `webapp/comfyui/lora_loader_chain.py` | Chain stock `LoraLoader` nodes |
