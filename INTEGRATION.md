# ComfySprites ↔ ComfyUI integration architecture

How the **ComfySprites** webapp relates to a **local ComfyUI** instance and this **ComfyUI-ComfySprites** custom node pack. Use this when designing or debugging Photo / Video Lab generation.

---

## Three processes

| Component | Default URL | Role |
|-----------|-------------|------|
| **ComfySprites** (parent repo) | `http://127.0.0.1:8765` | Dataset editor, `POST /api/build`, `GET /api/dropdowns`, Photo / Video Lab UI |
| **ComfyUI** | configurable (`COMFYUI_BASE_URL`) | Executes node graphs; `/prompt`, `/history`, `/view`, `/ws` |
| **ComfyUI-ComfySprites** (this pack) | `ComfyUI/custom_nodes/` | LoRA ensure nodes; downloads on GPU host; tokens injected by ComfySprites webapp |

They are **separate processes**. ComfySprites does not need to live inside `custom_nodes/`.

---

## Make backend data flow

```
Make (Alpine.js)
    │
    ├─► POST /api/build          (ComfySprites) — compose prompts + inference
    │
    └─► POST /api/make/generate   (ComfySprites)
            │
            ├─ Load workflow template (Photo Studio.json)
            ├─ Patch nodes from build (checkpoint, LoRAs, latent, prompts, KSampler, batch_size)
            ├─ Patch node 128 (ComfySprites Ensure SDXL LoRAs): loras_json + Settings tokens
            ├─ Compose detailers from nodes/ templates when enabled
            ├─ WS /ws?clientId=…         (ComfyUI) — before queue
            ├─ POST /prompt              (ComfyUI)
            └─ On complete: download Save node 76 → outputs/photos/
```

ComfySprites is the **prompt and metadata authority**; ComfyUI is the **compute engine**.

### Configuration

| Variable | Consumer | Purpose |
|----------|----------|---------|
| `COMFYUI_BASE_URL` | ComfySprites server | ComfyUI base URL (Settings / workspace `.env`) |
| `CIVITAI_TOKEN` / `HF_TOKEN` | ComfySprites webapp → workflow injection | Settings; sent on ensure node inputs (not ComfyUI host env) |

---

## LoRA ensure nodes (this pack)

| Node | Photo / Video wiring |
|------|----------------------|
| **ComfySprites Ensure SDXL LoRAs** | Photo Studio — bulk download before chained `ComfySpritesEnsureLoraLoader` nodes |
| **ComfySprites Ensure LoRA Loader** | One LoRA per node (STRING `lora_name`); chain for style → character → partner → act |
| **ComfySprites Ensure LTX LoRAs** | Video Studio — between diffusion `257` and Power Lora `314` (or baked-in node id in your export) |
| **ComfySprites Export Image** | Photo Studio node **`132`** — strip metadata + compress before PreviewImage `131` |
| **ComfySprites Export Audio** | Video Studio node **`319`** — audio prep before mux |
| **ComfySprites Export Video** | Video Studio node **`59`** — metadata-free H.264 mux (ComfySprites download target) |

Before `POST /prompt`, ComfySprites ([`../webapp/comfyui/inject_assets.py`](../webapp/comfyui/inject_assets.py)) patches:

- `loras_json` — from `composer.build()`
- `civitai_token` / `hf_token` — from [`load_api_keys()`](../webapp/env_settings.py)
- `enabled` on export nodes — from `request.export_compress` (defaults **on**; UI toggle planned)

**v1:** SDXL checkpoints, LoRAs, and ControlNets auto-download on the ComfyUI host when manifest rows include `download_url` or `version_id`. VAE / upscale weights are not auto-downloaded.

### Install

1. Copy or symlink this folder to `ComfyUI/custom_nodes/ComfyUI-ComfySprites`.
2. Restart ComfyUI.
3. Set Civitai / HF tokens in ComfySprites **Settings**.

---

## Webapp modules

| Module | Role |
|--------|------|
| [`../webapp/comfyui/workflow.py`](../webapp/comfyui/workflow.py) | Photo Studio load / patch / node id map |
| [`../webapp/comfyui/asset_manifest.py`](../webapp/comfyui/asset_manifest.py) | LoRA lists for ensure nodes |
| [`../webapp/comfyui/inject_assets.py`](../webapp/comfyui/inject_assets.py) | Patch ensure node inputs + tokens |
| [`../webapp/comfyui/generate.py`](../webapp/comfyui/generate.py) | Queue Photo / Video Lab jobs |
| [`../webapp/comfyui/workflows/`](../webapp/comfyui/workflows/) | API-format JSON templates + patch docs |

Workflow patch tables: [`Photo Studio.md`](../webapp/comfyui/workflows/Photo%20Studio.md), [`Video Studio.md`](../webapp/comfyui/workflows/Video%20Studio.md).

---

## Related docs

- [README.md](README.md) — install and node inputs (this pack)
- [../README.md](../README.md) — run ComfySprites webapp
- [../.cursor/comfyui-engine.mdc](../.cursor/comfyui-engine.mdc) — WS progress and live previews
