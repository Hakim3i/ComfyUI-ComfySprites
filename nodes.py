"""ComfyUI-ComfySprites — LoRA ensure + export nodes for Make / Animate workflows."""

from __future__ import annotations

from .comfysprites_assets.download import (
    checkpoint_entry_for_name,
    ensure_checkpoint_file,
    ensure_controlnets_from_json,
    ensure_lora_file,
    ensure_loras_from_json,
    lora_entry_for_name,
)
from .comfysprites_assets.paths import checkpoints_dir, loras_dir
from .comfysprites_export import export_audio, export_images, mux_video

_LOG = "[ComfySprites ensure]"


class ComfySpritesExportImage:
    """Strip metadata and compress images before ComfySprites downloads them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When off, pass frames through unchanged.",
                    },
                ),
                "format": (
                    ["webp", "jpeg", "png"],
                    {
                        "default": "webp",
                        "tooltip": "Output encoding when enabled (metadata is always stripped).",
                    },
                ),
                "quality": (
                    "INT",
                    {
                        "default": 85,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Lossy quality for webp/jpeg.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "export"
    CATEGORY = "ComfySprites/Export"

    def export(self, images, enabled: bool, format: str, quality: int):
        return (export_images(images, enabled=enabled, fmt=format, quality=quality),)


class ComfySpritesExportAudio:
    """Prepare generated audio for lean, metadata-free video muxing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When off, pass audio through unchanged.",
                    },
                ),
                "target_sample_rate": (
                    "INT",
                    {
                        "default": 44100,
                        "min": 8000,
                        "max": 48000,
                        "step": 1000,
                        "tooltip": "Resample rate when enabled.",
                    },
                ),
                "mono": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Downmix to mono when enabled.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "export"
    CATEGORY = "ComfySprites/Export"

    def export(
        self,
        audio,
        enabled: bool,
        target_sample_rate: int,
        mono: bool,
    ):
        return (
            export_audio(
                audio,
                enabled=enabled,
                target_sample_rate=target_sample_rate,
                mono=mono,
            ),
        )


class ComfySpritesExportVideo:
    """Mux frames + audio into a metadata-free H.264 MP4 for ComfySprites download."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1},
                ),
                "filename_prefix": ("STRING", {"default": "ComfySprites/Video"}),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When off, keep higher quality (lower CRF / higher audio bitrate).",
                    },
                ),
                "crf": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 51,
                        "step": 1,
                        "tooltip": "H.264 CRF when compression is enabled.",
                    },
                ),
                "audio_bitrate_kbps": (
                    "INT",
                    {
                        "default": 128,
                        "min": 32,
                        "max": 320,
                        "step": 8,
                        "tooltip": "AAC bitrate when compression is enabled.",
                    },
                ),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = "ComfySprites/Export"

    def export(
        self,
        images,
        frame_rate: float,
        filename_prefix: str,
        enabled: bool,
        crf: int,
        audio_bitrate_kbps: int,
        audio=None,
    ):
        import folder_paths

        output_dir = folder_paths.get_temp_directory()
        entry = mux_video(
            images,
            frame_rate=frame_rate,
            audio=audio,
            filename_prefix=filename_prefix,
            output_dir=output_dir,
            enabled=enabled,
            crf=crf,
            audio_bitrate_kbps=audio_bitrate_kbps,
        )
        return {
            "ui": {
                "gifs": [
                    {
                        "filename": entry["filename"],
                        "subfolder": entry["subfolder"],
                        "type": entry["type"],
                        "format": entry["format"],
                        "frame_rate": entry["frame_rate"],
                    }
                ]
            }
        }


class ComfySpritesEnsureCheckpointLoader:
    """Download SDXL checkpoint if missing, then load MODEL + CLIP + VAE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Checkpoint filename under models/checkpoints/ (not a dropdown).",
                    },
                ),
                "checkpoints_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[]",
                        "tooltip": "JSON array of checkpoint rows from ComfySprites /api/build.",
                    },
                ),
                "civitai_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Injected by ComfySprites webapp from Settings.",
                    },
                ),
                "hf_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Injected by ComfySprites webapp from Settings.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_checkpoint"
    CATEGORY = "ComfySprites"

    def load_checkpoint(
        self,
        ckpt_name: str,
        checkpoints_json: str,
        civitai_token: str,
        hf_token: str,
    ):
        import folder_paths
        import comfy.sd

        name = (ckpt_name or "").strip()
        if not name:
            raise RuntimeError(f"{_LOG} ckpt_name is empty")

        entry = checkpoint_entry_for_name(checkpoints_json, name)
        if entry is not None:
            ensure_checkpoint_file(
                entry,
                civitai_token=civitai_token or "",
                hf_token=hf_token or "",
            )
        elif not (checkpoints_dir() / name).is_file():
            resolved = folder_paths.get_full_path("checkpoints", name)
            if not resolved:
                raise RuntimeError(
                    f"{_LOG} checkpoint {name!r} not on disk and no manifest entry in "
                    "checkpoints_json"
                )

        ckpt_path = folder_paths.get_full_path("checkpoints", name)
        if not ckpt_path:
            ckpt_path = str(checkpoints_dir() / name)
        from pathlib import Path

        if not ckpt_path or not Path(ckpt_path).is_file():
            raise RuntimeError(f"{_LOG} checkpoint file missing after ensure: {name!r}")

        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        print(f"{_LOG} SDXL checkpoint loaded: {name}")
        return (out[0], out[1], out[2])


class ComfySpritesEnsureLoraLoader:
    """Download one LoRA if missing, apply it, return updated MODEL + CLIP."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "LoRA filename under models/loras/ (not a dropdown).",
                    },
                ),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
                "loras_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[]",
                        "tooltip": "JSON array of LoRA rows from ComfySprites /api/build.",
                    },
                ),
                "civitai_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Injected by ComfySprites webapp from Settings.",
                    },
                ),
                "hf_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Injected by ComfySprites webapp from Settings.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load_lora"
    CATEGORY = "ComfySprites"

    def load_lora(
        self,
        model,
        clip,
        lora_name: str,
        strength_model: float,
        strength_clip: float,
        loras_json: str,
        civitai_token: str,
        hf_token: str,
    ):
        import folder_paths
        import comfy.sd
        import comfy.utils

        name = (lora_name or "").strip()
        if not name:
            raise RuntimeError(f"{_LOG} lora_name is empty")

        entry = lora_entry_for_name(loras_json, name)
        if entry is not None:
            ensure_lora_file(
                entry,
                civitai_token=civitai_token or "",
                hf_token=hf_token or "",
            )
        elif not (loras_dir() / name).is_file():
            resolved = folder_paths.get_full_path("loras", name)
            if not resolved:
                raise RuntimeError(
                    f"{_LOG} LoRA {name!r} not on disk and no manifest entry in loras_json"
                )

        lora_path = folder_paths.get_full_path("loras", name)
        if not lora_path:
            lora_path = str(loras_dir() / name)
        from pathlib import Path

        if not lora_path or not Path(lora_path).is_file():
            raise RuntimeError(f"{_LOG} LoRA file missing after ensure: {name!r}")

        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model,
            clip,
            lora,
            float(strength_model),
            float(strength_clip),
        )
        return (model_lora, clip_lora)


class ComfySpritesEnsureSDXLLoras:
    """Download SDXL LoRAs from ``loras_json``, then pass model/clip through unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "loras_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[]",
                        "tooltip": "JSON array of LoRA entries from ComfySprites /api/build.",
                    },
                ),
                "civitai_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Injected by ComfySprites webapp from Settings (not read from env).",
                    },
                ),
                "hf_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Injected by ComfySprites webapp from Settings (not read from env).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "ensure"
    CATEGORY = "ComfySprites"

    def ensure(
        self,
        model,
        clip,
        loras_json: str,
        civitai_token: str,
        hf_token: str,
    ):
        applied = ensure_loras_from_json(
            loras_json,
            civitai_token=civitai_token or "",
            hf_token=hf_token or "",
        )
        if applied:
            print(f"{_LOG} SDXL LoRAs ready: {', '.join(applied)}")
        return (model, clip)


class ComfySpritesEnsureLTXLoras:
    """Download LTX LoRAs from ``loras_json``, then pass model through unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "loras_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[]",
                        "tooltip": "JSON array of LTX LoRA entries from ComfySprites.",
                    },
                ),
                "civitai_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Injected by ComfySprites webapp from Settings.",
                    },
                ),
                "hf_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Injected by ComfySprites webapp from Settings.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "ensure"
    CATEGORY = "ComfySprites"

    def ensure(
        self,
        model,
        loras_json: str,
        civitai_token: str,
        hf_token: str,
    ):
        applied = ensure_loras_from_json(
            loras_json,
            civitai_token=civitai_token or "",
            hf_token=hf_token or "",
        )
        if applied:
            print(f"{_LOG} LTX LoRAs ready: {', '.join(applied)}")
        return (model,)


class ComfySpritesEnsureControlNets:
    """Download ControlNet weights from ``controlnets_json`` (side-effect only)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnets_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[]",
                        "tooltip": "JSON array of ControlNet files from ComfySprites.",
                    },
                ),
                "civitai_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Injected by ComfySprites webapp from Settings.",
                    },
                ),
                "hf_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Injected by ComfySprites webapp from Settings.",
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "ensure"
    CATEGORY = "ComfySprites"
    OUTPUT_NODE = True

    def ensure(self, controlnets_json: str, civitai_token: str, hf_token: str):
        applied = ensure_controlnets_from_json(
            controlnets_json,
            civitai_token=civitai_token or "",
            hf_token=hf_token or "",
        )
        if applied:
            print(f"{_LOG} ControlNets ready: {', '.join(applied)}")
        return ()


NODE_CLASS_MAPPINGS = {
    "ComfySpritesEnsureCheckpointLoader": ComfySpritesEnsureCheckpointLoader,
    "ComfySpritesEnsureLoraLoader": ComfySpritesEnsureLoraLoader,
    "ComfySpritesEnsureSDXLLoras": ComfySpritesEnsureSDXLLoras,
    "ComfySpritesEnsureLTXLoras": ComfySpritesEnsureLTXLoras,
    "ComfySpritesEnsureControlNets": ComfySpritesEnsureControlNets,
    "ComfySpritesExportImage": ComfySpritesExportImage,
    "ComfySpritesExportAudio": ComfySpritesExportAudio,
    "ComfySpritesExportVideo": ComfySpritesExportVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfySpritesEnsureCheckpointLoader": "ComfySprites Ensure Checkpoint Loader",
    "ComfySpritesEnsureLoraLoader": "ComfySprites Ensure LoRA Loader",
    "ComfySpritesEnsureSDXLLoras": "ComfySprites Ensure SDXL LoRAs",
    "ComfySpritesEnsureLTXLoras": "ComfySprites Ensure LTX LoRAs",
    "ComfySpritesEnsureControlNets": "ComfySprites Ensure ControlNets",
    "ComfySpritesExportImage": "ComfySprites Export Image",
    "ComfySpritesExportAudio": "ComfySprites Export Audio",
    "ComfySpritesExportVideo": "ComfySprites Export Video",
}
