"""ComfyUI-ComfySprites — asset download, SDXL load, export nodes."""

from __future__ import annotations

from .comfysprites_assets.download import ensure_all_assets
from .comfysprites_assets.download import ensure_loras_from_json
from .comfysprites_assets.paths import checkpoints_dir, loras_dir
from .comfysprites_export import export_audio, export_images, mux_video

_LOG = "[ComfySprites]"


class ComfySpritesDownloader:
    """Download checkpoints, LoRAs, and ControlNets; output inference ``ckpt_name``."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Inference checkpoint filename for the SDXL Loader.",
                    },
                ),
                "checkpoints_json": (
                    "STRING",
                    {"multiline": True, "default": "[]"},
                ),
                "loras_json": (
                    "STRING",
                    {"multiline": True, "default": "[]"},
                ),
                "controlnets_json": (
                    "STRING",
                    {"multiline": True, "default": "[]"},
                ),
                "civitai_token": ("STRING", {"default": ""}),
                "hf_token": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ckpt_name",)
    FUNCTION = "download"
    CATEGORY = "ComfySprites"

    def download(
        self,
        ckpt_name: str,
        checkpoints_json: str,
        loras_json: str,
        controlnets_json: str,
        civitai_token: str,
        hf_token: str,
    ):
        applied = ensure_all_assets(
            checkpoints_json=checkpoints_json,
            loras_json=loras_json,
            controlnets_json=controlnets_json,
            civitai_token=civitai_token or "",
            hf_token=hf_token or "",
        )
        name = (ckpt_name or "").strip()
        if not name:
            raise RuntimeError(f"{_LOG} ckpt_name is empty")
        parts: list[str] = []
        if applied["checkpoints"]:
            parts.append(f"checkpoints={', '.join(applied['checkpoints'])}")
        if applied["loras"]:
            parts.append(f"loras={', '.join(applied['loras'])}")
        if applied["controlnets"]:
            parts.append(f"controlnets={', '.join(applied['controlnets'])}")
        if parts:
            print(f"{_LOG} downloaded: {'; '.join(parts)}")
        print(f"{_LOG} assets ready; inference checkpoint: {name}")
        return (name,)


class ComfySpritesSDXLLoader:
    """Load an SDXL checkpoint that is already on disk (after Downloader)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    "STRING",
                    {"default": "", "tooltip": "Checkpoint filename under models/checkpoints/."},
                ),
                "assets_ready": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Wire from Downloader output so loads run after downloads.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_checkpoint"
    CATEGORY = "ComfySprites"

    def load_checkpoint(self, ckpt_name: str, assets_ready: str):
        import folder_paths
        import comfy.sd
        from pathlib import Path

        del assets_ready
        name = (ckpt_name or "").strip()
        if not name:
            raise RuntimeError(f"{_LOG} ckpt_name is empty")

        ckpt_path = folder_paths.get_full_path("checkpoints", name)
        if not ckpt_path:
            ckpt_path = str(checkpoints_dir() / name)
        if not ckpt_path or not Path(ckpt_path).is_file():
            raise RuntimeError(
                f"{_LOG} checkpoint {name!r} not found under models/checkpoints/"
            )

        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        print(f"{_LOG} SDXL checkpoint loaded: {name}")
        return (out[0], out[1], out[2])


class ComfySpritesLoraLoader:
    """Apply one LoRA (files must already exist; use Downloader first)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (
                    "STRING",
                    {"default": "", "tooltip": "LoRA filename under models/loras/."},
                ),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
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
    ):
        import folder_paths
        import comfy.sd
        import comfy.utils
        from pathlib import Path

        name = (lora_name or "").strip()
        if not name:
            raise RuntimeError(f"{_LOG} lora_name is empty")

        lora_path = folder_paths.get_full_path("loras", name)
        if not lora_path:
            lora_path = str(loras_dir() / name)
        if not lora_path or not Path(lora_path).is_file():
            raise RuntimeError(f"{_LOG} LoRA {name!r} not found under models/loras/")

        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model,
            clip,
            lora,
            float(strength_model),
            float(strength_clip),
        )
        return (model_lora, clip_lora)


class ComfySpritesEnsureLTXLoras:
    """Download LTX LoRAs from ``loras_json``, then pass model through unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "loras_json": (
                    "STRING",
                    {"multiline": True, "default": "[]"},
                ),
                "civitai_token": ("STRING", {"default": ""}),
                "hf_token": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "ensure"
    CATEGORY = "ComfySprites"

    def ensure(self, model, loras_json: str, civitai_token: str, hf_token: str):
        applied = ensure_loras_from_json(
            loras_json,
            civitai_token=civitai_token or "",
            hf_token=hf_token or "",
        )
        if applied:
            print(f"{_LOG} LTX LoRAs ready: {', '.join(applied)}")
        return (model,)


class ComfySpritesExportImage:
    """Strip metadata and compress images before ComfySprites downloads them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "format": (["webp", "jpeg", "png"], {"default": "webp"}),
                "quality": ("INT", {"default": 85, "min": 1, "max": 100, "step": 1}),
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
                "enabled": ("BOOLEAN", {"default": True}),
                "target_sample_rate": (
                    "INT",
                    {"default": 44100, "min": 8000, "max": 48000, "step": 1000},
                ),
                "mono": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "export"
    CATEGORY = "ComfySprites/Export"

    def export(self, audio, enabled: bool, target_sample_rate: int, mono: bool):
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
                "enabled": ("BOOLEAN", {"default": True}),
                "crf": ("INT", {"default": 20, "min": 0, "max": 51, "step": 1}),
                "audio_bitrate_kbps": (
                    "INT",
                    {"default": 128, "min": 32, "max": 320, "step": 8},
                ),
            },
            "optional": {"audio": ("AUDIO",)},
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


NODE_CLASS_MAPPINGS = {
    "ComfySpritesDownloader": ComfySpritesDownloader,
    "ComfySpritesSDXLLoader": ComfySpritesSDXLLoader,
    "ComfySpritesLoraLoader": ComfySpritesLoraLoader,
    "ComfySpritesEnsureLTXLoras": ComfySpritesEnsureLTXLoras,
    "ComfySpritesExportImage": ComfySpritesExportImage,
    "ComfySpritesExportAudio": ComfySpritesExportAudio,
    "ComfySpritesExportVideo": ComfySpritesExportVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfySpritesDownloader": "ComfySprites Downloader",
    "ComfySpritesSDXLLoader": "ComfySprites SDXL Loader",
    "ComfySpritesLoraLoader": "ComfySprites LoRA Loader",
    "ComfySpritesEnsureLTXLoras": "ComfySprites Ensure LTX LoRAs",
    "ComfySpritesExportImage": "ComfySprites Export Image",
    "ComfySpritesExportAudio": "ComfySprites Export Audio",
    "ComfySpritesExportVideo": "ComfySprites Export Video",
}
