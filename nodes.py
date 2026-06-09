"""ComfyUI-ComfySprites — asset download, SDXL load, export nodes."""

from __future__ import annotations

from .comfysprites_assets.download import ensure_all_assets
from .comfysprites_assets.download import ensure_loras_from_json
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


class ComfySpritesDownloadOutput:
    """Terminal output node for the asset-download workflow (satisfies ComfyUI ``OUTPUT_NODE``)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "message": (
                    "STRING",
                    {"default": "", "tooltip": "Wire from ComfySprites Downloader output."},
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "output"
    OUTPUT_NODE = True
    CATEGORY = "ComfySprites"

    def output(self, message: str):
        text = (message or "").strip() or "ok"
        print(f"{_LOG} download workflow complete: {text}")
        return {"ui": {"text": [text]}}


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
    "ComfySpritesDownloadOutput": ComfySpritesDownloadOutput,
    "ComfySpritesEnsureLTXLoras": ComfySpritesEnsureLTXLoras,
    "ComfySpritesExportImage": ComfySpritesExportImage,
    "ComfySpritesExportAudio": ComfySpritesExportAudio,
    "ComfySpritesExportVideo": ComfySpritesExportVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfySpritesDownloader": "ComfySprites Downloader",
    "ComfySpritesDownloadOutput": "ComfySprites Download Output",
    "ComfySpritesEnsureLTXLoras": "ComfySprites Ensure LTX LoRAs",
    "ComfySpritesExportImage": "ComfySprites Export Image",
    "ComfySpritesExportAudio": "ComfySprites Export Audio",
    "ComfySpritesExportVideo": "ComfySprites Export Video",
}
