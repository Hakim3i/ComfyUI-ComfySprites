"""ComfyUI-ComfySprites — LoRA ensure + export nodes for Make / Animate workflows."""

from __future__ import annotations

from .comfysprites_assets.download import ensure_controlnets_from_json, ensure_loras_from_json
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
    "ComfySpritesEnsureSDXLLoras": ComfySpritesEnsureSDXLLoras,
    "ComfySpritesEnsureLTXLoras": ComfySpritesEnsureLTXLoras,
    "ComfySpritesEnsureControlNets": ComfySpritesEnsureControlNets,
    "ComfySpritesExportImage": ComfySpritesExportImage,
    "ComfySpritesExportAudio": ComfySpritesExportAudio,
    "ComfySpritesExportVideo": ComfySpritesExportVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfySpritesEnsureSDXLLoras": "ComfySprites Ensure SDXL LoRAs",
    "ComfySpritesEnsureLTXLoras": "ComfySprites Ensure LTX LoRAs",
    "ComfySpritesEnsureControlNets": "ComfySprites Ensure ControlNets",
    "ComfySpritesExportImage": "ComfySprites Export Image",
    "ComfySpritesExportAudio": "ComfySprites Export Audio",
    "ComfySpritesExportVideo": "ComfySprites Export Video",
}
