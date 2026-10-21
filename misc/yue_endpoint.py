# ---
# lambda-test: false
# ---

# # YuE Music Generation, Zero-Shot Cover & Extension Endpoint on Modal
#
# This script deploys a comprehensive music generation, zero-shot cover, and song extension
# API on Modal using **YuE2** and **SheetSage2**
# (from [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)).
#
# ## Core Capabilities
# 1. **Full-Song Generation (`/generate`, `/generate_audio`)**:
#    - Generates 48 kHz stereo music from text lyrics and style/genre prompts.
#    - Uses symbolic planning (CoT) to compose structured ABC notation scores before audio synthesis.
# 2. **Zero-Shot Cover (`/cover`, `/cover_audio`)**:
#    - Takes an audio file (`audio_url` or `audio_base64`) or existing ABC score (`score_abc`).
#    - Transcribes the melody using SheetSage2, locks the vocal/instrumental melody (`cot="melody"`),
#      and re-orchestrates the entire song in a new style (e.g. pop -> metal, acoustic -> synthwave).
# 3. **Song Extension & Continuation (`/extend`, `/extend_audio`)**:
#    - Extends an existing song with new lyrical sections (`[Verse 2]`, `[Bridge]`, `[Outro]`).
#    - Reuses the musical theme/ABC score from previous generation or audio to ensure seamless continuation.
# 4. **Audio-to-Score Transcription (`/transcribe`)**:
#    - Transcribes any audio into an editable ABC notation score using SheetSage2.
#
# ## Hardware & Deployment
# - **GPU**: NVIDIA A10G (24 GB VRAM) running native BF16 inference with 22 GiB budget.
# - **Storage**: Persistent Modal Volume (`yue_model_cache`) caching YuE2-3B, YuE2-Vae, and SheetSage2.

from __future__ import annotations

import base64
import os
import secrets
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional

import modal
from fastapi import Depends, Header, HTTPException, Security
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Storage and Volume Configuration
# ---------------------------------------------------------------------------

CONTAINER_CACHE_DIR = Path("/cache")
CONTAINER_TEMP_DIR = Path("/tmp/yue_work")

CONTAINER_CACHE_VOLUME = modal.Volume.from_name(
    "yue_model_cache", create_if_missing=True
)

YUE_MODEL_ID = "m-a-p/YuE2-3B"
VAE_ID = "m-a-p/YuE2-Vae"
SHEETSAGE_MODEL_ID = "m-a-p/SheetSage2"


# ---------------------------------------------------------------------------
# Modal Secrets Configuration
# ---------------------------------------------------------------------------
# Static list of secrets mounted into containers.
# Must be consistent between local deploy and remote container imports.
CONFIGURED_SECRETS = [
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("yue-api-secret"),
]

# ---------------------------------------------------------------------------
# Authentication Security Dependency
# ---------------------------------------------------------------------------

security_bearer = HTTPBearer(auto_error=False)


def verify_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> bool:
    """Validate Bearer token or X-API-Key header if YUE_API_KEY is configured.

    If YUE_API_KEY is not set in the environment, endpoints remain open for testing.
    """
    expected_key = os.environ.get("YUE_API_KEY")
    if not expected_key:
        return True

    provided_key = None
    if auth and auth.credentials:
        provided_key = auth.credentials
    elif x_api_key:
        provided_key = x_api_key

    if not provided_key or not secrets.compare_digest(provided_key, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing API key. Provide header 'Authorization: Bearer <token>' or 'X-API-Key: <token>'.",
        )
    return True


# ---------------------------------------------------------------------------
# Container Image Definitions
# ---------------------------------------------------------------------------

cuda_version = "12.4.0"
flavor = "devel"
operating_system = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_system}"

nvidia_cuda_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.12"
).entrypoint([])

# Image for YuE2 Music Generation Pipeline
yue_image = (
    nvidia_cuda_image.apt_install(
        "git",
        "ffmpeg",
        "libsndfile1",
        "curl",
        "wget",
    )
    .pip_install(
        "torch>=2.4.0",
        "torchaudio>=2.4.0",
        "transformers>=4.48.0",
        "accelerate>=1.0.0",
        "safetensors>=0.4.3",
        "tiktoken>=0.7.0",
        "numpy>=1.26.0",
        "soundfile>=0.12.1",
        "scipy>=1.11.0",
        "fastapi[standard]>=0.115.0",
        "pydantic>=2.10.0",
        "huggingface-hub[hf_transfer]>=0.25.0",
        "requests",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/multimodal-art-projection/YuE.git /opt/YuE",
        "cd /opt/YuE && pip install -e . --no-deps",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_CACHE": str(CONTAINER_CACHE_DIR / ".hf_hub_cache"),
        }
    )
)

# Image for SheetSage2 Audio-to-Score Transcription (Python 3.11 for MERT-v2)
sheetsage_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch>=2.4.0",
        "torchaudio>=2.4.0",
        "transformers>=4.45.0",
        "huggingface-hub[hf_transfer]>=0.25.0",
        "safetensors",
        "numpy",
        "scipy",
        "mir_eval",
        "pretty_midi",
        "mido",
        "requests",
        "fastapi[standard]>=0.115.0",
        "pydantic>=2.10.0",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_CACHE": str(CONTAINER_CACHE_DIR / ".hf_hub_cache"),
        }
    )
)

# Lightweight image for instant health checks
health_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]>=0.115.0"
)

# ---------------------------------------------------------------------------
# Modal App Setup
# ---------------------------------------------------------------------------

app = modal.App("yue-music-generator", image=yue_image)

with yue_image.imports():
    import subprocess
    import time

    import torch
    from yue2 import YuE2Pipeline


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------


class OutputFormat(str, Enum):
    """Audio output format options."""

    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"

    @property
    def media_type(self) -> str:
        return {
            OutputFormat.MP3: "audio/mpeg",
            OutputFormat.WAV: "audio/wav",
            OutputFormat.FLAC: "audio/flac",
        }[self]


class CoTMode(str, Enum):
    """Chain-of-Thought (Symbolic Planning) mode."""

    FULL = "full"  # Generate both melody and chords in ABC notation
    MELODY = "melody"  # Generate melody-only (best for zero-shot covers)
    OFF = "off"  # Direct audio generation without symbolic planning


class SongGenerationRequest(BaseModel):
    """Input parameters for standard YuE music generation."""

    lyrics: str = Field(
        ...,
        description="Lyrics with section tags, e.g. [Verse] ... [Chorus] ...",
        example="[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n[Chorus]\nLet the day come into view\nEvery road begins with you",
    )
    style: str = Field(
        ...,
        description="Music style description including genre, instruments, vocal character, tempo (BPM)",
        example="English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, 88 BPM",
    )
    cot: CoTMode = Field(
        default=CoTMode.FULL,
        description="Chain-of-thought symbolic planning mode: 'full' (melody + chords), 'melody' (melody only), or 'off'",
    )
    seed: Optional[int] = Field(
        default=831001, description="Random seed for deterministic generation"
    )
    abc: Optional[str] = Field(
        default=None,
        description="Optional custom ABC notation score to guide composition directly",
    )
    cfg_scale: Optional[float] = Field(
        default=1.0,
        ge=1.0,
        le=5.0,
        description="Classifier-free guidance scale for stylistic alignment",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.MP3, description="Audio output format: mp3, wav, or flac"
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="Optional webhook URL to receive results asynchronously.",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Optional tracking ID returned in responses and webhooks",
    )


class CoverSongRequest(BaseModel):
    """Input parameters for zero-shot audio covers."""

    style: str = Field(
        ...,
        description="Target style for the cover (e.g. 'Heavy metal, distorted electric guitars, fast drums, energetic rock vocals, 130 BPM')",
        example="Heavy metal, energetic rock vocals, distorted guitars, 130 BPM",
    )
    lyrics: str = Field(
        ...,
        description="Lyrics of the song to sing, matching the reference melody",
        example="[Verse]\nNeon fades along the lane\n[Chorus]\nLet the day come into view",
    )
    audio_url: Optional[str] = Field(
        default=None,
        description="URL of source audio recording to transcribe and cover",
    )
    audio_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded source audio recording to transcribe and cover",
    )
    score_abc: Optional[str] = Field(
        default=None,
        description="Pre-existing ABC melody score notation if already transcribed",
    )
    cot: CoTMode = Field(
        default=CoTMode.MELODY,
        description="Planning mode: 'melody' is recommended for covers as it gives harmonic freedom to the new genre",
    )
    seed: Optional[int] = Field(
        default=831001, description="Random seed for generation"
    )
    cfg_scale: Optional[float] = Field(
        default=1.0, ge=1.0, le=5.0, description="CFG guidance scale"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.MP3, description="Audio output format: mp3, wav, or flac"
    )
    webhook_url: Optional[str] = Field(default=None, description="Optional webhook URL")
    request_id: Optional[str] = Field(
        default=None, description="Optional request tracking ID"
    )


class ExtendSongRequest(BaseModel):
    """Input parameters for extending an existing song."""

    lyrics: str = Field(
        ...,
        description="Complete extended lyrics including earlier and new sections (e.g. [Verse 1] ... [Chorus] ... [Verse 2] ... [Bridge] ... [Outro])",
        example="[Verse 1]\nNeon fades along the lane\n[Chorus]\nLet the day come into view\n[Verse 2]\nMorning bells begin to ring\n[Outro]\nWe will sing beyond the night",
    )
    style: str = Field(
        ...,
        description="Music style prompt for the song",
        example="English, warm piano pop, expressive female voice, acoustic piano, 88 BPM",
    )
    original_score_abc: Optional[str] = Field(
        default=None,
        description="ABC score returned by previous generation to preserve melody and musical themes",
    )
    original_audio_url: Optional[str] = Field(
        default=None,
        description="URL of original audio to transcribe and extend if ABC score was not saved",
    )
    original_audio_base64: Optional[str] = Field(
        default=None,
        description="Base64 encoded original audio to transcribe and extend",
    )
    cot: CoTMode = Field(
        default=CoTMode.FULL,
        description="Planning mode: 'full' (melody + chords) or 'melody'",
    )
    seed: Optional[int] = Field(
        default=831001, description="Random seed for deterministic continuation"
    )
    cfg_scale: Optional[float] = Field(default=1.0, ge=1.0, le=5.0)
    output_format: OutputFormat = Field(default=OutputFormat.MP3)
    webhook_url: Optional[str] = None
    request_id: Optional[str] = None


class TranscribeAudioRequest(BaseModel):
    """Input parameters to transcribe an audio file to ABC notation."""

    audio_url: Optional[str] = Field(
        default=None, description="URL of audio file to transcribe"
    )
    audio_base64: Optional[str] = Field(
        default=None, description="Base64 encoded audio to transcribe"
    )
    melody_only: bool = Field(
        default=True,
        description="True for vocal/lead melody only (ideal for covers); False for melody and chord accompaniment",
    )


class TranscribeAudioResponse(BaseModel):
    """Response containing transcribed ABC score."""

    success: bool
    score_abc: str
    warnings: list[str] = []


class SongGenerationResponse(BaseModel):
    """Structured response for music generation."""

    success: bool
    message: str
    request_id: Optional[str] = None
    audio_base64: Optional[str] = None
    output_format: str
    sample_rate: int = 48000
    duration_seconds: Optional[float] = None
    score_abc: Optional[str] = None
    timing: Optional[dict] = None
    truncated: Optional[dict] = None


class AsyncAcceptedResponse(BaseModel):
    """Immediate response when webhook_url is specified."""

    success: bool
    task_id: str
    request_id: Optional[str] = None
    message: str


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def fetch_audio_bytes(
    audio_url: Optional[str] = None, audio_base64: Optional[str] = None
) -> bytes:
    """Download audio from URL or decode base64 string."""
    if audio_base64:
        return base64.b64decode(audio_base64)
    if audio_url:
        import requests

        response = requests.get(audio_url, timeout=120)
        response.raise_for_status()
        return response.content
    raise ValueError("Either audio_url or audio_base64 must be provided.")


def convert_audio(
    input_path: Path, output_path: Path, target_format: OutputFormat
) -> None:
    """Transcode generated audio to requested format using FFmpeg."""
    if target_format == OutputFormat.FLAC:
        cmd = ["ffmpeg", "-y", "-i", str(input_path), "-c:a", "flac", str(output_path)]
    elif target_format == OutputFormat.WAV:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-c:a",
            "pcm_s24le",
            str(output_path),
        ]
    elif target_format == OutputFormat.MP3:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "320k",
            str(output_path),
        ]
    else:
        raise ValueError(f"Unsupported format: {target_format}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")


def send_webhook(webhook_url: str, payload: dict) -> None:
    """Dispatch webhook notification with generation results."""
    import requests

    try:
        response = requests.post(webhook_url, json=payload, timeout=60)
        print(
            f"Webhook notification delivered to {webhook_url} (HTTP {response.status_code})"
        )
    except Exception as e:
        print(f"Failed to deliver webhook to {webhook_url}: {e}")


# ---------------------------------------------------------------------------
# SheetSage2 Audio Transcription Service
# ---------------------------------------------------------------------------


@app.cls(
    image=sheetsage_image,
    gpu="A10G",
    secrets=CONFIGURED_SECRETS,
    volumes={CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME},
    timeout=600,
    scaledown_window=300,
)
class SheetSageService:
    """SheetSage2 audio-to-score transcription service."""

    @modal.enter()
    def load_transcriber(self):
        import torch
        from transformers import AutoModel

        CONTAINER_CACHE_VOLUME.reload()
        print("Loading SheetSage2 audio transcription model...")
        self.model = (
            AutoModel.from_pretrained(
                SHEETSAGE_MODEL_ID,
                trust_remote_code=True,
            )
            .eval()
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )
        CONTAINER_CACHE_VOLUME.commit()
        print("SheetSage2 model ready.")

    @modal.method()
    def transcribe(self, audio_bytes: bytes, melody_only: bool = True) -> dict:
        """Transcribe audio into ABC score notation."""
        CONTAINER_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=CONTAINER_TEMP_DIR, suffix=".wav", delete=False
        ) as f:
            f.write(audio_bytes)
            tmp_path = Path(f.name)

        try:
            print(
                f"Transcribing audio ({len(audio_bytes)} bytes, melody_only={melody_only})..."
            )
            result = self.model.transcribe(str(tmp_path), melody_only=melody_only)
            abc_text = result.get("abc", "")
            warnings = result.get("warnings", [])
            print(
                f"Transcription complete: {len(abc_text)} characters of ABC notation."
            )
            return {
                "abc": abc_text,
                "warnings": warnings,
            }
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# YuE Generation Service Class
# ---------------------------------------------------------------------------


@app.cls(
    gpu="A10G",  # 24GB VRAM NVIDIA A10G GPU
    secrets=CONFIGURED_SECRETS,
    volumes={CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME},
    timeout=600,  # 10 minutes timeout for long full-song generations
    scaledown_window=30,  # Keep container warm for 30 seconds of idle
    min_containers=0,
)
class YuEService:
    """Modal service for music generation, covers, and extensions using YuE2."""

    @modal.enter()
    def initialize_pipeline(self):
        """Initialize YuE2Pipeline on container startup."""
        print(f"Initializing YuE service on {torch.cuda.get_device_name(0)}...")
        CONTAINER_TEMP_DIR.mkdir(parents=True, exist_ok=True)

        CONTAINER_CACHE_VOLUME.reload()

        total_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Detected GPU Total VRAM: {total_vram_gib:.2f} GiB")

        # Load YuE2 Pipeline with 22 GiB budget (leaving 2GiB workspace reserve)
        print("Loading YuE2-3B model and YuE2-Vae decoder...")
        start_time = time.time()
        self.pipeline = YuE2Pipeline.from_pretrained(
            YUE_MODEL_ID,
            vae=VAE_ID,
            device="cuda",
            memory_budget_gib=22.0,
            progress=True,
        )
        print(
            f"YuE2 pipeline loaded successfully in {time.time() - start_time:.2f} seconds."
        )

        CONTAINER_CACHE_VOLUME.commit()

    @modal.method()
    def generate_music(self, request_data: dict) -> dict:
        """Core generation method executed on the A10G GPU container."""
        req = SongGenerationRequest(**request_data)
        request_id = req.request_id or f"yue_{int(time.time() * 1000)}"
        print(
            f"Starting song generation [{request_id}] | Style: {req.style[:60]}... | CoT: {req.cot.value}"
        )

        start_time = time.time()
        temp_dir = Path(tempfile.mkdtemp(dir=CONTAINER_TEMP_DIR))
        raw_flac_path = temp_dir / "generated.flac"
        final_audio_path = temp_dir / f"final.{req.output_format.value}"

        try:
            yue_kwargs = {
                "id": request_id,
                "style": req.style,
                "lyrics": req.lyrics,
                "cot": req.cot.value,
                "seed": req.seed,
            }
            if req.abc:
                yue_kwargs["abc"] = req.abc
            if req.cfg_scale is not None:
                yue_kwargs["cfg_scale"] = req.cfg_scale

            song_result = self.pipeline(**yue_kwargs)

            song_result.save(raw_flac_path)
            duration_seconds = len(song_result.audio) / song_result.sample_rate

            convert_audio(raw_flac_path, final_audio_path, req.output_format)
            audio_bytes = final_audio_path.read_bytes()

            score_abc = song_result.abc
            timing_info = song_result.timing
            truncated_info = song_result.truncated

            total_elapsed = time.time() - start_time
            print(
                f"Completed song [{request_id}] | Duration: {duration_seconds:.2f}s | Elapsed: {total_elapsed:.2f}s"
            )

            result = {
                "success": True,
                "message": "Song generated successfully.",
                "request_id": request_id,
                "audio_bytes": audio_bytes,
                "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
                "output_format": req.output_format.value,
                "sample_rate": song_result.sample_rate,
                "duration_seconds": duration_seconds,
                "score_abc": score_abc,
                "timing": timing_info,
                "truncated": truncated_info,
            }

            if req.webhook_url:
                webhook_payload = {
                    k: v for k, v in result.items() if k != "audio_bytes"
                }
                send_webhook(req.webhook_url, webhook_payload)

            return result

        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            print(f"Error [{request_id}]: {error_msg}")
            if req.webhook_url:
                send_webhook(
                    req.webhook_url,
                    {
                        "success": False,
                        "request_id": request_id,
                        "message": error_msg,
                    },
                )
            raise RuntimeError(error_msg)
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    @modal.method()
    def generate_cover(self, request_data: dict) -> dict:
        """Handle zero-shot cover generation via transcription and score conditioning."""
        req = CoverSongRequest(**request_data)
        request_id = req.request_id or f"cover_{int(time.time() * 1000)}"

        # 1. Resolve ABC score
        abc_content = req.score_abc
        if not abc_content:
            if not req.audio_url and not req.audio_base64:
                raise ValueError(
                    "Cover requires either score_abc, audio_url, or audio_base64."
                )
            print(f"[{request_id}] Transcribing reference audio for cover...")
            audio_bytes = fetch_audio_bytes(req.audio_url, req.audio_base64)
            transcriber = SheetSageService()
            transcribe_res = transcriber.transcribe.remote(
                audio_bytes, melody_only=(req.cot == CoTMode.MELODY)
            )
            abc_content = transcribe_res["abc"]
            print(f"[{request_id}] Extracted ABC melody for cover.")

        # 2. Delegate to generate_music with score
        gen_req = {
            "style": req.style,
            "lyrics": req.lyrics,
            "cot": req.cot.value,
            "seed": req.seed,
            "abc": abc_content,
            "cfg_scale": req.cfg_scale,
            "output_format": req.output_format.value,
            "webhook_url": req.webhook_url,
            "request_id": request_id,
        }
        return self.generate_music.local(gen_req)

    @modal.method()
    def extend_song(self, request_data: dict) -> dict:
        """Extend an existing song with new lyrical sections preserving musical motifs."""
        req = ExtendSongRequest(**request_data)
        request_id = req.request_id or f"extend_{int(time.time() * 1000)}"

        # 1. Resolve anchor composition
        abc_content = req.original_score_abc
        if not abc_content and (req.original_audio_url or req.original_audio_base64):
            print(f"[{request_id}] Transcribing original audio to anchor extension...")
            audio_bytes = fetch_audio_bytes(
                req.original_audio_url, req.original_audio_base64
            )
            transcriber = SheetSageService()
            transcribe_res = transcriber.transcribe.remote(
                audio_bytes, melody_only=False
            )
            abc_content = transcribe_res["abc"]

        # 2. Synthesize extended piece with anchor score and extended lyrics
        gen_req = {
            "style": req.style,
            "lyrics": req.lyrics,
            "cot": req.cot.value,
            "seed": req.seed,
            "abc": abc_content,
            "cfg_scale": req.cfg_scale,
            "output_format": req.output_format.value,
            "webhook_url": req.webhook_url,
            "request_id": request_id,
        }
        return self.generate_music.local(gen_req)

    # -----------------------------------------------------------------------
    # FastAPI HTTP Endpoints
    # -----------------------------------------------------------------------

    # --- Standard Generation ---

    @modal.fastapi_endpoint(method="POST")
    def generate_audio(
        self,
        request: SongGenerationRequest,
        _auth: bool = Depends(verify_api_key),
    ):
        """Direct audio streaming endpoint for standard generation."""
        try:
            result = self.generate_music.local(request.model_dump())
            return Response(
                content=result["audio_bytes"],
                media_type=request.output_format.media_type,
                headers={
                    "Content-Disposition": f'inline; filename="{result["request_id"]}.{request.output_format.value}"',
                    "X-Duration-Seconds": str(result.get("duration_seconds", "")),
                    "X-Sample-Rate": str(result.get("sample_rate", 48000)),
                    "X-Request-ID": result.get("request_id", ""),
                },
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @modal.fastapi_endpoint(method="POST")
    def generate(
        self,
        request: SongGenerationRequest,
        _auth: bool = Depends(verify_api_key),
    ) -> SongGenerationResponse:
        """JSON generation endpoint returning base64 audio and ABC score."""
        try:
            result = self.generate_music.local(request.model_dump())
            return SongGenerationResponse(
                success=True,
                message=result["message"],
                request_id=result["request_id"],
                audio_base64=result["audio_base64"],
                output_format=result["output_format"],
                sample_rate=result["sample_rate"],
                duration_seconds=result["duration_seconds"],
                score_abc=result["score_abc"],
                timing=result["timing"],
                truncated=result["truncated"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @modal.fastapi_endpoint(method="POST")
    def generate_async(
        self,
        request: SongGenerationRequest,
        _auth: bool = Depends(verify_api_key),
    ) -> AsyncAcceptedResponse:
        """Asynchronous generation endpoint with webhook callback."""
        if not request.webhook_url:
            raise HTTPException(
                status_code=400,
                detail="webhook_url is required for asynchronous generation.",
            )

        task = self.generate_music.spawn(request.model_dump())
        return AsyncAcceptedResponse(
            success=True,
            task_id=task.object_id,
            request_id=request.request_id,
            message="Generation task enqueued. Results will be delivered to webhook_url upon completion.",
        )

    # --- Zero-Shot Covers ---

    @modal.fastapi_endpoint(method="POST")
    def cover(
        self,
        request: CoverSongRequest,
        _auth: bool = Depends(verify_api_key),
    ) -> SongGenerationResponse:
        """Zero-shot cover endpoint returning base64 audio and extracted ABC score."""
        try:
            result = self.generate_cover.local(request.model_dump())
            return SongGenerationResponse(
                success=True,
                message=result["message"],
                request_id=result["request_id"],
                audio_base64=result["audio_base64"],
                output_format=result["output_format"],
                sample_rate=result["sample_rate"],
                duration_seconds=result["duration_seconds"],
                score_abc=result["score_abc"],
                timing=result["timing"],
                truncated=result["truncated"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @modal.fastapi_endpoint(method="POST")
    def cover_audio(
        self,
        request: CoverSongRequest,
        _auth: bool = Depends(verify_api_key),
    ):
        """Zero-shot cover streaming endpoint returning the generated audio file."""
        try:
            result = self.generate_cover.local(request.model_dump())
            return Response(
                content=result["audio_bytes"],
                media_type=request.output_format.media_type,
                headers={
                    "Content-Disposition": f'inline; filename="{result["request_id"]}.{request.output_format.value}"',
                    "X-Duration-Seconds": str(result.get("duration_seconds", "")),
                    "X-Sample-Rate": str(result.get("sample_rate", 48000)),
                    "X-Request-ID": result.get("request_id", ""),
                },
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # --- Song Extension ---

    @modal.fastapi_endpoint(method="POST")
    def extend(
        self,
        request: ExtendSongRequest,
        _auth: bool = Depends(verify_api_key),
    ) -> SongGenerationResponse:
        """Song extension endpoint returning base64 audio and extended score."""
        try:
            result = self.extend_song.local(request.model_dump())
            return SongGenerationResponse(
                success=True,
                message=result["message"],
                request_id=result["request_id"],
                audio_base64=result["audio_base64"],
                output_format=result["output_format"],
                sample_rate=result["sample_rate"],
                duration_seconds=result["duration_seconds"],
                score_abc=result["score_abc"],
                timing=result["timing"],
                truncated=result["truncated"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @modal.fastapi_endpoint(method="POST")
    def extend_audio(
        self,
        request: ExtendSongRequest,
        _auth: bool = Depends(verify_api_key),
    ):
        """Song extension streaming endpoint returning the extended audio file."""
        try:
            result = self.extend_song.local(request.model_dump())
            return Response(
                content=result["audio_bytes"],
                media_type=request.output_format.media_type,
                headers={
                    "Content-Disposition": f'inline; filename="{result["request_id"]}.{request.output_format.value}"',
                    "X-Duration-Seconds": str(result.get("duration_seconds", "")),
                    "X-Sample-Rate": str(result.get("sample_rate", 48000)),
                    "X-Request-ID": result.get("request_id", ""),
                },
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Standalone Audio-to-ABC Transcription Endpoint
# ---------------------------------------------------------------------------


@app.function(image=sheetsage_image, secrets=CONFIGURED_SECRETS)
@modal.fastapi_endpoint(method="POST")
def transcribe(
    request: TranscribeAudioRequest,
    _auth: bool = Depends(verify_api_key),
) -> TranscribeAudioResponse:
    """Standalone endpoint to transcribe an audio file into ABC notation via SheetSage2."""
    try:
        audio_bytes = fetch_audio_bytes(request.audio_url, request.audio_base64)
        transcriber = SheetSageService()
        res = transcriber.transcribe.remote(
            audio_bytes, melody_only=request.melody_only
        )
        return TranscribeAudioResponse(
            success=True,
            score_abc=res["abc"],
            warnings=res.get("warnings", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Health & Caching Utilities
# ---------------------------------------------------------------------------


@app.function(image=health_image, secrets=CONFIGURED_SECRETS)
@modal.fastapi_endpoint(method="GET")
def health() -> dict:
    """Health check endpoint exposing system information and capabilities."""
    return {
        "status": "healthy",
        "auth_enabled": bool(os.environ.get("YUE_API_KEY")),
        "service": "yue-music-generator",
        "models": {
            "generation": YUE_MODEL_ID,
            "decoder": VAE_ID,
            "transcription": SHEETSAGE_MODEL_ID,
        },
        "capabilities": [
            "text-to-full-song",
            "zero-shot-cover (audio or ABC)",
            "song-extension",
            "audio-to-abc-transcription",
        ],
        "gpu_target": "NVIDIA A10G (24GB VRAM)",
        "output_sample_rate": 48000,
        "supported_formats": [f.value for f in OutputFormat],
        "supported_cot_modes": [m.value for m in CoTMode],
    }


@app.function(
    volumes={CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME},
    secrets=CONFIGURED_SECRETS,
    timeout=1800,
)
def download_models():
    """CLI utility to pre-download model weights into the Modal volume."""
    from huggingface_hub import snapshot_download

    print(f"Pre-downloading {YUE_MODEL_ID} to volume cache...")
    snapshot_download(YUE_MODEL_ID, local_files_only=False)

    print(f"Pre-downloading {VAE_ID} to volume cache...")
    snapshot_download(VAE_ID, local_files_only=False)

    print(f"Pre-downloading {SHEETSAGE_MODEL_ID} to volume cache...")
    snapshot_download(SHEETSAGE_MODEL_ID, local_files_only=False)

    CONTAINER_CACHE_VOLUME.commit()
    print("All models successfully cached in Modal Volume!")


# ---------------------------------------------------------------------------
# Local Entrypoint for CLI Testing
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    mode: str = "generate",  # "generate", "cover", "extend", "transcribe"
    style: str = "English, warm piano pop, expressive female voice, acoustic piano, 88 BPM",
    lyrics: str = "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n[Chorus]\nLet the day come into view\nEvery road begins with you",
    cot: str = "full",
    seed: int = 831001,
    format: str = "mp3",
    output_path: str = "output_song.mp3",
    audio_url: Optional[str] = None,
    score_abc: Optional[str] = None,
):
    """CLI entrypoint to test generate, cover, extend, or transcribe workflows."""
    print("=" * 70)
    print(f" YuE Music Engine on Modal A10G — Mode: [{mode.upper()}]")
    print("=" * 70)

    service = YuEService()
    out_file = Path(output_path)

    if mode == "cover":
        print(f"Target Style: {style}")
        print(f"Lyrics:\n{lyrics}")
        print(f"Audio URL:    {audio_url}")
        print(
            f"Direct ABC:   {'Provided' if score_abc else 'None (auto-transcribing)'}"
        )
        req = {
            "style": style,
            "lyrics": lyrics,
            "audio_url": audio_url,
            "score_abc": score_abc,
            "cot": cot if cot in ["melody", "full"] else "melody",
            "seed": seed,
            "output_format": format,
        }
        print("Submitting cover generation request to Modal...")
        result = service.generate_cover.remote(req)

    elif mode == "extend":
        print(f"Style:       {style}")
        print(f"Full Lyrics:\n{lyrics}")
        req = {
            "style": style,
            "lyrics": lyrics,
            "original_score_abc": score_abc,
            "original_audio_url": audio_url,
            "cot": cot,
            "seed": seed,
            "output_format": format,
        }
        print("Submitting extension request to Modal...")
        result = service.extend_song.remote(req)

    elif mode == "transcribe":
        if not audio_url:
            raise ValueError("mode=transcribe requires --audio-url <URL>")
        print(f"Transcribing: {audio_url}...")
        transcriber = SheetSageService()
        import requests

        audio_data = requests.get(audio_url, timeout=120).content
        res = transcriber.transcribe.remote(audio_data, melody_only=True)
        print("\n--- Transcribed ABC Score ---")
        print(res["abc"])
        if out_file.suffix == ".abc":
            out_file.write_text(res["abc"])
            print(f"Saved score to: {out_file.resolve()}")
        return

    else:  # default: "generate"
        print(f"Style:   {style}")
        print(f"Lyrics:\n{lyrics}")
        print(f"CoT:     {cot}")
        print(f"Seed:    {seed}")
        print(f"Format:  {format}")
        req = {
            "style": style,
            "lyrics": lyrics,
            "cot": cot,
            "seed": seed,
            "abc": score_abc,
            "output_format": format,
        }
        print("Submitting generation request to Modal A10G GPU...")
        result = service.generate_music.remote(req)

    out_file.write_bytes(result["audio_bytes"])
    print(f"\nSong saved to: {out_file.resolve()}")
    print(f"Audio Duration: {result['duration_seconds']:.2f} seconds")
    print(f"Sample Rate:    {result['sample_rate']} Hz")

    if result.get("score_abc"):
        print("\n--- Associated ABC Notation Score ---")
        print(result["score_abc"])
    print("=" * 70)
