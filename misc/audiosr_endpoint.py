# ---
# lambda-test: false
# ---

# # Audio Upscaling Endpoint with AudioSR on Modal
#
# This example demonstrates how to run a high-performance audio upscaling endpoint
# on Modal using AudioSR. The endpoint accepts an audio file URL, applies a low-pass
# filter to prepare the audio, and then upscales it to 48kHz using AudioSR.
#
# The workflow:
# 1. Download audio from URL (CPU)
# 2. Apply low-pass filter using ffmpeg to create optimal input for AudioSR (CPU)
# 3. Upscale the filtered audio using AudioSR (GPU - T4)
# 4. Return the upscaled audio file

from __future__ import annotations

from pathlib import Path
from io import BytesIO
import tempfile
import os

import modal

# Container mount directories
CONTAINER_CACHE_DIR = Path("/cache")
CONTAINER_TEMP_DIR = Path("/tmp/audiosr_work")

# Modal volume for caching model weights across container restarts
CONTAINER_CACHE_VOLUME = modal.Volume.from_name("audiosr_endpoint", create_if_missing=True)

# ## Building the container image

# We use a CUDA base image with necessary dependencies for AudioSR
cuda_version = "12.1.1"
flavor = "devel"
operating_system = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_system}"

nvidia_cuda_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.9"
).entrypoint([])

# Install all dependencies needed for AudioSR
audiosr_image = (
    nvidia_cuda_image
    .apt_install(
        "git",
        "ffmpeg",  # Required for audio processing and low-pass filtering
        "libsndfile1",  # Required for audio file handling
        "wget",
        "curl",
    )
    .pip_install(
        "torch==2.1.0",
        "torchaudio==2.1.0",
        "numpy<2",  # AudioSR requires numpy < 2
        "scipy",
        "librosa",
        "soundfile",
        "pydub",
        "requests",
        "fastapi[standard]==0.115.12",
        "pydantic==2.11.4",
        "huggingface-hub[hf_transfer]==0.33.1",
    )
    .run_commands(
        "pip install audiosr==0.0.7",
    )
    .env({
        "HF_HUB_CACHE": str(CONTAINER_CACHE_DIR / ".hf_hub_cache"),
    })
)

# ## Creating the Modal app

app = modal.App("audiosr_endpoint", image=audiosr_image)

with audiosr_image.imports():
    import time
    import subprocess
    import requests
    import base64
    from enum import Enum
    from typing import Optional
    from pydantic import BaseModel, Field
    from fastapi import Response, HTTPException
    import audiosr
    import torch
    import soundfile as sf

    # Supported output formats for upscaled audio
    class OutputFormat(Enum):
        WAV = "wav"
        FLAC = "flac"
        MP3 = "mp3"

        @property
        def mime_type(self):
            return {
                OutputFormat.WAV: "audio/wav",
                OutputFormat.FLAC: "audio/flac",
                OutputFormat.MP3: "audio/mpeg",
            }[self]

    # ### Defining request/response models

    class AudioUpscaleRequest(BaseModel):
        """Request model for audio upscaling."""
        audio_url: str  # URL to the input audio file (MP3, WAV, etc.)
        lowpass_frequency: Optional[int] = Field(
            default=12000,
            ge=4000,
            le=20000,
            description="Low-pass filter cutoff frequency in Hz"
        )
        lowpass_poles: Optional[int] = Field(
            default=2,
            ge=1,
            le=8,
            description="Number of poles for the low-pass filter"
        )

        ddim_steps: Optional[int] = Field(
            default=50,
            ge=10,
            le=200,
            description="Number of DDIM sampling steps (higher = better quality but slower)"
        )
        guidance_scale: Optional[float] = Field(
            default=3.5,
            ge=1.0,
            le=10.0,
            description="Guidance scale for the diffusion model"
        )
        seed: Optional[int] = Field(
            default=None,
            description="Random seed for reproducibility"
        )
        output_format: Optional[OutputFormat] = Field(
            default=OutputFormat.WAV,
            description="Output audio format"
        )

    class AudioUpscaleResponse(BaseModel):
        """Response model for audio upscaling."""
        success: bool
        message: str
        audio_base64: Optional[str] = None
        original_sample_rate: Optional[int] = None
        output_sample_rate: Optional[int] = None
        processing_time_seconds: Optional[float] = None


# ## Common configuration for AudioSR service

common_config = dict(
    volumes={
        CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME,
    },
    min_containers=0,
    buffer_containers=0,
    timeout=600,  # 10 minutes timeout for long audio files
)


# ## AudioSR Service Class

@app.cls(
    scaledown_window=60,  # Keep warm for 60 seconds for follow-up requests
    gpu="T4",  # T4 is sufficient for AudioSR
    **common_config,
)
class AudioSRService:
    """Audio super-resolution service using AudioSR."""

    @modal.enter()
    def load_model(self):
        """Load AudioSR model on container start."""
        print("Loading AudioSR basic model...")
        
        # Ensure temp directory exists
        CONTAINER_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load the basic model
        self.audiosr_model = audiosr.build_model(model_name="basic")
        print("AudioSR basic model loaded successfully.")

    def _download_audio(self, url: str, output_path: Path) -> None:
        """Download audio file from URL."""
        print(f"Downloading audio from: {url}")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded audio to: {output_path}")

    def _apply_lowpass_filter(
        self, 
        input_path: Path, 
        output_path: Path, 
        frequency: int = 12000,
        poles: int = 2
    ) -> None:
        """Apply low-pass filter using ffmpeg.
        
        This prepares the audio for AudioSR by creating a clean cutoff pattern
        that matches what AudioSR was trained on.
        
        Command: ffmpeg -i input.mp3 -af "lowpass=f=12000:p=2" output.wav
        """
        print(f"Applying low-pass filter (f={frequency}Hz, poles={poles})...")
        
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-i", str(input_path),
            "-af", f"lowpass=f={frequency}:p={poles}",
            "-ar", "24000",  # Ensure consistent sample rate for AudioSR input
            "-ac", "2",  # Stereo
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg low-pass filter failed: {result.stderr}")
        
        print(f"Low-pass filtered audio saved to: {output_path}")

    def _upscale_audio(
        self,
        input_path: Path,
        output_path: Path,
        ddim_steps: int = 50,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None
    ) -> dict:
        """Upscale audio using AudioSR with chunking for long files.
        
        AudioSR recommends max 5.12 seconds per chunk for best quality.
        For longer files, we split into chunks, process each, and concatenate.
        """
        import numpy as np
        import soundfile as sf
        
        # Constants
        CHUNK_DURATION = 5.12  # seconds - recommended by AudioSR
        OVERLAP_DURATION = 0.5  # seconds - overlap for crossfade
        OUTPUT_SR = 48000  # AudioSR output sample rate
        
        # Get original audio info
        original_info = sf.info(str(input_path))
        original_sr = original_info.samplerate
        duration = original_info.duration
        
        print(f"Audio duration: {duration:.2f}s, sample rate: {original_sr}Hz")
        
        # If audio is short enough, process directly
        if duration <= CHUNK_DURATION + 1.0:  # Add small buffer
            print(f"Processing audio directly (duration <= {CHUNK_DURATION}s)...")
            waveform = audiosr.super_resolution(
                self.audiosr_model,
                str(input_path),
                seed=seed if seed is not None else 42,
                guidance_scale=guidance_scale,
                ddim_steps=ddim_steps,
                latent_t_per_second=12.8
            )
            sf.write(str(output_path), waveform[0].T, OUTPUT_SR)
            print(f"Upscaled audio saved to: {output_path}")
            return {
                "original_sample_rate": original_sr,
                "output_sample_rate": OUTPUT_SR
            }
        
        # For long audio, process in chunks
        print(f"Audio is {duration:.2f}s, processing in {CHUNK_DURATION}s chunks...")
        
        # Load entire audio for chunking
        audio_data, sr = sf.read(str(input_path))
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(-1, 1)  # Ensure 2D
        
        # Calculate chunk parameters in samples
        chunk_samples = int(CHUNK_DURATION * sr)
        overlap_samples = int(OVERLAP_DURATION * sr)
        step_samples = chunk_samples - overlap_samples
        
        # Calculate output overlap in samples (at 48kHz)
        output_overlap_samples = int(OVERLAP_DURATION * OUTPUT_SR)
        
        # Process chunks
        chunks_upscaled = []
        work_dir = input_path.parent
        total_chunks = (len(audio_data) - overlap_samples) // step_samples + 1
        
        for i, start in enumerate(range(0, len(audio_data) - overlap_samples, step_samples)):
            end = min(start + chunk_samples, len(audio_data))
            chunk = audio_data[start:end]
            
            print(f"Processing chunk {i+1}/{total_chunks} ({start/sr:.2f}s - {end/sr:.2f}s)...")
            
            # Save chunk to temporary file
            chunk_path = work_dir / f"chunk_{i}.wav"
            sf.write(str(chunk_path), chunk, sr)
            
            # Process chunk with AudioSR
            try:
                waveform = audiosr.super_resolution(
                    self.audiosr_model,
                    str(chunk_path),
                    seed=(seed + i) if seed is not None else (42 + i),
                    guidance_scale=guidance_scale,
                    ddim_steps=ddim_steps,
                    latent_t_per_second=12.8
                )
                chunks_upscaled.append(waveform[0].T)  # Shape: (samples, channels)
            finally:
                # Clean up chunk file
                if chunk_path.exists():
                    chunk_path.unlink()
            
            print(f"Chunk {i+1} upscaled successfully")
        
        # Concatenate chunks with crossfade
        print(f"Concatenating {len(chunks_upscaled)} upscaled chunks...")
        
        if len(chunks_upscaled) == 1:
            final_audio = chunks_upscaled[0]
        else:
            # Create crossfade window
            fade_in = np.linspace(0, 1, output_overlap_samples).reshape(-1, 1)
            fade_out = np.linspace(1, 0, output_overlap_samples).reshape(-1, 1)
            
            # Start with first chunk
            final_audio = chunks_upscaled[0]
            
            for i in range(1, len(chunks_upscaled)):
                current_chunk = chunks_upscaled[i]
                
                # Apply crossfade
                # Fade out the end of the previous audio
                overlap_start = len(final_audio) - output_overlap_samples
                final_audio[overlap_start:] *= fade_out
                
                # Fade in the beginning of the current chunk
                current_chunk[:output_overlap_samples] *= fade_in
                
                # Combine: keep non-overlapping part + crossfaded overlap + rest of current
                final_audio = np.concatenate([
                    final_audio[:overlap_start],
                    final_audio[overlap_start:] + current_chunk[:output_overlap_samples],
                    current_chunk[output_overlap_samples:]
                ])
        
        # Save final output
        sf.write(str(output_path), final_audio, OUTPUT_SR)
        print(f"Upscaled audio saved to: {output_path} (duration: {len(final_audio)/OUTPUT_SR:.2f}s)")
        
        return {
            "original_sample_rate": original_sr,
            "output_sample_rate": OUTPUT_SR
        }

    def _convert_to_format(
        self,
        input_path: Path,
        output_path: Path,
        output_format: OutputFormat
    ) -> None:
        """Convert audio to the desired output format using ffmpeg."""
        if output_format == OutputFormat.WAV:
            # Already in WAV format, just copy
            if input_path != output_path:
                import shutil
                shutil.copy(input_path, output_path)
            return
        
        print(f"Converting to {output_format.value}...")
        
        cmd = ["ffmpeg", "-y", "-i", str(input_path)]
        
        if output_format == OutputFormat.MP3:
            cmd.extend(["-codec:a", "libmp3lame", "-qscale:a", "2"])
        elif output_format == OutputFormat.FLAC:
            cmd.extend(["-codec:a", "flac"])
        
        cmd.append(str(output_path))
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")

    @modal.fastapi_endpoint(method="POST")
    def upscale(self, request: AudioUpscaleRequest) -> Response:
        """Upscale audio from URL using AudioSR.
        
        Workflow:
        1. Download audio from URL
        2. Apply low-pass filter to create optimal input for AudioSR
        3. Upscale using AudioSR
        4. Return upscaled audio as base64 or binary
        """
        start_time = time.perf_counter()
        
        # Create unique work directory for this request
        work_dir = CONTAINER_TEMP_DIR / f"request_{int(time.time() * 1000)}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Download audio
            input_ext = Path(request.audio_url.split("?")[0]).suffix or ".mp3"
            input_path = work_dir / f"input{input_ext}"
            self._download_audio(request.audio_url, input_path)
            
            # Step 2: Apply low-pass filter
            filtered_path = work_dir / "filtered.wav"
            self._apply_lowpass_filter(
                input_path,
                filtered_path,
                frequency=request.lowpass_frequency,
                poles=request.lowpass_poles
            )
            
            # Step 3: Upscale with AudioSR
            upscaled_path = work_dir / "upscaled.wav"
            upscale_info = self._upscale_audio(
                filtered_path,
                upscaled_path,
                ddim_steps=request.ddim_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed
            )
            
            # Step 4: Convert to requested format
            output_ext = f".{request.output_format.value}"
            output_path = work_dir / f"output{output_ext}"
            self._convert_to_format(upscaled_path, output_path, request.output_format)
            
            # Read output file
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            
            processing_time = time.perf_counter() - start_time
            print(f"Total processing time: {processing_time:.2f}s")
            
            # Return as binary response
            return Response(
                content=audio_bytes,
                media_type=request.output_format.mime_type,
                headers={
                    "X-Original-Sample-Rate": str(upscale_info["original_sample_rate"]),
                    "X-Output-Sample-Rate": str(upscale_info["output_sample_rate"]),
                    "X-Processing-Time-Seconds": f"{processing_time:.2f}",
                }
            )
            
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        finally:
            # Cleanup work directory
            import shutil
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)

    @modal.fastapi_endpoint(method="POST")
    def upscale_json(self, request: AudioUpscaleRequest) -> AudioUpscaleResponse:
        """Upscale audio and return as JSON with base64-encoded audio.
        
        Same as /upscale but returns JSON response with metadata.
        """
        start_time = time.perf_counter()
        
        # Create unique work directory for this request
        work_dir = CONTAINER_TEMP_DIR / f"request_{int(time.time() * 1000)}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Download audio
            input_ext = Path(request.audio_url.split("?")[0]).suffix or ".mp3"
            input_path = work_dir / f"input{input_ext}"
            self._download_audio(request.audio_url, input_path)
            
            # Step 2: Apply low-pass filter
            filtered_path = work_dir / "filtered.wav"
            self._apply_lowpass_filter(
                input_path,
                filtered_path,
                frequency=request.lowpass_frequency,
                poles=request.lowpass_poles
            )
            
            # Step 3: Upscale with AudioSR
            upscaled_path = work_dir / "upscaled.wav"
            upscale_info = self._upscale_audio(
                filtered_path,
                upscaled_path,
                ddim_steps=request.ddim_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed
            )
            
            # Step 4: Convert to requested format
            output_ext = f".{request.output_format.value}"
            output_path = work_dir / f"output{output_ext}"
            self._convert_to_format(upscaled_path, output_path, request.output_format)
            
            # Read and encode output file
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            processing_time = time.perf_counter() - start_time
            print(f"Total processing time: {processing_time:.2f}s")
            
            return AudioUpscaleResponse(
                success=True,
                message="Audio upscaled successfully",
                audio_base64=audio_base64,
                original_sample_rate=upscale_info["original_sample_rate"],
                output_sample_rate=upscale_info["output_sample_rate"],
                processing_time_seconds=round(processing_time, 2)
            )
            
        except requests.RequestException as e:
            return AudioUpscaleResponse(
                success=False,
                message=f"Failed to download audio: {str(e)}"
            )
        except Exception as e:
            return AudioUpscaleResponse(
                success=False,
                message=f"Processing failed: {str(e)}"
            )
        finally:
            # Cleanup work directory
            import shutil
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)


# ## Local testing

@app.local_entrypoint()
def main(audio_url: str = "https://example.com/test.mp3"):
    """Test the AudioSR endpoint locally.
    
    Usage:
        modal run audiosr_endpoint.py --audio-url "https://your-audio-url.mp3"
    """
    service = AudioSRService()
    
    request = AudioUpscaleRequest(
        audio_url=audio_url,
        lowpass_frequency=12000,
        lowpass_poles=2,
        ddim_steps=50,
        guidance_scale=3.5,
        seed=42,
        output_format=OutputFormat.WAV
    )
    
    print(f"Upscaling audio from: {audio_url}")
    response = service.upscale.remote(request)
    
    # Save the upscaled audio
    output_path = Path("upscaled_audio.wav")
    output_path.write_bytes(response.content)
    print(f"Upscaled audio saved to {output_path}")


# ## Health check endpoint

@app.function(image=audiosr_image)
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "audiosr_endpoint"}
