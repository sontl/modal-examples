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
    from fastapi.responses import StreamingResponse
    import asyncio
    import json
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
        skip_lowpass: Optional[bool] = Field(
            default=False,
            description="Skip low-pass filtering (use when input already has clean cutoff pattern)"
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
    enable_memory_snapshot=True,  # Snapshot model loaded on CPU for faster cold starts
)


# ## AudioSR Service Class

@app.cls(
    scaledown_window=2,  # Quick scaledown for cost savings
    gpu="L4",  # L4 for faster processing (upgrade from T4)
    **common_config,
)
class AudioSRService:
    """Audio super-resolution service using AudioSR."""

    @modal.enter(snap=True)
    def load_model_to_cpu(self):
        """Load AudioSR model to CPU during snapshot phase.
        
        During snapshotting, no GPU is available. We load the model to CPU,
        and the model weights will be included in the memory snapshot.
        """
        print("Loading AudioSR model to CPU for snapshotting...")
        
        # Ensure temp directory exists
        CONTAINER_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load model to CPU - this will be snapshotted
        self.audiosr_model = audiosr.build_model(model_name="basic", device="cpu")
        print("AudioSR model loaded to CPU. Ready for snapshot.")

    @modal.enter(snap=False)
    def move_model_to_gpu(self):
        """Move the pre-loaded model to GPU after snapshot restore.
        
        After restoring from snapshot, the GPU is available. We just move
        the already-loaded model to CUDA - no need to rebuild.
        """
        import torch
        print("Moving AudioSR model to CUDA...")
        
        # Move the model to GPU (it's already loaded from snapshot)
        self.audiosr_model = self.audiosr_model.to("cuda")
        self.audiosr_model.device = torch.device("cuda")
        
        print("AudioSR model moved to CUDA. Ready for inference.")

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
        
        AudioSR processes mono audio internally. For stereo files, we process
        each channel separately and merge them back to preserve stereo imaging.
        
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
        num_channels = original_info.channels
        
        print(f"Audio duration: {duration:.2f}s, sample rate: {original_sr}Hz, channels: {num_channels}")
        
        # Load audio data
        audio_data, sr = sf.read(str(input_path))
        
        # Handle mono vs stereo
        if len(audio_data.shape) == 1:
            # Mono audio - process directly
            is_stereo = False
            channels_to_process = [audio_data]
            print("Processing mono audio...")
        else:
            # Stereo or multi-channel - process each channel separately
            is_stereo = True
            num_channels = audio_data.shape[1]
            channels_to_process = [audio_data[:, ch] for ch in range(num_channels)]
            print(f"Processing {num_channels}-channel audio (each channel separately)...")
        
        work_dir = input_path.parent
        upscaled_channels = []
        
        for ch_idx, channel_data in enumerate(channels_to_process):
            if is_stereo:
                print(f"\n=== Processing channel {ch_idx + 1}/{num_channels} ===")
            
            # Process this channel
            upscaled_channel = self._upscale_mono_channel(
                channel_data=channel_data,
                sr=sr,
                work_dir=work_dir,
                channel_idx=ch_idx,
                duration=duration,
                chunk_duration=CHUNK_DURATION,
                overlap_duration=OVERLAP_DURATION,
                output_sr=OUTPUT_SR,
                ddim_steps=ddim_steps,
                guidance_scale=guidance_scale,
                seed=seed
            )
            upscaled_channels.append(upscaled_channel)
        
        # Merge channels back
        if is_stereo:
            # Stack channels: ensure same length
            min_len = min(len(ch) for ch in upscaled_channels)
            upscaled_channels = [ch[:min_len] for ch in upscaled_channels]
            final_audio = np.column_stack(upscaled_channels)
            print(f"\nMerged {num_channels} channels into stereo output")
        else:
            final_audio = upscaled_channels[0]
        
        # Save final output
        sf.write(str(output_path), final_audio, OUTPUT_SR)
        print(f"Upscaled audio saved to: {output_path} (duration: {len(final_audio)/OUTPUT_SR:.2f}s)")
        
        return {
            "original_sample_rate": original_sr,
            "output_sample_rate": OUTPUT_SR
        }

    def _upscale_mono_channel(
        self,
        channel_data: "np.ndarray",
        sr: int,
        work_dir: Path,
        channel_idx: int,
        duration: float,
        chunk_duration: float,
        overlap_duration: float,
        output_sr: int,
        ddim_steps: int,
        guidance_scale: float,
        seed: Optional[int]
    ) -> "np.ndarray":
        """Upscale a single mono channel with chunking support."""
        import numpy as np
        import soundfile as sf
        
        # If audio is short enough, process directly
        if duration <= chunk_duration + 1.0:
            print(f"Channel {channel_idx + 1}: Processing directly (duration <= {chunk_duration}s)...")
            
            # Save channel to temp file
            temp_path = work_dir / f"channel_{channel_idx}_temp.wav"
            sf.write(str(temp_path), channel_data, sr)
            
            try:
                waveform = audiosr.super_resolution(
                    self.audiosr_model,
                    str(temp_path),
                    seed=(seed + channel_idx * 1000) if seed is not None else (42 + channel_idx * 1000),
                    guidance_scale=guidance_scale,
                    ddim_steps=ddim_steps,
                    latent_t_per_second=12.8
                )
                # AudioSR returns shape (1, channels, samples) - we want mono so take first channel
                return waveform[0, 0, :]  # Shape: (samples,)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        
        # For long audio, process in chunks
        print(f"Channel {channel_idx + 1}: Processing in {chunk_duration}s chunks...")
        
        # Calculate chunk parameters in samples
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap_duration * sr)
        step_samples = chunk_samples - overlap_samples
        
        # Calculate output overlap in samples (at output sample rate)
        output_overlap_samples = int(overlap_duration * output_sr)
        
        # Process chunks
        chunks_upscaled = []
        total_chunks = (len(channel_data) - overlap_samples) // step_samples + 1
        
        for i, start in enumerate(range(0, len(channel_data) - overlap_samples, step_samples)):
            end = min(start + chunk_samples, len(channel_data))
            chunk = channel_data[start:end]
            
            print(f"  Chunk {i+1}/{total_chunks} ({start/sr:.2f}s - {end/sr:.2f}s)...")
            
            # Save chunk to temporary file
            chunk_path = work_dir / f"channel_{channel_idx}_chunk_{i}.wav"
            sf.write(str(chunk_path), chunk, sr)
            
            # Process chunk with AudioSR
            try:
                waveform = audiosr.super_resolution(
                    self.audiosr_model,
                    str(chunk_path),
                    seed=(seed + channel_idx * 1000 + i) if seed is not None else (42 + channel_idx * 1000 + i),
                    guidance_scale=guidance_scale,
                    ddim_steps=ddim_steps,
                    latent_t_per_second=12.8
                )
                # Take first channel of output (mono)
                chunks_upscaled.append(waveform[0, 0, :])  # Shape: (samples,)
            finally:
                # Clean up chunk file
                if chunk_path.exists():
                    chunk_path.unlink()
        
        # Concatenate chunks with crossfade
        print(f"  Concatenating {len(chunks_upscaled)} chunks for channel {channel_idx + 1}...")
        
        if len(chunks_upscaled) == 1:
            return chunks_upscaled[0]
        
        # Create crossfade windows
        fade_in = np.linspace(0, 1, output_overlap_samples)
        fade_out = np.linspace(1, 0, output_overlap_samples)
        
        # Start with first chunk
        final_audio = chunks_upscaled[0].copy()
        
        for i in range(1, len(chunks_upscaled)):
            current_chunk = chunks_upscaled[i].copy()
            
            # Apply crossfade
            overlap_start = len(final_audio) - output_overlap_samples
            final_audio[overlap_start:] *= fade_out
            current_chunk[:output_overlap_samples] *= fade_in
            
            # Combine
            final_audio = np.concatenate([
                final_audio[:overlap_start],
                final_audio[overlap_start:] + current_chunk[:output_overlap_samples],
                current_chunk[output_overlap_samples:]
            ])
        
        return final_audio

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
            
            # Step 2: Apply low-pass filter (optional)
            if request.skip_lowpass:
                print("Skipping low-pass filter (skip_lowpass=True)")
                filtered_path = input_path
            else:
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

    @modal.fastapi_endpoint(method="POST")
    def upscale_stream(self, request: AudioUpscaleRequest) -> StreamingResponse:
        """Upscale audio with streaming progress updates via Server-Sent Events (SSE).
        
        Progress breakdown:
        - 0-5%: Downloading
        - 5-10%: Filtering  
        - 10-95%: Upscaling (divided among all chunks)
        - 95-100%: Converting and finalizing
        """
        
        def generate_events():
            start_time = time.perf_counter()
            work_dir = CONTAINER_TEMP_DIR / f"request_{int(time.time() * 1000)}"
            work_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Step 1: Download audio (0-5%)
                yield self._sse_event("progress", {
                    "step": "downloading",
                    "message": "Downloading audio from URL...",
                    "percent": 0
                })
                
                input_ext = Path(request.audio_url.split("?")[0]).suffix or ".mp3"
                input_path = work_dir / f"input{input_ext}"
                self._download_audio(request.audio_url, input_path)
                
                yield self._sse_event("progress", {
                    "step": "downloaded",
                    "message": "Audio downloaded successfully",
                    "percent": 5
                })
                
                # Step 2: Apply low-pass filter (5-10%)
                if request.skip_lowpass:
                    yield self._sse_event("progress", {
                        "step": "filtering",
                        "message": "Skipping low-pass filter",
                        "percent": 10
                    })
                    filtered_path = input_path
                else:
                    yield self._sse_event("progress", {
                        "step": "filtering",
                        "message": f"Applying low-pass filter ({request.lowpass_frequency}Hz)...",
                        "percent": 5
                    })
                    filtered_path = work_dir / "filtered.wav"
                    self._apply_lowpass_filter(
                        input_path, filtered_path,
                        frequency=request.lowpass_frequency,
                        poles=request.lowpass_poles
                    )
                    yield self._sse_event("progress", {
                        "step": "filtered",
                        "message": "Low-pass filter applied",
                        "percent": 10
                    })
                
                # Step 3: Upscale with streaming progress (10-95%)
                upscaled_path = work_dir / "upscaled.wav"
                for event in self._upscale_audio_stream(
                    filtered_path, upscaled_path, work_dir,
                    request.ddim_steps, request.guidance_scale, request.seed
                ):
                    yield self._sse_event("progress", event)
                
                # Step 4: Convert format (95-100%)
                yield self._sse_event("progress", {
                    "step": "converting",
                    "message": f"Converting to {request.output_format.value}...",
                    "percent": 95
                })
                
                output_path = work_dir / f"output.{request.output_format.value}"
                self._convert_to_format(upscaled_path, output_path, request.output_format)
                
                # Read and encode
                with open(output_path, "rb") as f:
                    audio_base64 = base64.b64encode(f.read()).decode("utf-8")
                
                processing_time = time.perf_counter() - start_time
                
                yield self._sse_event("complete", {
                    "message": "Audio upscaled successfully",
                    "audio_base64": audio_base64,
                    "processing_time_seconds": round(processing_time, 2),
                    "output_format": request.output_format.value,
                    "percent": 100
                })
                
            except Exception as e:
                yield self._sse_event("error", {"message": str(e)})
            finally:
                import shutil
                if work_dir.exists():
                    shutil.rmtree(work_dir, ignore_errors=True)
        
        return StreamingResponse(
            generate_events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
    
    def _sse_event(self, event_type: str, data: dict) -> str:
        """Format a Server-Sent Event message."""
        data["event"] = event_type
        return f"data: {json.dumps(data)}\n\n"
    
    def _upscale_audio_stream(self, input_path, output_path, work_dir, ddim_steps, guidance_scale, seed):
        """Upscale audio with streaming progress updates (10-95% of total progress)."""
        import numpy as np
        import soundfile as sf
        
        CHUNK_DURATION, OVERLAP_DURATION, OUTPUT_SR = 5.12, 0.5, 48000
        UPSCALE_START_PERCENT = 10
        UPSCALE_END_PERCENT = 95
        UPSCALE_RANGE = UPSCALE_END_PERCENT - UPSCALE_START_PERCENT  # 85%
        
        info = sf.info(str(input_path))
        duration, num_channels = info.duration, info.channels
        
        yield {
            "step": "analyzing", 
            "message": f"Audio: {duration:.1f}s, {num_channels} channel(s)",
            "percent": UPSCALE_START_PERCENT
        }
        
        audio_data, sr = sf.read(str(input_path))
        
        is_stereo = len(audio_data.shape) > 1
        channels = [audio_data[:, i] for i in range(audio_data.shape[1])] if is_stereo else [audio_data]
        
        # Calculate total chunks for percentage
        chunk_samples = int(CHUNK_DURATION * sr)
        overlap_samples = int(OVERLAP_DURATION * sr)
        step_samples = chunk_samples - overlap_samples
        
        total_chunks_all_channels = 0
        for ch_data in channels:
            if duration <= CHUNK_DURATION + 1.0:
                total_chunks_all_channels += 1
            else:
                total_chunks_all_channels += (len(ch_data) - overlap_samples) // step_samples + 1
        
        # Percent per chunk
        percent_per_chunk = UPSCALE_RANGE / total_chunks_all_channels
        current_chunk_global = 0
        
        upscaled = []
        for ch_idx, ch_data in enumerate(channels):
            yield {
                "step": "channel_start", 
                "channel": ch_idx + 1, 
                "total_channels": len(channels),
                "message": f"Starting channel {ch_idx + 1}/{len(channels)}",
                "percent": round(UPSCALE_START_PERCENT + current_chunk_global * percent_per_chunk)
            }
            
            result = None
            for evt in self._upscale_channel_stream(
                ch_data, sr, work_dir, ch_idx, duration,
                CHUNK_DURATION, OVERLAP_DURATION, OUTPUT_SR,
                ddim_steps, guidance_scale, seed, len(channels),
                current_chunk_global, total_chunks_all_channels, percent_per_chunk, UPSCALE_START_PERCENT
            ):
                if isinstance(evt, dict):
                    yield evt
                    if evt.get("step") == "chunk_complete":
                        current_chunk_global += 1
                else:
                    result = evt
            upscaled.append(result)
        
        if is_stereo:
            min_len = min(len(c) for c in upscaled)
            final = np.column_stack([c[:min_len] for c in upscaled])
        else:
            final = upscaled[0]
        
        sf.write(str(output_path), final, OUTPUT_SR)
        yield {
            "step": "upscale_complete", 
            "message": f"Upscaled to {OUTPUT_SR}Hz",
            "percent": UPSCALE_END_PERCENT
        }
    
    def _upscale_channel_stream(self, ch_data, sr, work_dir, ch_idx, duration,
                                 chunk_dur, overlap_dur, output_sr, ddim_steps, guidance_scale, seed, total_ch,
                                 global_chunk_start, total_chunks_global, percent_per_chunk, base_percent):
        """Upscale single channel with percentage progress."""
        import numpy as np
        import soundfile as sf
        
        chunk_samples = int(chunk_dur * sr)
        overlap_samples = int(overlap_dur * sr)
        step_samples = chunk_samples - overlap_samples
        output_overlap = int(overlap_dur * output_sr)
        
        global_chunk = global_chunk_start
        
        if duration <= chunk_dur + 1.0:
            # Single chunk
            current_percent = round(base_percent + global_chunk * percent_per_chunk)
            yield {
                "step": "chunk", 
                "chunk": 1, 
                "total_chunks": 1, 
                "channel": ch_idx + 1, 
                "total_channels": total_ch,
                "global_chunk": global_chunk + 1,
                "total_global_chunks": total_chunks_global,
                "message": f"Processing chunk 1/1 (channel {ch_idx + 1}/{total_ch})",
                "percent": current_percent
            }
            temp = work_dir / f"ch{ch_idx}_temp.wav"
            sf.write(str(temp), ch_data, sr)
            try:
                w = audiosr.super_resolution(self.audiosr_model, str(temp),
                    seed=(seed + ch_idx * 1000) if seed else (42 + ch_idx * 1000),
                    guidance_scale=guidance_scale, ddim_steps=ddim_steps, latent_t_per_second=12.8)
                
                # Signal chunk complete for counter
                yield {"step": "chunk_complete", "percent": round(base_percent + (global_chunk + 1) * percent_per_chunk)}
                yield w[0, 0, :]
            finally:
                temp.unlink() if temp.exists() else None
            return
        
        total_chunks = (len(ch_data) - overlap_samples) // step_samples + 1
        chunks = []
        
        for i, start in enumerate(range(0, len(ch_data) - overlap_samples, step_samples)):
            end = min(start + chunk_samples, len(ch_data))
            current_percent = round(base_percent + global_chunk * percent_per_chunk)
            
            yield {
                "step": "chunk", 
                "chunk": i + 1, 
                "total_chunks": total_chunks,
                "channel": ch_idx + 1, 
                "total_channels": total_ch,
                "global_chunk": global_chunk + 1,
                "total_global_chunks": total_chunks_global,
                "time_range": f"{start/sr:.1f}s-{end/sr:.1f}s",
                "message": f"Processing chunk {i + 1}/{total_chunks} (channel {ch_idx + 1}/{total_ch})",
                "percent": current_percent
            }
            
            cpath = work_dir / f"ch{ch_idx}_c{i}.wav"
            sf.write(str(cpath), ch_data[start:end], sr)
            try:
                w = audiosr.super_resolution(self.audiosr_model, str(cpath),
                    seed=(seed + ch_idx * 1000 + i) if seed else (42 + ch_idx * 1000 + i),
                    guidance_scale=guidance_scale, ddim_steps=ddim_steps, latent_t_per_second=12.8)
                chunks.append(w[0, 0, :])
                
                global_chunk += 1
                # Signal chunk complete
                yield {"step": "chunk_complete", "percent": round(base_percent + global_chunk * percent_per_chunk)}
            finally:
                cpath.unlink() if cpath.exists() else None
        
        # Crossfade merge
        yield {"step": "merging", "message": f"Merging {len(chunks)} chunks for channel {ch_idx + 1}"}
        
        fade_in, fade_out = np.linspace(0, 1, output_overlap), np.linspace(1, 0, output_overlap)
        final = chunks[0].copy()
        for c in chunks[1:]:
            c = c.copy()
            os = len(final) - output_overlap
            final[os:] *= fade_out
            c[:output_overlap] *= fade_in
            final = np.concatenate([final[:os], final[os:] + c[:output_overlap], c[output_overlap:]])
        
        yield final


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
