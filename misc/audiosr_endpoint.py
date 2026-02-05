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
                upscaled = waveform[0, 0, :]  # Shape: (samples,)
                
                # Trim to expected output duration (AudioSR pads internally)
                expected_output_samples = int(duration * output_sr)
                if len(upscaled) > expected_output_samples:
                    upscaled = upscaled[:expected_output_samples]
                
                return upscaled
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        
        # For long audio, process in chunks
        print(f"Channel {channel_idx + 1}: Processing in {chunk_duration}s chunks...")
        
        # Calculate chunk parameters in samples (input sample rate)
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap_duration * sr)
        step_samples = chunk_samples - overlap_samples
        
        # Calculate output parameters (output sample rate = 48kHz)
        output_chunk_samples = int(chunk_duration * output_sr)
        output_overlap_samples = int(overlap_duration * output_sr)
        output_step_samples = output_chunk_samples - output_overlap_samples
        
        # Calculate expected total output length
        expected_output_total = int(duration * output_sr)
        
        # Process chunks
        chunks_upscaled = []
        chunk_infos = []  # Store info about each chunk for proper trimming
        total_chunks = (len(channel_data) - overlap_samples) // step_samples + 1
        
        for i, start in enumerate(range(0, len(channel_data) - overlap_samples, step_samples)):
            end = min(start + chunk_samples, len(channel_data))
            chunk = channel_data[start:end]
            chunk_input_duration = len(chunk) / sr
            
            print(f"  Chunk {i+1}/{total_chunks} ({start/sr:.2f}s - {end/sr:.2f}s, duration={chunk_input_duration:.2f}s)...")
            
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
                upscaled_chunk = waveform[0, 0, :]  # Shape: (samples,)
                
                # Calculate expected output length for this chunk
                expected_chunk_output = int(chunk_input_duration * output_sr)
                
                # Trim to expected length (remove AudioSR's internal padding)
                if len(upscaled_chunk) > expected_chunk_output:
                    upscaled_chunk = upscaled_chunk[:expected_chunk_output]
                
                chunks_upscaled.append(upscaled_chunk)
                chunk_infos.append({
                    'input_start': start,
                    'input_end': end,
                    'input_duration': chunk_input_duration,
                    'output_length': len(upscaled_chunk)
                })
            finally:
                # Clean up chunk file
                if chunk_path.exists():
                    chunk_path.unlink()
        
        # Concatenate chunks with crossfade
        print(f"  Concatenating {len(chunks_upscaled)} chunks for channel {channel_idx + 1}...")
        
        if len(chunks_upscaled) == 1:
            final_audio = chunks_upscaled[0]
        else:
            # Create crossfade windows
            fade_in = np.linspace(0, 1, output_overlap_samples)
            fade_out = np.linspace(1, 0, output_overlap_samples)
            
            # Start with first chunk
            final_audio = chunks_upscaled[0].copy()
            
            for i in range(1, len(chunks_upscaled)):
                current_chunk = chunks_upscaled[i].copy()
                
                # Ensure we have enough samples for crossfade
                if len(final_audio) < output_overlap_samples or len(current_chunk) < output_overlap_samples:
                    # Not enough for crossfade, just concatenate
                    final_audio = np.concatenate([final_audio, current_chunk])
                    continue
                
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
        
        # Trim final output to exact expected duration
        if len(final_audio) > expected_output_total:
            print(f"  Trimming output from {len(final_audio)/output_sr:.2f}s to {expected_output_total/output_sr:.2f}s")
            final_audio = final_audio[:expected_output_total]
        
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
