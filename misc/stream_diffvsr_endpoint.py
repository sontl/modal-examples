# ---
# lambda-test: false
# ---

# # Stream-DiffVSR Video Super-Resolution API on Modal
#
# This implementation provides an API endpoint for Stream-DiffVSR, a causally conditioned 
# diffusion framework for efficient online video super-resolution.
#
# Key features:
# - 4x upscaling with diffusion-based enhancement
# - Four-step distilled denoiser for fast inference  
# - Auto-regressive Temporal Guidance (ARTG) module
# - Temporal Processor Module (TPM) for detail and temporal coherence
# - Processes 720p frames in ~0.328 seconds on RTX4090
#
# Model: https://huggingface.co/Jamichsu/Stream-DiffVSR
# Paper: https://github.com/jamichss/Stream-DiffVSR

from __future__ import annotations

from pathlib import Path
import tempfile
import os

import modal

# Model configuration
MODEL_ID = "Jamichsu/Stream-DiffVSR"
BASE_MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"
MAX_FRAMES_PER_BATCH = 30  # Process frames in batches to manage memory

# Build Stream-DiffVSR environment for modern GPUs (A100, RTX 6000 Pro, RTX 50-Series)
# Following their CUDA 12 setup from README for newer GPUs
cuda_version = "12.4.1"
flavor = "cudnn-devel"  # Need devel for compilation
operating_system = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_system}"

nvidia_cuda_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.10"
).entrypoint([])

stream_diffvsr_image = (
    nvidia_cuda_image
    .apt_install(
        "git", "git-lfs", "ffmpeg", "build-essential", "ninja-build", "cmake",
    )
    # Install PyTorch with CUDA 12 wheels (following their RTX 6000 Pro/RTX 50-Series setup)
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "fastapi[standard]==0.115.12",
        "pydantic==2.11.4", 
        "requests>=2.32.2",
        "wheel",
        "packaging",
        "ninja",
        "huggingface_hub==0.25.1",  # Match their version
        "hf_transfer",
        # Core AI dependencies - upgraded for CUDA 12 compatibility
        "accelerate>=0.34.0",
        "diffusers>=0.31.0",
        "transformers>=4.45.0",
        "safetensors==0.4.5",
        "einops==0.8.0",
        "peft",
        # Image/Video processing
        "imageio==2.36.1",
        "imageio-ffmpeg==0.6.0",
        "pillow==10.4.0",
        "opencv-python==4.10.0.84",
        "av==14.1.0",
        # Utilities
        "numpy==1.26.4",
        "scipy==1.13.1",
        "tqdm==4.66.5",
        "pyyaml==6.0.2",
        "matplotlib==3.9.2",
        # Memory optimization - xformers for RTX 6000 Pro / RTX 50-Series / A100
        "xformers==0.0.28.post1",  # Compatible with torch 2.4.0 + CUDA 12.4
    )
    .run_commands(
        # Setup Git LFS
        "git lfs install",
        # Clone Stream-DiffVSR repository
        "rm -rf /Stream-DiffVSR",
        "git clone https://github.com/jamichss/Stream-DiffVSR.git /Stream-DiffVSR",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONPATH": "/Stream-DiffVSR",
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        "HF_HOME": "/cache/huggingface",
    })
    .run_commands(
        # Pre-download the model during image build
        "python -c \"from huggingface_hub import snapshot_download; snapshot_download('Jamichsu/Stream-DiffVSR', local_dir='/cache/stream_diffvsr_model')\"",
    )
    .workdir("/Stream-DiffVSR")
)

app = modal.App("stream_diffvsr_endpoint", image=stream_diffvsr_image)

with stream_diffvsr_image.imports():
    import time
    import sys
    import os
    import shutil
    import subprocess
    from enum import Enum
    from typing import Optional, List
    import requests
    import numpy as np
    from PIL import Image
    import imageio
    from tqdm import tqdm
    import torch
    from accelerate.utils import set_seed
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

    from pydantic import BaseModel, Field, HttpUrl
    from fastapi import Response, HTTPException
    
    # Add Stream-DiffVSR paths to sys.path
    stream_diffvsr_paths = [
        '/Stream-DiffVSR',
        '/Stream-DiffVSR/pipeline',
        '/Stream-DiffVSR/temporal_autoencoder',
    ]
    for path in stream_diffvsr_paths:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # Import Stream-DiffVSR components
    try:
        from pipeline.stream_diffvsr_pipeline import StreamDiffVSRPipeline, ControlNetModel, UNet2DConditionModel
        from diffusers import DDIMScheduler
        from temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny
        
        STREAM_DIFFVSR_AVAILABLE = True
        print("✅ Stream-DiffVSR imported successfully")
    except Exception as e:
        print(f"❌ Stream-DiffVSR imports failed: {e}")
        import traceback
        traceback.print_exc()
        STREAM_DIFFVSR_AVAILABLE = False


    class OutputFormat(Enum):
        MP4 = "mp4"
        AVI = "avi"
        MOV = "mov"

        @property
        def mime_type(self):
            return {
                OutputFormat.MP4: "video/mp4",
                OutputFormat.AVI: "video/x-msvideo",
                OutputFormat.MOV: "video/quicktime"
            }[self]


    class StreamDiffVSRRequest(BaseModel):
        url: HttpUrl
        output_format: Optional[OutputFormat] = Field(default=OutputFormat.MP4)
        num_inference_steps: Optional[int] = Field(default=4, ge=1, le=50)
        seed: Optional[int] = Field(default=42)
        max_frames: Optional[int] = Field(default=None, ge=1, le=1000)
        batch_size: Optional[int] = Field(default=30, ge=1, le=100)  # Match their MAX_FRAMES_PER_SEQ


    def get_video_fps(video_path: str) -> tuple[str, float]:
        """Get video FPS using ffprobe"""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        rate = result.stdout.strip()
        if not rate:
            raise RuntimeError("Could not determine input video FPS.")
        if "/" in rate:
            num, den = rate.split("/", 1)
            fps_value = float(num) / float(den)
        else:
            fps_value = float(rate)
        return rate, fps_value


    def has_audio(video_path: str) -> bool:
        """Check if video has audio stream"""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip() == "audio"


    def extract_frames(video_path: str, frames_dir: Path, max_frames: Optional[int] = None) -> List[Path]:
        """Extract frames from video using ffmpeg"""
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", video_path,
            "-vsync", "0",
        ]
        
        if max_frames:
            cmd.extend(["-vframes", str(max_frames)])
        
        cmd.append(str(frames_dir / "frame_%06d.png"))
        
        subprocess.run(cmd, check=True)
        
        frame_paths = sorted(frames_dir.glob("frame_*.png"))
        return frame_paths


    def assemble_video(frames_dir: Path, output_path: Path, fps_rate: str):
        """Encode frames back to video"""
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-framerate", fps_rate,
            "-i", str(frames_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "medium",
            "-movflags", "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)


    def extract_audio(video_path: str, audio_path: Path):
        """Extract audio from video"""
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", video_path,
            "-vn",
            "-c:a", "aac",
            "-b:a", "192k",
            str(audio_path),
        ]
        subprocess.run(cmd, check=True)


    def mux_audio(video_path: Path, audio_path: Path, output_path: Path):
        """Mux audio into video"""
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "copy",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)


@app.cls(
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="L40S",  # L40S (48GB) - supports CUDA 12, cost-effective for video upscaling
    timeout=3600,
    scaledown_window=10,  # 10 seconds
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class StreamDiffVSRService:
    
    @modal.enter(snap=True)
    def setup(self):
        print("Setting up Stream-DiffVSR...")
        
        # Initialize seed for reproducibility
        set_seed(42)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Allow TF32 for faster computation
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Verify Stream-DiffVSR is available
        if not STREAM_DIFFVSR_AVAILABLE:
            raise RuntimeError("Stream-DiffVSR is not available. Check installation logs.")
        
        self.pipeline = None
        self.of_model = None
        
        print("Stream-DiffVSR setup complete")

    def _load_component(self, cls, weight_path, model_id, subfolder):
        """Load a model component from pretrained weights"""
        path = weight_path if weight_path else model_id
        sub = None if weight_path else subfolder
        return cls.from_pretrained(path, subfolder=sub)

    def init_pipeline(self):
        """Initialize Stream-DiffVSR pipeline"""
        if self.pipeline is not None:
            return self.pipeline, self.of_model
        
        print("Initializing Stream-DiffVSR pipeline...")
        
        # Use locally cached model path (downloaded during image build)
        local_model_path = "/cache/stream_diffvsr_model"
        
        try:
            # Clear GPU memory before loading
            torch.cuda.empty_cache()
            
            # Load model components from local cache
            print(f"Loading ControlNet from {local_model_path}...")
            controlnet = ControlNetModel.from_pretrained(local_model_path, subfolder="controlnet")
            
            print(f"Loading UNet from {local_model_path}...")
            unet = UNet2DConditionModel.from_pretrained(local_model_path, subfolder="unet")
            
            print(f"Loading Temporal VAE from {local_model_path}...")
            vae = TemporalAutoencoderTiny.from_pretrained(local_model_path, subfolder="vae")
            
            print(f"Loading Scheduler from {local_model_path}...")
            scheduler = DDIMScheduler.from_pretrained(local_model_path, subfolder="scheduler")
            
            # Create pipeline
            print("Creating StreamDiffVSRPipeline...")
            self.pipeline = StreamDiffVSRPipeline.from_pretrained(
                local_model_path,
                controlnet=controlnet,
                vae=vae,
                unet=unet,
                scheduler=scheduler,
                local_files_only=True,
            )
            
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory efficient attention
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print("✅ xformers memory efficient attention enabled")
            except Exception as e:
                print(f"⚠️ Could not enable xformers: {e}")
            
            # Load optical flow model (RAFT) for temporal guidance
            print("Loading RAFT optical flow model...")
            self.of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(self.device).eval()
            self.of_model.requires_grad_(False)
            
            print("✅ Stream-DiffVSR pipeline initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize Stream-DiffVSR pipeline: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            raise RuntimeError(f"Stream-DiffVSR pipeline initialization failed: {e}")
        
        return self.pipeline, self.of_model

    def download_video(self, video_url: str) -> str:
        """Download video from URL"""
        response = requests.get(str(video_url), stream=True)
        response.raise_for_status()
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()
        return temp_file.name

    def load_frames(self, frame_paths: List[Path]) -> List[Image.Image]:
        """Load frames as PIL Images (matching their app.py)"""
        frames = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as img:
                frames.append(img.convert("RGB"))
        return frames

    def upscale_frames(
        self, 
        frames: List[Image.Image], 
        num_inference_steps: int = 4
    ) -> List[Image.Image]:
        """
        Upscale frames using Stream-DiffVSR (4x upscaling).
        Matches their app.py implementation exactly.
        """
        input_w, input_h = frames[0].size
        print(f"Upscaling {len(frames)} frames from {input_w}x{input_h} (4x upscale -> {input_w*4}x{input_h*4})")
        print(f"  Inference steps: {num_inference_steps}")
        
        with torch.inference_mode():
            # Match their app.py exactly
            output = self.pipeline(
                "",  # Empty prompt
                frames,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=0,
                of_model=self.of_model,
            )
        
        # Extract upscaled frames - each frame is a list with one image
        frames_hr = output.images
        return [frame[0] if isinstance(frame, (list, tuple)) else frame for frame in frames_hr]

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: StreamDiffVSRRequest):
        temp_input_path = None
        work_dir = None
        
        try:
            # Download input video
            print(f"Downloading video from {request.url}")
            temp_input_path = self.download_video(str(request.url))
            
            # Initialize pipeline
            self.init_pipeline()
            
            # Set seed for reproducibility
            if request.seed is not None:
                set_seed(request.seed)
            
            # Create working directory
            work_dir = Path(tempfile.mkdtemp())
            input_frames_dir = work_dir / "input_frames"
            output_frames_dir = work_dir / "output_frames"
            output_frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Get video FPS
            fps_rate, fps_value = get_video_fps(temp_input_path)
            print(f"Source FPS: {fps_rate} ({fps_value:.2f})")
            
            # Extract frames
            print("Extracting frames...")
            frame_paths = extract_frames(temp_input_path, input_frames_dir, request.max_frames)
            total_frames = len(frame_paths)
            print(f"Extracted {total_frames} frames")
            
            # Time the inference
            start_time = time.time()
            torch.cuda.empty_cache()
            
            # Process frames in batches
            batch_size = request.batch_size or MAX_FRAMES_PER_BATCH
            
            total_batches = (total_frames + batch_size - 1) // batch_size
            processed_frame_count = 0
            
            for batch_idx in range(total_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, total_frames)
                batch_frame_paths = frame_paths[batch_start:batch_end]
                
                print(f"Processing batch {batch_idx + 1}/{total_batches} (frames {batch_start + 1}-{batch_end})")
                
                # Load batch frames
                batch_frames = self.load_frames(batch_frame_paths)
                
                # Upscale batch (4x upscaling like their app.py)
                upscaled_frames = self.upscale_frames(
                    batch_frames,
                    num_inference_steps=request.num_inference_steps
                )
                
                # Save upscaled frames
                for frame, frame_path in zip(upscaled_frames, batch_frame_paths):
                    output_path = output_frames_dir / frame_path.name
                    frame.save(output_path)
                    processed_frame_count += 1
                
                # Cleanup batch memory
                del batch_frames
                del upscaled_frames
                torch.cuda.empty_cache()
            
            inference_time = time.time() - start_time
            print(f"Stream-DiffVSR inference completed in {inference_time:.2f}s")
            print(f"Average time per frame: {inference_time / total_frames:.3f}s")
            
            # Assemble output video
            print("Assembling output video...")
            video_no_audio = work_dir / "upscaled_no_audio.mp4"
            assemble_video(output_frames_dir, video_no_audio, fps_rate)
            
            # Handle audio
            final_output = work_dir / f"upscaled.{request.output_format.value}"
            if has_audio(temp_input_path):
                print("Muxing audio...")
                audio_path = work_dir / "audio.m4a"
                extract_audio(temp_input_path, audio_path)
                mux_audio(video_no_audio, audio_path, final_output)
            else:
                print("No audio detected, skipping mux")
                shutil.move(str(video_no_audio), str(final_output))
            
            # Read output video
            if not final_output.exists():
                raise HTTPException(status_code=500, detail="Output video not generated")
            
            with open(final_output, 'rb') as f:
                video_bytes = f.read()
            
            if len(video_bytes) == 0:
                raise HTTPException(status_code=500, detail="Output video is empty")
            
            return Response(
                content=video_bytes,
                media_type=request.output_format.mime_type,
                headers={
                    "Content-Disposition": f"attachment; filename=stream_diffvsr_upscaled.{request.output_format.value}",
                    "X-Processing-Time": str(inference_time),
                    "X-Frames-Processed": str(total_frames),
                    "X-FPS": str(fps_value),
                    "X-Inference-Steps": str(request.num_inference_steps),
                }
            )
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        finally:
            # Cleanup
            if temp_input_path and os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if work_dir and work_dir.exists():
                shutil.rmtree(work_dir)

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        """Health check endpoint for monitoring service status"""
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
                "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            }
        
        return {
            "status": "healthy",
            "model": MODEL_ID,
            "pipeline_loaded": self.pipeline is not None,
            "optical_flow_loaded": self.of_model is not None,
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "stream_diffvsr_available": STREAM_DIFFVSR_AVAILABLE,
            **gpu_info,
        }


@app.local_entrypoint()
def main():
    """Local entrypoint for testing"""
    print("Stream-DiffVSR Modal Endpoint")
    print("=" * 50)
    print(f"Model: {MODEL_ID}")
    print(f"Features:")
    print("  - 4x upscaling with diffusion-based enhancement")
    print("  - Four-step distilled denoiser for fast inference")
    print("  - Auto-regressive Temporal Guidance (ARTG)")
    print("  - Temporal Processor Module (TPM)")
    print("  - Processes 720p frames in ~0.328s on RTX4090")
    print()
    print("To deploy, run: modal deploy stream_diffvsr_endpoint.py")
    print("To serve locally, run: modal serve stream_diffvsr_endpoint.py")
