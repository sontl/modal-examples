# ---
# lambda-test: false
# ---

# # Video2X Video Upscaling Endpoint on Modal

# This example demonstrates how to run a Video2X video upscaling endpoint
# on Modal GPUs. Video2X is a framework for upscaling video, audio, and images
# using various AI upscaling algorithms (RealESRGAN, libplacebo, etc.).

# Based on the working Google Colab setup that uses the pre-built Video2X 6.2.0 package.

# Endpoint: POST /upscale
# Usage: Send a JSON body with:
#   {
#     "video_url": "https://example.com/video.mp4",
#     "scale": "4x",           # 2x, 3x, or 4x
#     "processor": "realesrgan",
#     "model": "realesr-animevideov3",  # realesr-animevideov3, realesrgan-plus-anime, realesrgan-plus
#     "codec": "libx264",
#     "preset": "slow",
#     "crf": 20
#   }

from __future__ import annotations

import os
from pathlib import Path
import shutil
import modal

# Container mount directories
CONTAINER_CACHE_DIR = Path("/cache")

# Modal volume for caching models/downloads
CONTAINER_CACHE_VOLUME = modal.Volume.from_name("video2x_cache", create_if_missing=True)

# ## Building the container image
# Using CUDA with Ubuntu 22.04 to match the working Video2X pre-built package
# The pre-built .deb is for Ubuntu 22.04, so we use a compatible CUDA base image
cuda_version = "12.1.1"
flavor = "devel"
operating_system = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_system}"

# Video2X pre-built package URL (version 6.2.0 for Ubuntu 22.04)
VIDEO2X_DEB_URL = "https://github.com/k4yt3x/video2x/releases/download/6.2.0/video2x-linux-ubuntu2204-amd64.deb"

# Definition of the Video2X image
# We use the pre-built .deb package instead of building from source (more reliable)
video2x_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .entrypoint([])
    # Add FFmpeg 7 PPA and install dependencies (matching Colab setup)
    .run_commands(
        "apt-get update",
        "apt-get install -y software-properties-common curl wget",
        # Download Video2X .deb package
        f"curl -LO {VIDEO2X_DEB_URL}",
        # Add FFmpeg 7 PPA for proper ffmpeg support
        "add-apt-repository -y ppa:ubuntuhandbook1/ffmpeg7",
        # Add NVIDIA graphics drivers PPA for Vulkan support
        "add-apt-repository -y ppa:graphics-drivers/ppa",
        "apt-get update",
        # Install libvulkan1, vulkan tools, ffmpeg, and the video2x .deb
        # Also try to install libnvidia-gl for Vulkan ICD support
        # Use --no-install-recommends to avoid pulling in the full driver
        "apt-get install -y --no-install-recommends libvulkan1 vulkan-tools mesa-vulkan-drivers ffmpeg libnvidia-gl-535 ./video2x-linux-ubuntu2204-amd64.deb || "
        "apt-get install -y libvulkan1 vulkan-tools mesa-vulkan-drivers ffmpeg ./video2x-linux-ubuntu2204-amd64.deb",
        # Create ICD directories for runtime configuration
        "mkdir -p /usr/share/vulkan/icd.d /etc/vulkan/icd.d",
    )
    .pip_install("requests", "fastapi[standard]", "pydantic")
    .env({
        "VIDEO2X_VERBOSE": "1",
        "XDG_CACHE_HOME": str(CONTAINER_CACHE_DIR),
        # Ensure NVIDIA driver library paths are included
        "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu",
    })
)

app = modal.App("video2x_endpoint", image=video2x_image)

with video2x_image.imports():
    import base64
    import time
    from enum import Enum
    from typing import Optional
    import requests
    from fastapi import HTTPException, Response
    from pydantic import BaseModel, Field

    class UpscaleMode(str, Enum):
        X2 = "2x"
        X3 = "3x"
        X4 = "4x"

    class RealESRGANModel(str, Enum):
        ANIMEVIDEOV3 = "realesr-animevideov3"  # Best for anime, supports 2x, 3x, 4x
        PLUS_ANIME = "realesrgan-plus-anime"  # For anime, only 4x
        PLUS = "realesrgan-plus"  # For real-life videos, only 4x

    class UpscaleRequest(BaseModel):
        video_url: str
        scale: Optional[UpscaleMode] = Field(default=UpscaleMode.X4, description="Upscaling factor (2x, 3x, or 4x)")
        processor: str = Field(default="realesrgan", description="Upscaling processor (realesrgan, libplacebo)")
        model: str = Field(default="realesr-animevideov3", description="RealESRGAN model to use")
        codec: str = Field(default="libx264", description="Output video codec")
        preset: str = Field(default="slow", description="Encoder preset (ultrafast, fast, medium, slow, etc.)")
        crf: int = Field(default=20, description="Constant Rate Factor (0-51, lower = better quality)")
        log_level: str = Field(default="info", description="Video2X log level")
        process_timeout: int = Field(default=1800, description="Timeout for processing in seconds")

# ## The Video2XService class

@app.cls(
    gpu="T4",  # T4 works well with Video2X based on Colab testing
    volumes={
        CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME,
    },
    timeout=3600,
    scaledown_window=20,  # 20 seconds
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=26, target_inputs=22)  # Allow a 20% burst
class Video2XService:
    
    @modal.enter(snap=True)
    def check_installation(self):
        """Verify Video2X is installed and configure Vulkan at runtime."""
        import subprocess
        import os
        import glob
        import json
        
        # Check Video2X version
        try:
            res = subprocess.run(["video2x", "--version"], capture_output=True, text=True, check=True)
            print(f"Video2X Version: {res.stdout.strip()}")
        except Exception as e:
            print(f"Video2X check failed: {e}")
        
        # Check GPU info
        try:
            res = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], 
                               capture_output=True, text=True, check=True)
            print(f"GPU: {res.stdout.strip()}")
        except Exception as e:
            print(f"nvidia-smi failed: {e}")
        
        # Configure Vulkan ICD at runtime (NVIDIA libs are mounted at runtime)
        # The libnvidia-gl-535 package provides proper Vulkan support
        vulkan_lib_candidates = [
            "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so*",
            "/usr/lib/x86_64-linux-gnu/libnvidia-vulkan-producer.so*",
        ]
        
        nvidia_vulkan_lib = None
        for pattern in vulkan_lib_candidates:
            matches = glob.glob(pattern)
            if matches:
                nvidia_vulkan_lib = matches[0]
                break
        
        # Create the NVIDIA Vulkan ICD JSON file if we found a library
        if nvidia_vulkan_lib:
            icd_json_path = "/usr/share/vulkan/icd.d/nvidia_icd.json"
            icd_content = {
                "file_format_version": "1.0.0",
                "ICD": {
                    "library_path": nvidia_vulkan_lib,
                    "api_version": "1.3"
                }
            }
            try:
                with open(icd_json_path, 'w') as f:
                    json.dump(icd_content, f)
                os.environ["VK_ICD_FILENAMES"] = icd_json_path
                print(f"Vulkan ICD configured: {nvidia_vulkan_lib}")
            except Exception as e:
                print(f"Failed to create ICD JSON: {e}")
        else:
            print("WARNING: Could not find NVIDIA Vulkan library!")

    def _download_video(self, url: str, target_path: Path):
        print(f"Downloading video from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

    def _run_upscale_impl(self, request: UpscaleRequest) -> bytes:
        import subprocess
        import uuid
        
        job_id = str(uuid.uuid4())
        job_dir = Path(f"/tmp/{job_id}")
        job_dir.mkdir(parents=True, exist_ok=True)
        
        input_path = job_dir / "input.mp4"
        output_path = job_dir / "output.mp4"
        
        try:
            self._download_video(request.video_url, input_path)
            
            scale_factor = int(request.scale.value.replace("x", ""))
            
            # Build command using Video2X 6.x CLI format (matching Colab)
            cmd = [
                "video2x",
                "--input", str(input_path),
                "--output", str(output_path),
                "--processor", request.processor,
                # Encoder options
                "--codec", request.codec,
                "--log-level", request.log_level,
                "-e", f"preset={request.preset}",
                "-e", f"crf={request.crf}",
            ]
            
            # Add processor-specific options
            if request.processor == "realesrgan":
                cmd.extend([
                    "--realesrgan-model", request.model,
                    "--scaling-factor", str(scale_factor),
                ])
            
            print(f"Running command: {' '.join(cmd)}")
            t0 = time.time()
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=request.process_timeout,
            )
            
            duration = time.time() - t0
            print(f"Processing took {duration:.2f}s")
            print(f"Video2X stdout: {process.stdout[-2000:] if len(process.stdout) > 2000 else process.stdout}")
            if process.stderr:
                print(f"Video2X stderr: {process.stderr[-1000:] if len(process.stderr) > 1000 else process.stderr}")
            
            # Check if output file exists - this is the primary success indicator
            # Video2X sometimes exits with non-zero code (e.g., -11/SIGSEGV) during cleanup
            # even when the output file was successfully written
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Output file created successfully: {output_path.stat().st_size} bytes")
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                return video_bytes
            
            # If no output file, then we have a real failure
            if process.returncode != 0:
                raise RuntimeError(f"Video2X failed with return code {process.returncode}: {process.stderr}")
            
            raise RuntimeError("Output file was not created by Video2X.")
        except subprocess.TimeoutExpired as e:
            print(f"Video2X timed out after {request.process_timeout}s")
            raise RuntimeError(f"Upscaling timed out after {request.process_timeout} seconds")
        except subprocess.CalledProcessError as e:
            print(f"Video2X Command Failed: {e.stdout} \n {e.stderr}")
            raise RuntimeError(f"Upscaling failed: {e.stderr}")
        finally:
            if job_dir.exists():
                shutil.rmtree(job_dir)

    @modal.method()
    def run_upscale_remote(self, request: UpscaleRequest) -> bytes:
        return self._run_upscale_impl(request)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def upscale(self, request: UpscaleRequest):
        try:
            video_bytes = self._run_upscale_impl(request)
            return Response(
                content=video_bytes,
                media_type="video/mp4",
                headers={"X-Video2X-Status": "Success"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.local_entrypoint()
def main():
    service = Video2XService()
    # Big Buck Bunny snippet (small, 360p, 10 seconds)
    test_url = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
    
    # Test with the new request format (4x upscale with realesr-animevideov3 model)
    request = UpscaleRequest(
        video_url=test_url,
        scale=UpscaleMode.X4,
        processor="realesrgan",
        model="realesr-animevideov3",
        codec="libx264",
        preset="slow",
        crf=20,
        log_level="info",
    )
    print(f"Testing Video2X upscale with URL: {test_url}")
    print(f"Settings: scale={request.scale}, model={request.model}")
    
    try:
        video_bytes = service.run_upscale_remote.remote(request)
        print(f"Upscaling successful. Received {len(video_bytes)} bytes.")
        
        output_path = Path("video2x_output.mp4")
        output_path.write_bytes(video_bytes)
        print(f"Saved to {output_path}")
    except Exception as e:
        print(f"Upscaling failed: {e}")
