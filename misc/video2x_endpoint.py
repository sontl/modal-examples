# ---
# lambda-test: false
# ---

# # Video2X Video Upscaling Endpoint on Modal

# This example demonstrates how to run a Video2X video upscaling endpoint
# on Modal GPUs. Video2X is a framework for upscaling video, audio, and images
# using various AI upscaling algorithms.

# Endpoint path: POST /
# Usage: Send a JSON body with {"video_url": "...", "scale": "2x"}

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
# Using CUDA 12.6.3 which is available for Ubuntu 24.04
cuda_version = "12.6.3"
flavor = "devel"
operating_system = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_system}"

# Definition of the Video2X image
# We build Video2X from source using the provided 'just' script for Ubuntu
video2x_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .entrypoint([])
    # Install runtime and build dependencies
    .apt_install(
        "git",
        "wget",
        "curl",
        "ffmpeg",
        "libvulkan1",
        "mesa-vulkan-drivers",
        "vulkan-tools",
        "build-essential",
        "cmake",
        "pkg-config",
        "libssl-dev",
        "ca-certificates",
        "python3-pip",
        "python3-venv",
    )
    # Install Rust and Just for building
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "bash -c 'source $HOME/.cargo/env && cargo install just'",
    )
    # Clone and build Video2X
    .run_commands(
        "git clone --recurse-submodules https://github.com/k4yt3x/video2x.git /root/video2x",
        "mkdir -p /root/video2x/build",
    )
    .workdir("/root/video2x")
    # We use the ubuntu2404 target from the justfile.
    .run_commands(
        "bash -c 'source $HOME/.cargo/env && just ubuntu2404'",
        # The build produces a .deb file in the current directory. Install it.
        "dpkg -i *.deb || apt-get install -f -y",
    )
    .pip_install("requests", "fastapi[standard]", "pydantic")
    .env({
        "VIDEO2X_VERBOSE": "1",
        "XDG_CACHE_HOME": str(CONTAINER_CACHE_DIR),
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
        X4 = "4x"

    class UpscaleRequest(BaseModel):
        video_url: str
        scale: Optional[UpscaleMode] = Field(default=UpscaleMode.X2)
        driver: str = Field(default="realesrgan", description="Upscaling driver (realesrgan, waifu2x, etc.)")
        process_timeout: int = Field(default=1800, description="Timeout for processing in seconds")

# ## The Video2XService class

@app.cls(
    gpu="A10G",
    volumes={
        CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME,
    },
    timeout=3600,
)
class Video2XService:
    
    @modal.enter()
    def check_installation(self):
        """Verify Video2X is installed and working."""
        import subprocess
        try:
            res = subprocess.run(["video2x", "--version"], capture_output=True, text=True, check=True)
            print(f"Video2X Version: {res.stdout.strip()}")
        except Exception as e:
            print(f"Video2X check failed: {e}")

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
            
            cmd = [
                "video2x",
                "-i", str(input_path),
                "-o", str(output_path),
                "-p", request.driver,
                "-s", str(scale_factor),
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            t0 = time.time()
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            duration = time.time() - t0
            print(f"Upscaling completed in {duration:.2f}s")
            
            if not output_path.exists():
                raise RuntimeError("Output file was not created by Video2X.")
            
            with open(output_path, "rb") as f:
                video_bytes = f.read()
                
            return video_bytes
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
    # Big Buck Bunny snippet (small)
    test_url = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
    
    request = UpscaleRequest(video_url=test_url, scale=UpscaleMode.X2)
    print(f"Testing upscale with URL: {test_url}")
    
    try:
        video_bytes = service.run_upscale_remote.remote(request)
        print(f"Upscaling successful. Received {len(video_bytes)} bytes.")
        
        output_path = Path("video2x_output.mp4")
        output_path.write_bytes(video_bytes)
        print(f"Saved to {output_path}")
    except Exception as e:
        print(f"Upscaling failed: {e}")
