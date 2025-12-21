# ---
# lambda-test: false
# ---

# # Real-ESRGAN Image Upscaling Endpoint on Modal

# This example demonstrates how to run a Real-ESRGAN image upscaling endpoint
# on Modal GPUs. Real-ESRGAN is a practical algorithm for general image restoration
# that can upscale images by 2x, 3x, or 4x while enhancing details.

# The endpoint accepts an image URL and returns the upscaled image as base64 or binary.
# Endpoint path: POST /

from __future__ import annotations

from pathlib import Path

import modal

# Container mount directories
CONTAINER_CACHE_DIR = Path("/cache")

# Modal volume for caching model weights across container restarts
CONTAINER_CACHE_VOLUME = modal.Volume.from_name("realesrgan_cache", create_if_missing=True)

# ## Building the container image

cuda_version = "12.6.3"
flavor = "devel"
operating_system = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_system}"

realesrgan_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .entrypoint([])
    .apt_install("git", "libgl1", "libglib2.0-0", "wget")
    .pip_install(
        "numpy==1.26.4",
        "torch==2.1.2",
        "torchvision==0.16.2",
        "opencv-python-headless",
        "Pillow",
        "requests",
        "fastapi[standard]==0.115.12",
        "pydantic==2.11.4",
    )
    .pip_install(
        "basicsr==1.4.2",
        "realesrgan",
    )
    .run_commands("pip install numpy==1.26.4 --force-reinstall --no-deps")
    .env({
        "CUDA_CACHE_PATH": str(CONTAINER_CACHE_DIR / ".nv_cache"),
    })
)

# ## Creating the Modal app

app = modal.App("realesrgan_endpoint", image=realesrgan_image)

with realesrgan_image.imports():
    import base64
    import time
    from enum import Enum
    from typing import Optional

    import cv2
    import numpy as np
    import requests
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from fastapi import Response
    from pydantic import BaseModel, Field
    from realesrgan import RealESRGANer

    class UpscaleMode(str, Enum):
        X2 = "2x"
        X3 = "3x"
        X4 = "4x"

    class ResponseFormat(str, Enum):
        BASE64 = "base64"
        BINARY = "binary"

    class UpscaleRequest(BaseModel):
        image_url: str
        scale: Optional[UpscaleMode] = Field(default=UpscaleMode.X2)
        tile_size: Optional[int] = Field(default=0, ge=0, le=1024)
        response_format: Optional[ResponseFormat] = Field(default=ResponseFormat.BASE64)


# ## The RealESRGANService class

@app.cls(
    gpu="T4",
    volumes={
        CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME,
    },
    min_containers=0,
    buffer_containers=0,
    scaledown_window=10,
    timeout=600,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=50, target_inputs=45)  # Allow a 20% burst
class RealESRGANService:
    
    @modal.enter(snap=True)
    def load(self):
        """Load Real-ESRGAN models with GPU memory snapshot support."""
        print("Loading Real-ESRGAN models...")
        
        model_dir = CONTAINER_CACHE_DIR / "realesrgan_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self._download_models(model_dir)
        
        # RealESRGAN x4plus model
        model_x4 = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4
        )
        self.upscaler_x4 = RealESRGANer(
            scale=4,
            model_path=str(model_dir / "RealESRGAN_x4plus.pth"),
            model=model_x4, tile=0, tile_pad=10, pre_pad=0,
            half=True, device="cuda"
        )
        
        # RealESRGAN x2plus model
        model_x2 = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=2
        )
        self.upscaler_x2 = RealESRGANer(
            scale=2,
            model_path=str(model_dir / "RealESRGAN_x2plus.pth"),
            model=model_x2, tile=0, tile_pad=10, pre_pad=0,
            half=True, device="cuda"
        )
        
        print("Real-ESRGAN models loaded. Ready for GPU memory snapshot.")

    def _download_models(self, model_dir: Path):
        models = {
            "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        }
        for filename, url in models.items():
            filepath = model_dir / filename
            if not filepath.exists():
                print(f"Downloading {filename}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {filename}")

    def _load_image_from_url(self, url: str) -> np.ndarray:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to decode image from URL")
        return img

    @modal.fastapi_endpoint(method="POST", docs=True)
    def upscale(self, request: UpscaleRequest):
        """Upscale an image. POST to / with JSON body."""
        t0 = time.perf_counter()
        
        print(f"Loading image from: {request.image_url}")
        img = self._load_image_from_url(request.image_url)
        print(f"Input image shape: {img.shape}")
        
        scale_map = {UpscaleMode.X2: 2, UpscaleMode.X3: 3, UpscaleMode.X4: 4}
        outscale = scale_map[request.scale]
        
        upscaler = self.upscaler_x2 if request.scale == UpscaleMode.X2 else self.upscaler_x4
        
        if request.tile_size > 0:
            upscaler.tile = request.tile_size
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        output, _ = upscaler.enhance(img, outscale=outscale)
        
        torch.cuda.synchronize()
        print(f"Upscale time: {time.perf_counter() - t1:.2f}s")
        print(f"Output image shape: {output.shape}")
        
        _, buffer = cv2.imencode(".png", output)
        image_bytes = buffer.tobytes()
        
        total_time = time.perf_counter() - t0
        print(f"Total processing time: {total_time:.2f}s")
        
        if request.response_format == ResponseFormat.BINARY:
            return Response(
                content=image_bytes,
                media_type="image/png",
                headers={
                    "X-Original-Width": str(img.shape[1]),
                    "X-Original-Height": str(img.shape[0]),
                    "X-Upscaled-Width": str(output.shape[1]),
                    "X-Upscaled-Height": str(output.shape[0]),
                    "X-Scale": request.scale.value,
                    "X-Processing-Time": str(round(total_time, 2)),
                }
            )
        
        return {
            "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
            "original_size": {"width": img.shape[1], "height": img.shape[0]},
            "upscaled_size": {"width": output.shape[1], "height": output.shape[0]},
            "scale": request.scale.value,
            "processing_time_seconds": round(total_time, 2),
        }


@app.local_entrypoint()
def main():
    service = RealESRGANService()
    test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
    
    request = UpscaleRequest(image_url=test_url, scale=UpscaleMode.X2)
    print(f"Testing upscale with URL: {test_url}")
    result = service.upscale.remote(request)
    
    print(f"Original: {result['original_size']}, Upscaled: {result['upscaled_size']}")
    print(f"Processing time: {result['processing_time_seconds']}s")
    
    output_path = Path("realesrgan_output.png")
    output_path.write_bytes(base64.b64decode(result["image_base64"]))
    print(f"Saved to {output_path}")
