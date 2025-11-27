# ---
# lambda-test: false
# ---

# # Serve a Z-Image endpoint on Modal with Flash Attention optimizations

# This example demonstrates how to run a high-performance Z-Image endpoint
# on Modal GPUs using Flash Attention 2 optimizations. Z-Image is a state-of-the-art 
# text-to-image generation model that can create high-quality images from text prompts.

# The endpoint supports flexible image generation with various parameters
# and automatically processes generated images.

from __future__ import annotations

from pathlib import Path
from io import BytesIO

import modal

# Container mount directories
CONTAINER_CACHE_DIR = Path("/cache")
CONTAINER_CLOUD_MOUNT_DIR = Path("/outputs")

# Modal volume for caching compiled model artifacts and other caches across container restarts to reduce cold start times.
CONTAINER_CACHE_VOLUME = modal.Volume.from_name("z_image_endpoint", create_if_missing=True)

# ## Building the container image

# We start with an NVIDIA CUDA base image that includes the necessary GPU drivers
# and development tools.

# Image configuration and setup
cuda_version = "12.6.3"
flavor = "devel"
operating_system = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_system}"

nvidia_cuda_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.12"
).entrypoint([])

# We then install all the Python dependencies needed for Z-Image inference.

z_image_endpoint_image = (
    nvidia_cuda_image
    .apt_install("git")
    .pip_install(
        "packaging",
        "ninja",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
    )
    .pip_install(
        "flash-attn>=2.6.3",
    )
    .run_commands(
        "pip install git+https://github.com/huggingface/diffusers.git",
    )
    .pip_install(
        "accelerate~=1.8.1",
        "fastapi[standard]==0.115.12",
        "huggingface-hub[hf_transfer]==0.33.1",
        "pydantic==2.11.4",
        "safetensors==0.5.3",
        "sentencepiece==0.2.0",
        "transformers==4.53.0",
        "Pillow~=11.2.1",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "CUDA_CACHE_PATH": str(CONTAINER_CACHE_DIR / ".nv_cache"),
        "HF_HUB_CACHE": str(CONTAINER_CACHE_DIR / ".hf_hub_cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(CONTAINER_CACHE_DIR / ".inductor_cache"),
        "TRITON_CACHE_DIR": str(CONTAINER_CACHE_DIR / ".triton_cache"),
        "TORCH_COMPILE_DEBUG": "0",
    })
)

# ## Creating the Modal app

# We create a Modal App using the defined image and import necessary dependencies
# within the container's runtime environment.

app = modal.App("z_image_endpoint", image=z_image_endpoint_image)

with z_image_endpoint_image.imports():
    import time
    from enum import Enum
    from typing import Optional
    import torch
    from diffusers import ZImagePipeline
    from pydantic import BaseModel, Field
    from PIL import Image
    from fastapi import Response
    
    # Supported output formats for generated images
    class OutputFormat(Enum):
        PNG = "PNG"
        JPEG = "JPEG"
        WEBP = "WEBP"

        @property
        def mime_type(self):
            return {
                OutputFormat.PNG: "image/png",
                OutputFormat.JPEG: "image/jpeg",
                OutputFormat.WEBP: "image/webp"
            }[self]

    # ### Defining request/response model

    # We use Pydantic to define a strongly-typed request model. This gives us
    # automatic validation for our API endpoint.

    class ImageGenerationRequest(BaseModel):
        prompt: str  # Text prompt for image generation
        height: Optional[int] = Field(default=1024, ge=512, le=2048, multiple_of=64)
        width: Optional[int] = Field(default=1024, ge=512, le=2048, multiple_of=64)
        num_inference_steps: Optional[int] = Field(default=9, ge=1, le=50)
        guidance_scale: Optional[float] = Field(default=0.0, ge=0.0, le=20.0)
        seed: Optional[int] = None
        output_format: Optional[OutputFormat] = Field(default=OutputFormat.JPEG)
        output_quality: Optional[int] = Field(default=90, ge=1, le=100)

# ## The ZImageService class

# This class handles model loading, optimization, and inference. We use Modal's
# class decorator to control the lifecycle of our cloud container as well as to
# configure auto-scaling parameters, the GPU type, and necessary secrets.


@app.cls(
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    gpu="L40S",
    volumes={
        CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME,
    },
    min_containers=0,
    buffer_containers=0,
    scaledown_window=2,
    timeout=3600,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class ZImageService:
    # ## Model loading and optimization

    @modal.enter(snap=True)
    def load(self):
        print("Loading Z-Image pipeline with Flash Attention 2 optimizations...")
        
        # Load the pipeline with bfloat16 for optimal performance
        self.pipeline = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            cache_dir=CONTAINER_CACHE_DIR
        )
        
        self.pipeline.to("cuda")
        
        # Enable Flash Attention 2 for better efficiency
        try:
            self.pipeline.transformer.set_attention_backend("flash")
            print("Flash Attention 2 enabled successfully")
        except Exception as e:
            print(f"Could not enable Flash Attention 2: {e}")
            print("Falling back to default SDPA attention")

        print("Pipeline loaded and optimized. Ready for GPU memory snapshot.")

    # ## The main inference endpoint

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: ImageGenerationRequest):
        generator = (
            torch.Generator("cuda").manual_seed(request.seed)
            if request.seed is not None
            else None
        )

        # Prepare inputs for the pipeline
        inputs = {
            "prompt": request.prompt,
            "height": request.height,
            "width": request.width,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
        }
        
        if generator is not None:
            inputs["generator"] = generator

        # Time the inference
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Generate image using the Z-Image pipeline
        output = self.pipeline(**inputs)
        image = output.images[0]

        torch.cuda.synchronize()
        print(f"inference time: {time.perf_counter() - t0:.2f}s")
        t1 = time.perf_counter()

        # Process and return the result
        byte_stream = BytesIO()
        image.save(
            byte_stream, 
            format=request.output_format.value,
            quality=request.output_quality if request.output_format in [OutputFormat.JPEG, OutputFormat.WEBP] else None
        )
        image_bytes = byte_stream.getvalue()

        torch.cuda.synchronize()
        print(f"image processing time: {time.perf_counter() - t1:.2f}s")
        
        return Response(
            content=image_bytes,
            media_type=request.output_format.mime_type
        )


# ## Local testing

# You can test the endpoint locally using Modal's built-in testing utilities.

@app.local_entrypoint()
def main():
    """Test the Z-Image endpoint locally"""
    service = ZImageService()
    
    # Example prompt
    prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
    
    request = ImageGenerationRequest(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
        seed=42,
        output_format=OutputFormat.PNG
    )
    
    print(f"Generating image with prompt: {prompt[:100]}...")
    response = service.inference.remote(request)
    
    # Save the generated image
    output_path = Path("z_image_output.png")
    output_path.write_bytes(response.content)
    print(f"Image saved to {output_path}")
