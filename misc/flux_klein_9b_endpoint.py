# ---
# lambda-test: false
# ---

# # Serve a Flux Klein 9B endpoint on Modal
#
# This example demonstrates how to run a high-performance Flux Klein 9B endpoint
# on Modal GPUs. Flux Klein 9B is a state-of-the-art text-to-image generation model
# from Black Forest Labs that can create high-quality images from text prompts
# in just 4 inference steps.

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
CONTAINER_CACHE_VOLUME = modal.Volume.from_name("flux_klein_9b_endpoint", create_if_missing=True)

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

# We then install all the Python dependencies needed for Flux Klein 9B inference.

flux_klein_image = (
    nvidia_cuda_image
    .apt_install("git")
    .pip_install(
        "packaging",
        "ninja",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
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

app = modal.App("flux_klein_9b_endpoint", image=flux_klein_image)

with flux_klein_image.imports():
    import time
    import base64
    from enum import Enum
    from typing import Optional, Union, List
    import torch
    from diffusers import Flux2KleinPipeline
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
        image: Optional[Union[str, List[str]]] = None # Base64 encoded image(s) for image-to-image generation
        height: Optional[int] = Field(default=1024, ge=512, le=2048, multiple_of=64)
        width: Optional[int] = Field(default=1024, ge=512, le=2048, multiple_of=64)
        num_inference_steps: Optional[int] = Field(default=4, ge=1, le=50)  # Klein uses 4 steps by default
        guidance_scale: Optional[float] = Field(default=1.0, ge=0.0, le=20.0)  # Klein uses 1.0 by default
        seed: Optional[int] = None
        output_format: Optional[OutputFormat] = Field(default=OutputFormat.JPEG)
        output_quality: Optional[int] = Field(default=90, ge=1, le=100)

    def decode_base64_image(image_str: str) -> Image.Image:
        image_bytes = base64.b64decode(image_str)
        return Image.open(BytesIO(image_bytes)).convert("RGB")

# ## Common configuration for all Flux Klein 9B services

# Shared configuration dict to avoid repetition across service classes.
common_config = dict(
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    gpu="L40S",  # Using L40S for the 9B model - requires ~29GB VRAM
    volumes={
        CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME,
    },
    min_containers=0,
    buffer_containers=0,
    timeout=3600,
    enable_memory_snapshot=True,
    # experimental_options={"enable_gpu_snapshot": True},  # Temporarily disabled
)


# ## FluxKleinBase class

# This base class handles model loading, optimization, and inference.
# Subclasses inherit all the logic and only differ in their scaledown_window.

class FluxKleinBase:
    """Base class containing all shared logic for Flux Klein 9B inference."""

    @modal.enter(snap=True)
    def load_to_cpu(self):
        """Load model weights to CPU memory for snapshotting.
        
        This runs during the CPU memory snapshot phase when no GPU is available.
        The weights are loaded to CPU and will be snapshotted for faster cold starts.
        """
        print("Loading Flux Klein 9B pipeline to CPU...")
        
        # Load the pipeline with bfloat16 for optimal performance
        # Keep on CPU during snapshot phase - GPU is not available yet
        self.pipeline = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9B",
            torch_dtype=torch.bfloat16,
            cache_dir=CONTAINER_CACHE_DIR
        )
        
        print("Pipeline loaded to CPU. Ready for memory snapshot.")

    @modal.enter(snap=False)
    def move_to_gpu(self):
        """Move the model to GPU after GPU is attached.
        
        This runs after the container restores from snapshot and a GPU is available.
        """
        print("Moving pipeline to GPU...")
        
        # Now we can safely move to GPU as the GPU is attached
        self.pipeline.to("cuda")
        
        print("Pipeline moved to GPU. Ready for inference.")

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

        # Handle image input for image-to-image generation
        if request.image:
            if isinstance(request.image, list):
                inputs["image"] = [decode_base64_image(img) for img in request.image]
            else:
                inputs["image"] = decode_base64_image(request.image)

        # Time the inference
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Generate image using the Flux Klein 9B pipeline
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


# ## Service classes with different scaledown windows

@app.cls(
    scaledown_window=2,  # Aggressive scaledown for cost savings
    **common_config,
)
class FluxKlein9BService(FluxKleinBase):
    """Standard service with aggressive scaledown (2s) for cost savings."""
    pass


@app.cls(
    scaledown_window=10,  # Longer scaledown for burst requests
    **common_config,
)
class FluxKlein9BBatchService(FluxKleinBase):
    """Batch-optimized service with longer scaledown (10s) for burst requests."""
    pass


# ## Local testing

# You can test the endpoint locally using Modal's built-in testing utilities.

@app.local_entrypoint()
def main():
    """Test the Flux Klein 9B endpoint locally"""
    service = FluxKlein9BService()
    
    # Example prompt
    prompt = "A cat holding a sign that says hello world"
    
    request = ImageGenerationRequest(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=42,
        output_format=OutputFormat.PNG
    )
    
    print(f"Generating image with prompt: {prompt[:100]}...")
    response = service.inference.remote(request)
    
    # Save the generated image
    output_path = Path("flux_klein_output.png")
    output_path.write_bytes(response.content)
    print(f"Image saved to {output_path}")
