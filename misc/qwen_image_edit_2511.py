# ---
# lambda-test: false
# ---

# # Serve a Qwen Image Edit 2511 endpoint on Modal

# This example demonstrates how to run a high-performance Qwen Image Edit 2511 endpoint
# on Modal GPUs. Qwen-Image-Edit-2511 is a state-of-the-art image editing model that
# supports enhanced character consistency, multi-person group photos, and built-in LoRA effects.

# The endpoint supports flexible image editing with various parameters
# and automatically processes edited images.

# ## Import dependencies and set up paths

from __future__ import annotations

from pathlib import Path
from io import BytesIO

import modal

# Container mount directories
CONTAINER_CACHE_DIR = Path("/cache")
CONTAINER_CLOUD_MOUNT_DIR = Path("/outputs")

# Modal volume for caching compiled model artifacts and other caches across container restarts to reduce cold start times.
CONTAINER_CACHE_VOLUME = modal.Volume.from_name("qwen_image_edit_2511_endpoint", create_if_missing=True)

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

# We then install all the Python dependencies needed for Qwen Image Edit 2511 inference.
# Note: Qwen-Image-Edit-2511 requires the latest version of diffusers from GitHub.

qwen_image_edit_2511_image = (
    nvidia_cuda_image
    .apt_install("git")
    .run_commands(
        # Install the latest diffusers from GitHub (required for QwenImageEditPlusPipeline)
        "pip install git+https://github.com/huggingface/diffusers.git",
    )
    .pip_install(
        "accelerate~=1.8.1",
        "fastapi[standard]==0.115.12",
        "huggingface-hub[hf_transfer]==0.33.1",
        "opencv-python-headless==4.11.0.86",
        "pydantic==2.11.4",
        "safetensors==0.5.3",
        "sentencepiece==0.2.0",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
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
    })
)

# ## Creating the Modal app

# We create a Modal App using the defined image and import necessary dependencies
# within the container's runtime environment.

app = modal.App("qwen_image_edit_2511_endpoint", image=qwen_image_edit_2511_image)

with qwen_image_edit_2511_image.imports():
    import time
    from enum import Enum
    from typing import Optional, List
    import numpy as np
    import torch
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.utils import load_image
    from pydantic import BaseModel, Field, HttpUrl
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

    class ImageEditRequest(BaseModel):
        image_urls: List[HttpUrl]  # URLs of the input images to edit (supports 1-2 images for multi-person fusion)
        prompt: str  # Edit instruction describing the desired output
        negative_prompt: Optional[str] = Field(default=" ", description="Negative prompt for guidance")
        true_cfg_scale: Optional[float] = Field(default=4.0, ge=0.0, le=20.0, description="True CFG scale for classifier-free guidance")
        num_inference_steps: Optional[int] = Field(default=40, ge=1, le=100, description="Number of inference steps")
        num_images: Optional[int] = Field(default=1, ge=1, le=4, description="Number of images to generate")
        seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
        output_format: Optional[OutputFormat] = Field(default=OutputFormat.JPEG, description="Output image format")
        output_quality: Optional[int] = Field(default=90, ge=1, le=100, description="Output image quality (for JPEG/WEBP)")

# ## The QwenImageEdit2511Service class

# This class handles model loading, optimization, and inference. We use Modal's
# class decorator to control the lifecycle of our cloud container as well as to
# configure auto-scaling parameters, the GPU type, and necessary secrets.


@app.cls(
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    gpu="A100-80GB",  # H100 recommended for Qwen-Image-Edit-2511 (larger model than 2509)
    volumes={
        CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME,
    },
    min_containers=0,
    buffer_containers=0,
    scaledown_window=60,  # 1 minute
    timeout=3600,  # 1 hour
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class QwenImageEdit2511Service:
    # ## Model loading

    @modal.enter(snap=True)
    def load(self):
        """Load the Qwen-Image-Edit-2511 pipeline into CPU memory for snapshotting."""
        print("Loading Qwen-Image-Edit-2511 pipeline...")
        
        # Load the pipeline with bfloat16 precision
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511",
            torch_dtype=torch.bfloat16,
            cache_dir=CONTAINER_CACHE_DIR
        )
        
        self.pipeline.set_progress_bar_config(disable=None)
        print("Pipeline loaded successfully. Ready for GPU memory snapshot.")

    @modal.enter(snap=False)
    def setup(self):
        """Move the pipeline to GPU after snapshot restoration."""
        print("Moving pipeline to GPU...")
        self.pipeline.to("cuda")
        print("Pipeline ready for inference on GPU.")

    # ## The main inference endpoint

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: ImageEditRequest):
        """
        Edit images based on text prompts using Qwen-Image-Edit-2511.
        
        This model supports:
        - Enhanced character consistency preservation
        - Multi-person group photo fusion (2 separate images into 1)
        - Built-in LoRA effects (lighting, viewpoint changes, etc.)
        - Industrial design and material replacement
        - Geometric reasoning for design annotations
        """
        # Load and preprocess input images
        input_images = [load_image(str(url)) for url in request.image_urls]
        input_images = [img.convert("RGB") for img in input_images]
        
        # Set up generator for reproducibility
        generator = (
            torch.manual_seed(request.seed)
            if request.seed is not None
            else None
        )

        # Prepare inputs for the pipeline (following the official example)
        # Note: guidance_scale is not used by Qwen-Image-Edit-2511 (not guidance-distilled)
        inputs = {
            "image": input_images,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "true_cfg_scale": request.true_cfg_scale,
            "num_inference_steps": request.num_inference_steps,
            "num_images_per_prompt": request.num_images,
        }
        
        if generator is not None:
            inputs["generator"] = generator

        # Time the inference
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Generate edited images using the Qwen Image Edit 2511 pipeline
        with torch.inference_mode():
            output = self.pipeline(**inputs)
            images = output.images

        torch.cuda.synchronize()
        print(f"inference time: {time.perf_counter() - t0:.2f}s")
        t1 = time.perf_counter()

        # Process and return the result
        import concurrent.futures
        
        def process_image(image):
            byte_stream = BytesIO()
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            save_kwargs = {"format": request.output_format.value}
            if request.output_format in [OutputFormat.JPEG, OutputFormat.WEBP]:
                save_kwargs["quality"] = request.output_quality
            
            image.save(byte_stream, **save_kwargs)
            return byte_stream.getvalue()

        # Process images in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            image_bytes = list(executor.map(process_image, images))

        torch.cuda.synchronize()
        print(f"image processing time: {time.perf_counter() - t1:.2f}s")
        
        # Return the first image if only one was requested, otherwise all images
        if len(image_bytes) == 1:
            return Response(
                content=image_bytes[0],
                media_type=request.output_format.mime_type
            )
        else:
            # Return multiple images as response
            return Response(
                content=b"".join(image_bytes),
                media_type="application/octet-stream"
            )


# ## Running the endpoint locally for testing
#
# To deploy this endpoint, run:
# ```bash
# modal deploy qwen_image_edit_2511.py
# ```
#
# To run locally for testing:
# ```bash
# modal serve qwen_image_edit_2511.py
# ```
#
# Example API call:
# ```bash
# curl -X POST "https://your-modal-endpoint.modal.run/inference" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "image_urls": ["https://example.com/image1.png", "https://example.com/image2.png"],
#     "prompt": "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square.",
#     "num_inference_steps": 40,
#     "true_cfg_scale": 4.0
#   }' \
#   --output output.jpg
# ```
