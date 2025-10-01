# ---
# lambda-test: false
# ---

# # Serve a Qwen Image Edit endpoint on Modal with Nunchaku optimizations

# This example demonstrates how to run a high-performance Qwen Image Edit endpoint
# on Modal GPUs using Nunchaku optimizations. Qwen Image Edit is a state-of-the-art 
# image editing model that can modify images based on text prompts and reference images.

# The endpoint supports flexible image editing with various parameters
# and automatically processes edited images.

from __future__ import annotations

from pathlib import Path
from io import BytesIO

import modal

# Container mount directories
CONTAINER_CACHE_DIR = Path("/cache")
CONTAINER_CLOUD_MOUNT_DIR = Path("/outputs")

# Modal volume for caching compiled model artifacts and other caches across container restarts to reduce cold start times.
CONTAINER_CACHE_VOLUME = modal.Volume.from_name("qwen_image_edit_endpoint", create_if_missing=True)

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

# We then install all the Python dependencies needed for Qwen Image Edit inference.

qwen_image_edit_endpoint_image = (
    nvidia_cuda_image
    .apt_install("git")
    .run_commands(
        "pip install git+https://github.com/huggingface/diffusers.git",
        # Then install the specific wheel which will override if needed
        "pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1/nunchaku-1.0.1+torch2.7-cp312-cp312-linux_x86_64.whl",
    )
    .pip_install(
        "accelerate~=1.8.1",
        "fastapi[standard]==0.115.12",
        "huggingface-hub[hf_transfer]==0.33.1",
        "opencv-python-headless==4.11.0.86",
        "para-attn==0.3.32",
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
        "TORCH_COMPILE_DEBUG": "0",  # Enable debug info for torch.compile
    })
)

# ## Creating the Modal app

# We create a Modal App using the defined image and import necessary dependencies
# within the container's runtime environment.

app = modal.App("qwen_image_edit_endpoint", image=qwen_image_edit_endpoint_image)

with qwen_image_edit_endpoint_image.imports():
    import math
    import time
    from enum import Enum
    from typing import Optional, List
    import cv2
    import numpy as np
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
    from diffusers.utils import load_image
    from pydantic import BaseModel, Field, HttpUrl
    from PIL import Image
    from fastapi import Response
    from fastapi.responses import JSONResponse
    
    # Add Nunchaku import
    from nunchaku.utils import get_precision, get_gpu_memory
    from nunchaku import NunchakuQwenImageTransformer2DModel  

    # From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),  # We use shift=3 in distillation
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),  # We use shift=3 in distillation
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,  # set shift_terminal to None
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    
    # Supported output formats for generated images
    class OutputFormat(Enum):
        PNG = "PNG"
        JPEG = "JPEG"  # Changed from JPG to JPEG
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
        image_urls: List[HttpUrl]  # URLs of the input images to edit
        prompt: str  # Edit instruction
        true_cfg_scale: Optional[float] = Field(default=1.0, ge=0.0, le=20.0, multiple_of=0.1)
        num_inference_steps: Optional[int] = Field(default=4, ge=1, le=20)
        num_images: Optional[int] = Field(default=1, ge=1, le=4)
        seed: Optional[int] = None
        output_format: Optional[OutputFormat] = Field(default=OutputFormat.JPEG)
        output_quality: Optional[int] = Field(default=90, ge=1, le=100)

# ## The QwenImageEditService class

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
    scaledown_window=2,  # 1 seconds
    timeout=3600,  # 1 hour
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class QwenImageEditService:
    # ## Model loading and optimization

    @modal.enter(snap=True)
    def load(self):
        print("Loading Qwen Image Edit pipeline with Nunchaku optimizations...")
        
        num_inference_steps = 8  # you can also use the 8-step model to improve the quality
        rank = 128  # you can also use the rank=128 model to improve the quality
        precision = get_precision()
        model_path = f"nunchaku-tech/nunchaku-qwen-image-edit-2509/svdq-{precision}_r{rank}-qwen-image-edit-2509-lightningv2.0-{num_inference_steps}steps.safetensors"

        # Load the model
        self.transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            model_path,
            cache_dir=CONTAINER_CACHE_DIR
        )

        # Create scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        # Load the pipeline
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509", 
            transformer=self.transformer, 
            scheduler=scheduler, 
            torch_dtype=torch.bfloat16,
            cache_dir=CONTAINER_CACHE_DIR
        )

        if get_gpu_memory() > 18:
            self.pipeline.enable_model_cpu_offload()
        else:
            # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
            self.transformer.set_offload(
                False, use_pin_memory=False, num_blocks_on_gpu=1
            )  # increase num_blocks_on_gpu if you have more VRAM
            self.pipeline._exclude_from_cpu_offload.append("transformer")
            self.pipeline.enable_sequential_cpu_offload()

        print("Pipeline loaded and optimized. Ready for GPU memory snapshot.")

    # ## The main inference endpoint

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: ImageEditRequest):
        # Load and preprocess input images
        input_images = [load_image(str(url)) for url in request.image_urls]
        input_images = [img.convert("RGB") for img in input_images]
        
        generator = (
            torch.Generator("cuda").manual_seed(request.seed)
            if request.seed is not None
            else None
        )

        # Prepare inputs for the pipeline
        inputs = {
            "image": input_images,
            "prompt": request.prompt,
            "true_cfg_scale": request.true_cfg_scale,
            "num_inference_steps": request.num_inference_steps,
        }
        
        if generator is not None:
            inputs["generator"] = generator

        # Time the inference
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Generate edited images using the Qwen Image Edit pipeline
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
            
            image.save(
                byte_stream, 
                format=request.output_format.value,
                quality=request.output_quality if request.output_format in [OutputFormat.JPEG, OutputFormat.WEBP] else None
            )
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