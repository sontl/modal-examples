# ---
# lambda-test: false
# ---

# # Serve a fast FLUX.1 [dev] endpoint on Modal with Nunchaku optimizations

# This example demonstrates how to run a high-performance FLUX.1 image generation endpoint
# on Modal GPUs using Nunchaku optimizations. FLUX.1 is a state-of-the-art text-to-image model 
# from Black Forest Labs that produces high-quality images from text prompts.

# The endpoint supports flexible image generation with various parameters
# and automatically uploads generated images to cloud storage (Cloudflare R2).

# ## Import dependencies and set up paths

# We start by importing the necessary libraries and defining our storage paths.
# We use Modal Volumes for caching model artifacts and Modal CloudBucketMounts for
# storing generated images.

from __future__ import annotations

from pathlib import Path
from io import BytesIO

import modal

# Container mount directories
CONTAINER_CACHE_DIR = Path("/cache")
CONTAINER_CLOUD_MOUNT_DIR = Path("/outputs")

# Modal volume for caching compiled model artifacts and other caches across container restarts to reduce cold start times.
CONTAINER_CACHE_VOLUME = modal.Volume.from_name("flux_kontext_nunchanku_endpoint", create_if_missing=True)

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

# We then install all the Python dependencies needed for FLUX.1 inference.

flux_kontext_nunchanku_endpoint_image = (
    nvidia_cuda_image
    .apt_install("git")
    .run_commands(
        "pip install git+https://github.com/huggingface/diffusers.git",
        # Then install the specific wheel which will override if needed
        "pip install https://huggingface.co/mit-han-lab/nunchaku/resolve/main/nunchaku-0.3.1+torch2.7-cp312-cp312-linux_x86_64.whl",
    )
    .pip_install(
        "accelerate~=1.8.1",
        "fastapi[standard]==0.115.12",
        "huggingface-hub[hf_transfer]==0.33.1",
        "numpy==2.2.4",
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

app = modal.App("flux_kontext_nunchanku_endpoint", image=flux_kontext_nunchanku_endpoint_image)

with flux_kontext_nunchanku_endpoint_image.imports():
    import concurrent.futures
    import time
    from enum import Enum
    from typing import Optional, List
    import cv2
    import numpy as np
    import torch
    from diffusers import FluxKontextPipeline
    from diffusers.utils import load_image
    from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
    from pydantic import BaseModel, Field, HttpUrl
    from PIL import Image
    from fastapi import Response
    from fastapi.responses import JSONResponse
    # Add Nunchaku import
    from nunchaku.utils import get_precision
    from nunchaku import NunchakuFluxTransformer2dModel  

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

    class InferenceRequest(BaseModel):
        image_url: HttpUrl  # URL of the input image to edit
        prompt: str  # Edit instruction
        negative_prompt: Optional[str] = None
        height: Optional[int] = Field(default=None, ge=256, le=1024, multiple_of=16)
        width: Optional[int] = Field(default=None, ge=256, le=1024, multiple_of=16)
        guidance_scale: Optional[float] = Field(default=2.5, ge=0.0, le=20.0, multiple_of=0.1)
        num_images: Optional[int] = Field(default=1, ge=1, le=4)
        seed: Optional[int] = None
        output_format: Optional[OutputFormat] = Field(default=OutputFormat.JPEG)
        output_quality: Optional[int] = Field(default=90, ge=1, le=100)

# ## The FluxKontextnunchankuService class

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
)
class FluxKontextnunchankuService:
    # ## Model optimization methods

    # These methods apply various optimizations to make model inference faster.
    # The main optimizations are first block cache and torch compile.

    def _optimize(self):
         # apply first block cache, see: [ParaAttention](https://github.com/chengzeyi/ParaAttention)
        # apply_cache_on_pipe(
        #     self.pipe,
        #     residual_diff_threshold=0.12,  # don't recommend going higher
        # )
        # Nunchaku handles most optimizations for the transformer
        # We only apply optimizations to the VAE
        self.pipe.transformer.to(memory_format=torch.channels_last)
        self.pipe.vae.fuse_qkv_projections()
        self.pipe.vae.to(memory_format=torch.channels_last)

    def _compile(self):
        # Load a real image for compilation
        image_url = "https://i.ibb.co/TB3vGhfb/test-image.png"
        real_image = load_image(str(image_url))
        
        print("triggering torch compile")
        self.pipe(prompt="the girl is dancing", image=real_image, height=1024, width=1024, num_images_per_prompt=1)

    # ## Mega-cache management

    # PyTorch "mega-cache" serializes compiled model artifacts into a blob that
    # can be easily transferred to another machine with the same GPU.

    def _load_mega_cache(self):
        print("loading torch mega-cache")
        try:
            if self.mega_cache_bin_path.exists():
                with open(self.mega_cache_bin_path, "rb") as f:
                    artifact_bytes = f.read()

                if artifact_bytes:
                    torch.compiler.load_cache_artifacts(artifact_bytes)
            else:
                print("torch mega cache not found, regenerating...")
        except Exception as e:
            print(f"error loading torch mega-cache: {e}")

    def _save_mega_cache(self):
        print("saving torch mega-cache")
        try:
            artifacts = torch.compiler.save_cache_artifacts()
            artifact_bytes, _ = artifacts

            with open(self.mega_cache_bin_path, "wb") as f:
                f.write(artifact_bytes)

            # persist changes to volume
            CONTAINER_CACHE_VOLUME.commit()
        except Exception as e:
            print(f"error saving torch mega-cache: {e}")

    # ## Memory Snapshotting

    # We utilize memory snapshotting to avoid reloading model weights into host memory
    # during subsequent container starts.

    @modal.enter(snap=True)
    def load(self):
        print("Loading base pipeline...")
        # Load base pipeline without transformer during snapshot
        self.pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            transformer=None,  # We'll load the transformer in setup
            torch_dtype=torch.bfloat16
        ).to("cpu")

        # Initialize mega cache paths
        mega_cache_dir = CONTAINER_CACHE_DIR / ".mega_cache"
        mega_cache_dir.mkdir(parents=True, exist_ok=True)
        self.mega_cache_bin_path = mega_cache_dir / "flux_torch_mega"

    @modal.enter(snap=False)
    def setup(self):
        print("Initializing Nunchaku transformer...")
        # Initialize Nunchaku transformer now that GPU is available
        precision = "int4"
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"mit-han-lab/nunchaku-flux.1-kontext-dev/svdq-{precision}_r32-flux.1-kontext-dev.safetensors"
        )
        transformer.set_attention_impl("nunchaku-fp16")

        ### LoRA Related Code ###
        # transformer.update_lora_params(
        #     "alimama-creative/FLUX.1-Turbo-Alpha/diffusion_pytorch_model.safetensors"
        # )  # Path to your LoRA safetensors, can also be a remote HuggingFace path
        # transformer.set_lora_strength(1)  # Your LoRA strength here
        ### End of LoRA Related Code ###

        
        # Set the transformer in the pipeline
        self.pipe.transformer = transformer
        self.pipe.to("cuda")
        
        self._load_mega_cache()
        self._optimize()
        # self._compile()
        self._save_mega_cache()

    # ## The main inference endpoint

    # This method handles incoming requests, generates images, and uploads them
    # to cloud storage.

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        # Load and preprocess input image
        input_image = load_image(str(request.image_url))
        
        generator = (
            torch.Generator("cuda").manual_seed(request.seed)
            if request.seed is not None
            else None
        )

        # Time the inference
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Generate edited images using the FLUX Kontext pipeline
        pipe_args = {
            "image": input_image,
            "prompt": request.prompt,
        }
        
        # Add optional parameters if provided
        if request.negative_prompt is not None:
            pipe_args["negative_prompt"] = request.negative_prompt
        if request.height is not None:
            pipe_args["height"] = request.height
        if request.width is not None:
            pipe_args["width"] = request.width
        if request.guidance_scale is not None:
            pipe_args["guidance_scale"] = request.guidance_scale
        if request.num_images is not None:
            pipe_args["num_images_per_prompt"] = request.num_images
        if generator is not None:
            pipe_args["generator"] = generator
        
        # pipe_args["num_inference_steps"] = 8

        images = self.pipe(**pipe_args).images

        torch.cuda.synchronize()
        print(f"inference time: {time.perf_counter() - t0:.2f}s")
        t1 = time.perf_counter()

        # If only one image is requested, return it directly as binary response
        if request.num_images == 1:
            byte_stream = BytesIO()
            if isinstance(images[0], np.ndarray):
                images[0] = Image.fromarray(images[0])
            
            images[0].save(
                byte_stream, 
                format=request.output_format.value,
                quality=request.output_quality if request.output_format in [OutputFormat.JPEG, OutputFormat.WEBP] else None
            )
            return Response(
                content=byte_stream.getvalue(),
                media_type=request.output_format.mime_type
            )

        # For multiple images, return a JSON response with base64 encoded images
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
        
        # Return images as multipart response
        return Response(
            content=b"\n".join(image_bytes),
            media_type="multipart/mixed; boundary=frame"
        )
