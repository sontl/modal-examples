# # Generate videos from prompts with Lightricks LTX-Video

# This example demonstrates how to run the [LTX-Video](https://github.com/Lightricks/LTX-Video)
# video generation model by [Lightricks](https://www.lightricks.com/) on Modal.

# LTX-Video is fast! Generating a twenty second 480p video at moderate quality
# takes as little as two seconds on a warm container.

# Here's one that we generated:

# <center>
# <video controls autoplay loop muted>
# <source src="https://modal-cdn.com/blonde-woman-blinking.mp4" type="video/mp4" />
# </video>
# </center>

# ## Setup

# We start by importing dependencies we need locally,
# defining a Modal [App](https://modal.com/docs/guide/apps),
# and defining the container [Image](https://modal.com/docs/guide/images)
# that our video model will run in.


import string
import time
from pathlib import Path
from typing import Annotated, Optional

import fastapi
import modal

app = modal.App("example-ltx-video")


# ## Storing data on Modal Volumes

# On Modal, we save large or expensive-to-compute data to
# [distributed Volumes](https://modal.com/docs/guide/volumes)
# that are accessible both locally and remotely.

# We'll store the LTX-Video model's weights and the outputs we generate
# on Modal Volumes.

# We store the outputs on a Modal Volume so that clients
# don't need to sit around waiting for the video to be generated.

VOLUME_NAME = "ltx-outputs"
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = Path("/outputs")

# We store the weights on a Modal Volume so that we don't
# have to fetch them from the Hugging Face Hub every time
# a container boots. This download takes about two minutes,
# depending on traffic and network speed.

MODEL_VOLUME_NAME = "ltx-model"
model = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

# Add a cache volume for compiled model artifacts
CONTAINER_CACHE_DIR = Path("/cache")
CONTAINER_CACHE_VOLUME = modal.Volume.from_name("ltx-cache", create_if_missing=True)

# For more on storing Modal weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        "pip install git+https://github.com/huggingface/diffusers.git",
    )
    .pip_install(
        "accelerate",
        "fastapi[standard]==0.115.8",
        "hf_transfer",
        "imageio",
        "imageio-ffmpeg",
        "sentencepiece",
        "torch",
        "transformers",
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

# We don't have to change any of the Hugging Face code to do this --
# we just set the location of Hugging Face's cache to be on a Volume
# using the `HF_HOME` environment variable.

MODEL_PATH = Path("/models")
image = image.env({"HF_HOME": str(MODEL_PATH)})

# ## Setting up our LTX class

# We use the `@cls` decorator to specify the infrastructure our inference function needs,
# as defined above.

# That decorator also gives us control over the
# [lifecycle](https://modal.com/docs/guide/lifecycle-functions)
# of our cloud container.

# Specifically, we use the `enter` method to load the model into GPU memory
# (from the Volume if it's present or the Hub if it's not)
# before the container is marked ready for inputs.

# This helps reduce tail latencies caused by cold starts.
# For details and more tips, see [this guide](https://modal.com/guide/cold-start).

# The actual inference code is in a `modal.method` of the class.


MINUTES = 60  # seconds


@app.cls(
    image=image,
    volumes={
        OUTPUTS_PATH: outputs, 
        MODEL_PATH: model,
        CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME
    },
    gpu="L40S",
    timeout=10 * MINUTES,
    scaledown_window=1 * MINUTES,
    enable_memory_snapshot=True,
)
class LTX:
    
    def _optimize(self):
        import torch

        # torch compile configs
        config = torch._inductor.config
        config.conv_1x1_as_mm = True
        config.coordinate_descent_check_all_directions = True
        config.coordinate_descent_tuning = True
        config.disable_progress = False
        config.epilogue_fusion = False
        config.shape_padding = True

        self.pipe.transformer.to(memory_format=torch.channels_last)
        self.pipe.transformer = torch.compile(
            self.pipe.transformer, mode="max-autotune", fullgraph=True
        )

    def _compile(self):
        # monkey-patch torch inductor remove_noop_ops pass for dynamic compilation
        # swallow AttributeError: 'SymFloat' object has no attribute 'size' and return false
        import torch

        # Trigger compilation with a sample prompt
        print("triggering torch compile")
        self.pipe(
            prompt="a person walking",
            negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
            width=384,
            height=256,
            num_frames=8,
            guidance_scale=1.0,
            generator=torch.Generator().manual_seed(0),
            output_type="latent",
        )

    def _load_mega_cache(self):
        print("loading torch mega-cache")
        import torch
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
        import torch
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

    @modal.enter(snap=True)
    def load_model(self):
        import torch
        from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline, AutoModel
        from diffusers.hooks import apply_group_offloading

        print("downloading (if necessary) and loading model")
        # Initialize condition pipeline
        self.pipe = LTXConditionPipeline.from_pretrained(
            "Lightricks/LTX-Video-0.9.7-distilled", 
            torch_dtype=torch.bfloat16,
        )
        
        # Initialize upsampler pipeline
        self.pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
            "Lightricks/ltxv-spatial-upscaler-0.9.7",
            vae=self.pipe.vae,
            torch_dtype=torch.bfloat16
        )
        
        # Set up mega cache paths
        mega_cache_dir = CONTAINER_CACHE_DIR / ".mega_cache"
        mega_cache_dir.mkdir(parents=True, exist_ok=True)
        self.mega_cache_bin_path = mega_cache_dir / "ltx_torch_mega"

    @modal.enter(snap=False)
    def setup(self):
        self.pipe.to("cuda")
        self.pipe_upsample.to("cuda")
        self.pipe.vae.enable_tiling()
        
        self._load_mega_cache()
        self._optimize()
        self._compile()
        self._save_mega_cache()

    def round_to_nearest_resolution_acceptable_by_vae(self, height, width):
        height = height - (height % self.pipe.vae_temporal_compression_ratio)
        width = width - (width % self.pipe.vae_temporal_compression_ratio)
        return height, width

    @modal.method()
    def generate(
        self,
        prompt,
        negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
        num_frames=161,
        width=1152,
        height=768,
        guidance_scale=1.0,
        guidance_rescale=0.7,
        decode_timestep=0.05,
        decode_noise_scale=0.025,
        image_cond_noise_scale=0.0,
        downscale_factor=2/3,
        adain_factor=1.0,
        denoise_strength=0.999,
    ):
        from diffusers.utils import export_to_video
        import torch
        # 1. Generate video at smaller resolution
        downscaled_height, downscaled_width = int(height * downscale_factor), int(width * downscale_factor)
        downscaled_height, downscaled_width = self.round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
        
        latents = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=downscaled_width,
            height=downscaled_height,
            num_frames=num_frames,
            timesteps=[1000, 993, 987, 981, 975, 909, 725, 0.03],
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            image_cond_noise_scale=image_cond_noise_scale,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            generator=torch.Generator().manual_seed(0),
            output_type="latent",
        ).frames

        # 2. Upscale generated video using latent upsampler
        upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
        upscaled_latents = self.pipe_upsample(
            latents=latents,
            adain_factor=adain_factor,
            output_type="latent"
        ).frames

        # 3. Denoise the upscaled video with few steps to improve texture
        video = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=upscaled_width,
            height=upscaled_height,
            num_frames=num_frames,
            denoise_strength=denoise_strength,
            timesteps=[1000, 909, 725, 421, 0],
            latents=upscaled_latents,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            image_cond_noise_scale=image_cond_noise_scale,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            generator=torch.Generator().manual_seed(0),
            output_type="pil",
        ).frames[0]

        # 4. Downscale the video to the expected resolution if needed
        # if upscaled_height != height or upscaled_width != width:
        #     video = [frame.resize((width, height)) for frame in video]

        # save to disk using prompt as filename
        mp4_name = slugify(prompt)
        export_to_video(video, Path(OUTPUTS_PATH) / mp4_name, fps=24)
        outputs.commit()
        return mp4_name

    @modal.fastapi_endpoint(method="POST", docs=True)
    def web(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        guidance_rescale: Optional[float] = None,
        decode_timestep: Optional[float] = None,
        decode_noise_scale: Optional[float] = None,
        image_cond_noise_scale: Optional[float] = None,
        downscale_factor: Optional[float] = None,
        adain_factor: Optional[float] = None,
        denoise_strength: Optional[float] = None,
    ):
        import fastapi
        
        mp4_name = self.generate.local(  # run in the same container
            prompt=prompt,
            negative_prompt=negative_prompt or "worst quality, inconsistent motion, blurry, jittery, distorted",
            num_frames=num_frames or 161,
            width=width or 1152,
            height=height or 768,
            guidance_scale=guidance_scale or 1.0,
            guidance_rescale=guidance_rescale or 0.7,
            decode_timestep=decode_timestep or 0.05,
            decode_noise_scale=decode_noise_scale or 0.025,
            image_cond_noise_scale=image_cond_noise_scale or 0.0,
            downscale_factor=downscale_factor or 2/3,
            adain_factor=adain_factor or 1.0,
            denoise_strength=denoise_strength or 0.999,
        )
        return fastapi.responses.FileResponse(
            path=f"{Path(OUTPUTS_PATH) / mp4_name}",
            media_type="video/mp4",
            filename=mp4_name,
        )


# ## Generate videos from the command line

# We trigger LTX-Video inference from our local machine by running the code in
# the local entrypoint below with `modal run`.

# It will spin up a new replica to generate a video.
# Then it will, by default, generate a second video to demonstrate
# the lower latency when hitting a warm container.

# You can trigger inference with:

# ```bash
# modal run ltx
# ```

# All outputs are saved both locally and on a Modal Volume.
# You can explore the contents of Modal Volumes from your Modal Dashboard
# or from the command line with the `modal volume` command.

# ```bash
# modal volume ls ltx-outputs
# ```

# See `modal volume --help` for details.

# Optional command line flags for the script can be viewed with:

# ```bash
# modal run ltx --help
# ```

# Using these flags, you can tweak your generation from the command line:

# ```bash
# modal run --detach ltx --prompt="a cat playing drums in a jazz ensemble" --num-inference-steps=64
# ```


@app.local_entrypoint()
def main(
    prompt: Optional[str] = None,
    negative_prompt="worst quality, blurry, jittery, distorted",
    num_frames: int = 115,
    width: int = 1152,
    height: int = 768,
    guidance_scale: float = 1.0,
    guidance_rescale: float = 0.7,
    decode_timestep: float = 0.05,
    decode_noise_scale: float = 0.025,
    image_cond_noise_scale: float = 0.0,
    downscale_factor: float = 2/3,
    adain_factor: float = 1.0,
    denoise_strength: float = 0.999,
    twice: bool = False,  # Changed default to False since we're doing multiple runs
):
    # Define 10 creative prompts
    prompts = [
        "A majestic eagle soaring through a sunset-lit canyon, wings spread wide against the orange sky",
        "A professional chef preparing sushi in a modern kitchen, with precise knife movements and artistic plating",
    ]

    ltx = LTX()

    def run(current_prompt):
        print(f"🎥 Generating a video from the prompt '{current_prompt}'")
        start = time.time()
        mp4_name = ltx.generate.remote(
            prompt=current_prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            image_cond_noise_scale=image_cond_noise_scale,
            downscale_factor=downscale_factor,
            adain_factor=adain_factor,
            denoise_strength=denoise_strength,
        )
        duration = time.time() - start
        print(f"🎥 Client received video in {int(duration)}s")
        print(f"🎥 LTX video saved to Modal Volume at {mp4_name}")

        local_dir = Path("/tmp/ltx")
        local_dir.mkdir(exist_ok=True, parents=True)
        local_path = local_dir / mp4_name
        local_path.write_bytes(b"".join(outputs.read_file(mp4_name)))
        print(f"🎥 LTX video saved locally at {local_path}")
        print("\n" + "="*80 + "\n")  # Add separator between runs

    # If a specific prompt was provided via command line, use only that one
    if prompt is not None:
        run(prompt)
    else:
        # Run all 10 predefined prompts
        print("🎬 Starting generation of 10 different videos...")
        for i, current_prompt in enumerate(prompts, 1):
            print(f"\n📽 Video {i}/10")
            run(current_prompt)
            
        print("✨ All 10 videos have been generated successfully!")


# ## Addenda

# The remainder of the code in this file is utility code.

DEFAULT_PROMPT = (
    "A woman with short brown hair, wearing a maroon sleeveless top and a silver necklace, "
    "walks through a room while talking, then a woman with pink hair and a white shirt "
    "appears in the doorway and yells. The first woman walks from left to right, her "
    "expression serious; she has light skin and her eyebrows are slightly furrowed. The "
    "second woman stands in the doorway, her mouth open in a yell; she has light skin and "
    "her eyes are wide. The room is dimly lit, with a bookshelf visible in the background. "
    "The camera follows the first woman as she walks, then cuts to a close-up of the second "
    "woman's face. The scene is captured in real-life footage."
)


def slugify(prompt):
    for char in string.punctuation:
        prompt = prompt.replace(char, "")
    prompt = prompt.replace(" ", "_")
    prompt = prompt[:230]  # some OSes limit filenames to <256 chars
    mp4_name = str(int(time.time())) + "_" + prompt + ".mp4"
    return mp4_name
