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
from typing import Optional

import modal

app = modal.App("example-ltx-video")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        "pip install git+https://github.com/huggingface/diffusers.git",
    )
    .pip_install(
        "accelerate",
        "hf_transfer",
        "imageio",
        "imageio-ffmpeg",
        "sentencepiece",
        "torch",
        "transformers",
    )
    
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

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

# We don't have to change any of the Hugging Face code to do this --
# we just set the location of Hugging Face's cache to be on a Volume
# using the `HF_HOME` environment variable.

MODEL_PATH = Path("/models")
image = image.env({"HF_HOME": str(MODEL_PATH)})

# For more on storing Modal weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).


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
    volumes={OUTPUTS_PATH: outputs, MODEL_PATH: model},
    gpu="H100",
    timeout=10 * MINUTES,
    scaledown_window=15 * MINUTES,
)
class LTX:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline, AutoModel
        from diffusers.hooks import apply_group_offloading

        # transformer = AutoModel.from_pretrained(
        #     "Lightricks/LTX-Video-0.9.7-distilled",
        #     subfolder="transformer",
        #     torch_dtype=torch.bfloat16
        # )
        # transformer.enable_layerwise_casting(
        #     storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
        # )

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

        # Configure group-offloading for memory optimization
        # onload_device = torch.device("cuda")
        # offload_device = torch.device("cpu")
        
        # Apply group offloading to different components
                # Apply group offloading to different components
        # self.pipe.transformer.enable_group_offload(
        #     onload_device=onload_device, 
        #     offload_device=offload_device, 
        #     offload_type="leaf_level", 
        #     use_stream=True
        # )
        # apply_group_offloading(
        #     self.pipe.text_encoder, 
        #     onload_device=onload_device, 
        #     offload_type="block_level", 
        #     num_blocks_per_group=2
        # )
        # apply_group_offloading(
        #     self.pipe.vae, 
        #     onload_device=onload_device, 
        #     offload_type="leaf_level"
        # )
        
        self.pipe.to("cuda")
        self.pipe_upsample.to("cuda")
        self.pipe.vae.enable_tiling()

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
        "A street performer doing an acrobatic dance routine in Times Square, surrounded by amazed onlookers",
        "A young artist painting a vibrant mural on a city wall, with colorful paint splatters and dynamic brushstrokes",
        "A group of dolphins jumping through ocean waves at sunrise, creating beautiful water splashes",
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
