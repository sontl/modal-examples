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
    .pip_install(
        "accelerate==1.6.0",
        "diffusers==0.33.1",
        "hf_transfer==0.1.9",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.5.1",
        "sentencepiece==0.2.0",
        "torch==2.7.0",
        "transformers==4.51.3",
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
    image=image,  # use our container Image
    volumes={OUTPUTS_PATH: outputs, MODEL_PATH: model},  # attach our Volumes
    gpu="L4",  # use a big, fast GPU
    timeout=10 * MINUTES,  # run inference for up to 10 minutes
    scaledown_window=15 * MINUTES,  # stay idle for 15 minutes before scaling down
)
class LTX:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import LTXPipeline, AutoModel
        from diffusers.hooks import apply_group_offloading
        
        # Load transformer with FP8 layerwise weight-casting
        transformer = AutoModel.from_pretrained(
            "Lightricks/LTX-Video",
            subfolder="transformer",
            torch_dtype=torch.bfloat16
        )
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
        )

        # Initialize pipeline with optimized transformer
        self.pipe = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video", 
            transformer=transformer, 
            torch_dtype=torch.bfloat16
        )

        # Configure group-offloading for memory optimization
        onload_device = torch.device("cuda")
        offload_device = torch.device("cpu")
        
        # Apply group offloading to different components
        self.pipe.transformer.enable_group_offload(
            onload_device=onload_device, 
            offload_device=offload_device, 
            offload_type="leaf_level", 
            use_stream=True
        )
        apply_group_offloading(
            self.pipe.text_encoder, 
            onload_device=onload_device, 
            offload_type="block_level", 
            num_blocks_per_group=2
        )
        apply_group_offloading(
            self.pipe.vae, 
            onload_device=onload_device, 
            offload_type="leaf_level"
        )
        
        self.pipe.to("cuda")

    @modal.method()
    def generate(
        self,
        prompt,
        negative_prompt="",
        num_inference_steps=200,
        guidance_scale=4.5,
        num_frames=19,
        width=704,
        height=480,
    ):
        from diffusers.utils import export_to_video

        frames = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            width=width,
            height=height,
        ).frames[0]

        # save to disk using prompt as filename
        mp4_name = slugify(prompt)
        export_to_video(frames, Path(OUTPUTS_PATH) / mp4_name)
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
    num_inference_steps: int = 10,  # 10 when testing, 100 or more when generating
    guidance_scale: float = 2.5,
    num_frames: int = 150,  # produces ~10s of video
    width: int = 704,
    height: int = 480,
    twice: bool = True,  # run twice to show cold start latency
):
    if prompt is None:
        prompt = DEFAULT_PROMPT

    ltx = LTX()

    def run():
        print(f"🎥 Generating a video from the prompt '{prompt}'")
        start = time.time()
        mp4_name = ltx.generate.remote(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            width=width,
            height=height,
        )
        duration = time.time() - start
        print(f"🎥 Client received video in {int(duration)}s")
        print(f"🎥 LTX video saved to Modal Volume at {mp4_name}")

        local_dir = Path("/tmp/ltx")
        local_dir.mkdir(exist_ok=True, parents=True)
        local_path = local_dir / mp4_name
        local_path.write_bytes(b"".join(outputs.read_file(mp4_name)))
        print(f"🎥 LTX video saved locally at {local_path}")

    run()

    if twice:
        print("🎥 Generating a video from a warm container")
        run()


# ## Addenda

# The remainder of the code in this file is utility code.

DEFAULT_PROMPT = (
    "A woman with long brown hair and light skin smiles at another woman with long blonde hair."
    "The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek."
    "The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and "
    "natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
)


def slugify(prompt):
    for char in string.punctuation:
        prompt = prompt.replace(char, "")
    prompt = prompt.replace(" ", "_")
    prompt = prompt[:230]  # some OSes limit filenames to <256 chars
    mp4_name = str(int(time.time())) + "_" + prompt + ".mp4"
    return mp4_name
