# Simple ComfyUI example using memory snapshot to speed up cold starts.

# CAUTION: Some custom nodes may not work with memory snapshots, especially if they make calls to torch (i.e. require a GPU) on initialization.
# Run `modal deploy memory_snapshot_example.py` to deploy with memory snapshot enabled.

# Image building and model downloading is directly taken from the core example: https://modal.com/docs/examples/comfyapp
# The notable changes are copying the custom node in the image and the cls object
import subprocess
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .apt_install("libgl1")
    .apt_install("libglib2.0-0")
    .pip_install("fastapi[standard]==0.115.4")
    .pip_install("comfy-cli==1.4.1")
    .pip_install("torch==2.6.0")
    .pip_install("torchvision==0.21.0")
    .pip_install("torchaudio==2.6.0")
    .pip_install("sageattention")
    .run_commands("comfy --skip-prompt install --fast-deps --nvidia")
    .run_commands(  # download the ComfyUI Essentials custom node pack
        "comfy node registry-install comfyui-kjnodes"
    )
    .run_commands(  # download the ComfyUI Essentials custom node pack
        "comfy node registry-install ComfyUI-GGUF"
    )
    .run_commands(  # download the ComfyUI Essentials custom node pack
        "comfy node registry-install rgthree-comfy"
    )
    .run_commands(  # download the ComfyUI Essentials custom node pack
        "comfy node registry-install comfyui-videohelpersuite"
    )
    .run_commands(  # download the ComfyUI Essentials custom node pack
        "comfy node registry-install comfyui-logicutils"
    )
    .run_commands(  # download the ComfyUI Essentials custom node pack
        "comfy node registry-install comfyui_layerstyle"
    )
    .run_commands(  # download the ComfyUI Essentials custom node pack
        "comfy node registry-install ComfyUI-Crystools"
    )
    .run_commands(  # download the ComfyUI Essentials custom node pack
        "git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git /root/comfy/ComfyUI/custom_nodes/SeedVR2_VideoUpscaler",
        "pip install -r /root/comfy/ComfyUI/custom_nodes/SeedVR2_VideoUpscaler/requirements.txt"
    )
)

# Add custom node that patches core ComfyUI so that we can use Modal's [memory snapshot](https://modal.com/docs/guide/memory-snapshot)
image = image.add_local_dir(
    local_path=Path(__file__).parent / "memory_snapshot_helper",
    remote_path="/root/comfy/ComfyUI/custom_nodes/memory_snapshot_helper",
    copy=True,
)


def hf_download():
    import os
    from huggingface_hub import hf_hub_download

    # Create necessary directories
    os.makedirs("/root/comfy/ComfyUI/models/vae", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/text_encoders", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/clip_vision", exist_ok=True)

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = (
    image.pip_install("huggingface_hub[hf_transfer]==0.30.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        hf_download,
        volumes={"/cache": vol},
    )
)

# Lastly, copy the ComfyUI workflow JSON to the container.
image = image.add_local_file(
    Path(__file__).parent / "SeedVR2_VideoUpscale.json", "/root/SeedVR2_VideoUpscale.json"
)

app = modal.App(name="upscaler-seedVR2", image=image)


@app.cls(
    max_containers=1,
    gpu="L40S",
    volumes={"/cache": vol},
    enable_memory_snapshot=True,  # snapshot container state for faster cold starts
)
@modal.concurrent(max_inputs=10)
class UpscalerSeedVR2:
    port: int = 8000

    # Snapshot ComfyUI server launch state, which includes import torch and custom node initialization (GPU not available during this step)
    @modal.enter(snap=True)
    def launch_comfy_background(self):
        cmd = f"comfy launch --background -- --port {self.port}"
        subprocess.run(cmd, shell=True, check=True)

    # Restore ComfyUI server state. Re-enables the CUDA device for inference.
    @modal.enter(snap=False)
    def restore_snapshot(self):
        import requests

        response = requests.post(f"http://127.0.0.1:{self.port}/cuda/set_device")
        if response.status_code != 200:
            print("Failed to set CUDA device")
        else:
            print("Successfully set CUDA device")

    @modal.web_server(port, startup_timeout=60)
    def ui(self):
        subprocess.Popen(
            f"comfy launch -- --listen 0.0.0.0 --port {self.port}", shell=True
        )
