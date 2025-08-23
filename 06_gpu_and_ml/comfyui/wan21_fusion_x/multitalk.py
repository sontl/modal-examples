# Simple ComfyUI example using memory snapshot to speed up cold starts.
# This script builds a Modal image for running ComfyUI with a set of custom nodes and models,
# leveraging Modal's memory snapshot feature for fast cold starts.

# CAUTION: Some custom nodes may not work with memory snapshots, especially if they make calls to torch (i.e. require a GPU) on initialization.
# Run `modal deploy memory_snapshot_example.py` to deploy with memory snapshot enabled.

# Image building and model downloading is directly taken from the core example: https://modal.com/docs/examples/comfyapp
# The notable changes are copying the custom node in the image and the cls object

import subprocess
from pathlib import Path

import modal

# Build the Modal image with all required system and Python dependencies,
# as well as custom ComfyUI nodes and helper scripts.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Required for cloning repositories
    .apt_install("libgl1")  # Required for OpenCV and some image processing
    .apt_install("libglib2.0-0")  # Required for OpenCV and some image processing
    .pip_install("fastapi[standard]==0.115.4")  # FastAPI for serving APIs
    .pip_install("comfy-cli==1.4.1")  # ComfyUI core
    .pip_install("sageattention")  # Additional dependency for custom nodes
    .run_commands("comfy --skip-prompt install --fast-deps --nvidia")  # Pre-install ComfyUI dependencies
    # Install various ComfyUI custom node packs via the registry
    .run_commands("comfy node registry-install comfyui-kjnodes")
    .run_commands("comfy node registry-install ComfyUI-GGUF")
    .run_commands("comfy node registry-install rgthree-comfy")
    .run_commands("comfy node registry-install comfyui-videohelpersuite")
    .run_commands("comfy node registry-install comfyui-logicutils")
    .run_commands("comfy node registry-install audio-separation-nodes-comfyui")
    # Clone and install requirements for the WanVideoWrapper custom node
    .run_commands(
        "git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git "
        "/root/comfy/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper"
        " && cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper"
        " && pip install -r requirements.txt"
    )
)

# Add a custom node that patches core ComfyUI so that we can use Modal's memory snapshot feature.
image = image.add_local_dir(
    local_path=Path(__file__).parent / "memory_snapshot_helper",
    remote_path="/root/comfy/ComfyUI/custom_nodes/memory_snapshot_helper",
    copy=True,
)

def hf_download():
    """
    Download all required model files from HuggingFace Hub and create symlinks
    in the appropriate ComfyUI model directories.
    """
    import os
    from huggingface_hub import hf_hub_download

    # Create necessary directories for different model types
    os.makedirs("/root/comfy/ComfyUI/models/diffusion_models/fusion", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/vae", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/text_encoders", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/clip_vision", exist_ok=True)

    wan_fusion_x_model = hf_hub_download(
        repo_id="vrgamedevgirl84/Wan14BT2VFusioniX",
        filename="Wan14BT2VFusioniX_fp16_.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {wan_fusion_x_model} /root/comfy/ComfyUI/models/diffusion_models/fusion/Wan14BT2VFusioniX_fp16_.safetensors",
        shell=True,
        check=True,
    )
    
    # Download the main Multitalk diffusion model
    multitalk_model = hf_hub_download(
        repo_id="Kijai/WanVideo_comfy",
        filename="WanVideo_2_1_Multitalk_14B_fp8_e4m3fn.safetensors",
        cache_dir="/cache",
    )
    # Symlink to ComfyUI's diffusion_models directory
    subprocess.run(
        f"ln -s {multitalk_model} /root/comfy/ComfyUI/models/diffusion_models/WanVideo_2_1_Multitalk_14B_fp8_e4m3fn.safetensors",
        shell=True,
        check=True,
    )

    meigen_multitalk_model = hf_hub_download(
        repo_id="MeiGen-AI/MeiGen-MultiTalk",
        filename="multitalk.safetensors",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -s {meigen_multitalk_model} /root/comfy/ComfyUI/models/diffusion_models/multitalk.safetensors",
        shell=True,
        check=True,
    )

    # Download the 480p I2V model
    wan_480p_i2v_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {wan_480p_i2v_model} /root/comfy/ComfyUI/models/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors",
        shell=True,
        check=True,
    ) 

    wan_480p_i2v_fp8_model = hf_hub_download(
        repo_id="Kijai/WanVideo_comfy",
        filename="Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -s {wan_480p_i2v_fp8_model} /root/comfy/ComfyUI/models/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
        shell=True,
        check=True,
    )
    
    
    # Download the text encoder model
    text_encoder_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/text_encoders/umt5_xxl_fp16.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {text_encoder_model} /root/comfy/ComfyUI/models/text_encoders/native_umt5_xxl_fp16.safetensors",
        shell=True,
        check=True,
    )

    # Download the T2V fp16 model
    wan_fp16_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {wan_fp16_model} /root/comfy/ComfyUI/models/diffusion_models/wan2.1_t2v_14B_fp16.safetensors",
        shell=True,
        check=True,
    )

    # Download the T2V fp8 model
    wan_fp8_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/diffusion_models/wan2.1_t2v_14B_fp8_e4m3fn.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {wan_fp8_model} /root/comfy/ComfyUI/models/diffusion_models/wan2.1_t2v_14B_fp8_e4m3fn.safetensors",
        shell=True,
        check=True,
    )

    wan_fp8_i2v_model = hf_hub_download(
        repo_id="Kijai/WanVideo_comfy",
        filename="Wan2_1-I2V-14B-480P_fp8_e5m2.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {wan_fp8_i2v_model} /root/comfy/ComfyUI/models/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e5m2.safetensors",
        shell=True,
        check=True,
    )
    
    

    # Download the VAE model
    wan_vae_p32_model = hf_hub_download(
        repo_id="Kijai/WanVideo_comfy",
        filename="Wan2_1_VAE_fp32.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {wan_vae_p32_model} /root/comfy/ComfyUI/models/vae/Wan2_1_VAE_fp32.safetensors",
        shell=True,
        check=True,
    )

    wan_vae_p16_model = hf_hub_download(
        repo_id="Kijai/WanVideo_comfy",
        filename="Wan2_1_VAE_bf16.safetensors",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -s {wan_vae_p16_model} /root/comfy/ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors",
        shell=True,
        check=True,
    )


    # Download the CLIP vision model
    wan_clip_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/clip_vision/clip_vision_h.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {wan_clip_model} /root/comfy/ComfyUI/models/clip_vision/native_clip_vision_h.safetensors",
        shell=True,
        check=True,
    )

    # Download the LoRA model
    lora_model = hf_hub_download(
        repo_id="Kijai/WanVideo_comfy",
        filename="Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {lora_model} /root/comfy/ComfyUI/models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors",
        shell=True,
        check=True,
    )

# Create or connect to a Modal volume for caching HuggingFace downloads
vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

# Add HuggingFace Hub and enable fast transfer, then run the model download function in the image build
image = (
    image.pip_install("huggingface_hub[hf_transfer]==0.30.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        hf_download,
        volumes={"/cache": vol},
    )
)

# Lastly, copy the ComfyUI workflow JSON to the container.
# This workflow file defines the ComfyUI workflow to be used by the API.
image = image.add_local_file(
    Path(__file__).parent / "multitalk_workflow_api.json", "/root/workflow_api.json"
)

# Create the Modal app with the built image
app = modal.App(name="multitalk", image=image)

# Define the Modal class for the ComfyUI server, with memory snapshot enabled for fast cold starts.
@app.cls(
    max_containers=1,  # Only allow one container at a time for this app
    gpu="L40S",        # Specify the GPU type
    volumes={"/cache": vol},  # Mount the HuggingFace cache volume
    enable_memory_snapshot=True,  # snapshot container state for faster cold starts
)
@modal.concurrent(max_inputs=10)  # Allow up to 10 concurrent requests
class MultiTalk:
    port: int = 8000  # Port for the ComfyUI server

    # Snapshot ComfyUI server launch state, which includes import torch and custom node initialization (GPU not available during this step)
    @modal.enter(snap=True)
    def launch_comfy_background(self):
        # Launch ComfyUI in background mode (no listening on 0.0.0.0 yet)
        cmd = f"comfy launch --background -- --port {self.port}"
        subprocess.run(cmd, shell=True, check=True)

    # Restore ComfyUI server state. Re-enables the CUDA device for inference.
    @modal.enter(snap=False)
    def restore_snapshot(self):
        # After restoring the memory snapshot, re-enable CUDA for inference
        import requests

        response = requests.post(f"http://127.0.0.1:{self.port}/cuda/set_device")
        if response.status_code != 200:
            print("Failed to set CUDA device")
        else:
            print("Successfully set CUDA device")

    @modal.web_server(port, startup_timeout=60)
    def ui(self):
        # Start the ComfyUI server in web server mode, listening on all interfaces
        subprocess.Popen(
            f"comfy launch -- --listen 0.0.0.0 --port {self.port}", shell=True
        )
