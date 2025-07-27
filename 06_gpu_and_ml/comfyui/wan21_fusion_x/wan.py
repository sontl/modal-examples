# Simple ComfyUI example using memory snapshot to speed up cold starts.

# CAUTION: Some custom nodes may not work with memory snapshots, especially if they make calls to torch (i.e. require a GPU) on initialization.
# Run `modal deploy memory_snapshot_example.py` to deploy with memory snapshot enabled.

# Image building and model downloading is directly taken from the core example: https://modal.com/docs/examples/comfyapp
# The notable changes are copying the custom node in the image and the cls object
import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .apt_install("libgl1")
    .apt_install("libglib2.0-0")
    .pip_install("fastapi[standard]==0.115.4")
    .pip_install("comfy-cli==1.4.1")
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
        "comfy node registry-install comfyui-multigpu"
    )
    .run_commands(
        "comfy --skip-prompt model download --url  https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/resolve/main/OtherLoRa%27s/Wan14B_RealismBoost.safetensors?download=true --relative-path models/loras",
    ).run_commands(
        "comfy --skip-prompt model download --url  https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/resolve/main/OtherLoRa%27s/DetailEnhancerV1.safetensors?download=true --relative-path models/loras",
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
    os.makedirs("/root/comfy/ComfyUI/models/diffusion_models/fusion", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/vae", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/text_encoders", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/clip_vision", exist_ok=True)

    wan_fusion_x_model = hf_hub_download(
        repo_id="QuantStack/Wan2.1_I2V_14B_FusionX-GGUF",
        filename="Wan2.1_I2V_14B_FusionX-Q8_0.gguf",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -s {wan_fusion_x_model} /root/comfy/ComfyUI/models/diffusion_models/fusion/Wan2.1_I2V_14B_FusionX-Q8_0.gguf",
        shell=True,
        check=True,
    )

    wan_vae_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/vae/wan_2.1_vae.safetensors",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -s {wan_vae_model} /root/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors",
        shell=True,
        check=True,
    )

    wan_text_encoder_model = hf_hub_download(
        repo_id="city96/umt5-xxl-encoder-gguf",
        filename="umt5-xxl-encoder-Q8_0.gguf",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -s {wan_text_encoder_model} /root/comfy/ComfyUI/models/text_encoders/umt5-xxl-encoder-Q8_0.gguf",
        shell=True,
        check=True,
    )

    wan_clip_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/clip_vision/clip_vision_h.safetensors",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -s {wan_clip_model} /root/comfy/ComfyUI/models/clip_vision/clip_vision_h.safetensors",
        shell=True,
        check=True,
    )
    
    # download LoRA models
    os.makedirs("/root/comfy/ComfyUI/models/loras", exist_ok=True)
    lora_model = hf_hub_download(
        repo_id="Kijai/WanVideo_comfy",
        filename="Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -s {lora_model} /root/comfy/ComfyUI/models/loras/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
        shell=True,
        check=True,
    )

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

    lora_model = hf_hub_download(
        repo_id="alibaba-pai/Wan2.1-Fun-Reward-LoRAs",
        filename="Wan2.1-Fun-14B-InP-MPS.safetensors",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -s {lora_model} /root/comfy/ComfyUI/models/loras/Wan2.1-Fun-14B-InP-MPS.safetensors",
        shell=True,
        check=True,
    )

    lora_model = hf_hub_download(
        repo_id="Kijai/WanVideo_comfy",
        filename="Wan21_AccVid_I2V_480P_14B_lora_rank32_fp16.safetensors",
        cache_dir="/cache",
    )
    
    subprocess.run(
        f"ln -s {lora_model} /root/comfy/ComfyUI/models/loras/Wan21_AccVid_I2V_480P_14B_lora_rank32_fp16.safetensors",
        shell=True,
        check=True,
    )

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
    Path(__file__).parent / "wan_workflow_api.json", "/root/workflow_api.json"
)

# Add the WanFusionX API workflow
image = image.add_local_file(
    Path(__file__).parent / "WanFusionXAPI.json", "/root/WanFusionXAPI.json"
)

app = modal.App(name="wan21-fusion-x", image=image)


@app.cls(
    max_containers=1,
    gpu="A10G",
    volumes={"/cache": vol},
    enable_memory_snapshot=True,  # snapshot container state for faster cold starts
)
@modal.concurrent(max_inputs=10)
class WanFusionX:
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

    @modal.method()
    def infer(self, workflow_path: str = "/root/WanFusionXAPI.json"):
        # sometimes the ComfyUI server stops responding (we think because of memory leaks), so this makes sure it's still up
        self.poll_server_health()

        # runs the comfy run --workflow command as a subprocess
        cmd = f"comfy run --workflow {workflow_path} --wait --timeout 1200 --verbose"
        subprocess.run(cmd, shell=True, check=True)

        # completed workflows write output images to this directory
        output_dir = "/root/comfy/ComfyUI/output"

        # looks up the name of the output image file based on the workflow
        workflow = json.loads(Path(workflow_path).read_text())
        file_prefix = [
            node.get("inputs")
            for node in workflow.values()
            if node.get("class_type") == "VHS_VideoCombine"
        ][0]["filename_prefix"]

        # returns the video as bytes
        for f in Path(output_dir).iterdir():
            if f.name.startswith(file_prefix) and f.suffix in ['.mp4', '.avi', '.mov']:
                return f.read_bytes()
        
        # If no video file found, raise an error
        raise FileNotFoundError(f"No video output found with prefix: {file_prefix}")

    @modal.method()
    def process_image_bytes(self, image: bytes, prompt: str = None):
        """Process image bytes and return generated video bytes"""
        # Save uploaded image to ComfyUI input directory
        client_id = uuid.uuid4().hex
        image_filename = f"input_{client_id}.png"
        input_dir = Path("/root/comfy/ComfyUI/input")
        input_dir.mkdir(exist_ok=True)
        
        image_path = input_dir / image_filename
        image_path.write_bytes(image)

        # Load the workflow template
        workflow_data = json.loads(Path("/root/WanFusionXAPI.json").read_text())

        # Update the image input
        workflow_data["52"]["inputs"]["image"] = image_filename

        # Update the prompt if provided
        if prompt:
            workflow_data["6"]["inputs"]["text"] = prompt

        # Give the output video a unique id per client request
        workflow_data["30"]["inputs"]["filename_prefix"] = f"FusionXi2v/FusionX_{client_id}"

        # Save this updated workflow to a new file
        new_workflow_file = f"/tmp/{client_id}.json"
        with open(new_workflow_file, "w") as f:
            json.dump(workflow_data, f)

        # Run inference on the currently running container
        video_bytes = self.infer.local(new_workflow_file)

        # Clean up input file
        image_path.unlink(missing_ok=True)

        return video_bytes

    @modal.fastapi_endpoint(method="POST", label="image-to-video")
    def image_to_video(self, image: bytes, prompt: str = "The boys eyes glow and colored musical notes can be seen in the reflection."):
        from fastapi import Response

        # Process the image using the local method
        video_bytes = self.process_image_bytes.local(image, prompt)
        return Response(video_bytes, media_type="video/mp4")

    def poll_server_health(self) -> Dict:
        import socket
        import urllib

        try:
            # check if the server is up (response should be immediate)
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/system_stats")
            urllib.request.urlopen(req, timeout=5)
            print("ComfyUI server is healthy")
        except (socket.timeout, urllib.error.URLError) as e:
            # if no response in 5 seconds, stop the container
            print(f"Server health check failed: {str(e)}")
            modal.experimental.stop_fetching_inputs()

            # all queued inputs will be marked "Failed", so you need to catch these errors in your client and then retry
            raise Exception("ComfyUI server is not healthy, stopping container")

    @modal.web_server(port, startup_timeout=60)
    def ui(self):
        subprocess.Popen(
            f"comfy launch -- --listen 0.0.0.0 --port {self.port}", shell=True
        )
