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
        "git clone https://github.com/cubiq/ComfyUI_essentials.git /root/comfy/ComfyUI/custom_nodes/ComfyUI_essentials"
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
    os.makedirs("/root/comfy/ComfyUI/models/diffusion_models", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/upscale_models", exist_ok=True)

    wan_1_3_B_model = hf_hub_download(
        repo_id="Kijai/WanVideo_comfy",
        filename="Wan2_1-T2V-1_3B_bf16.safetensors",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -s {wan_1_3_B_model} /root/comfy/ComfyUI/models/diffusion_models/Wan2_1-T2V-1_3B_bf16.safetensors",
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
        filename="umt5-xxl-encoder-Q6_K.gguf",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -s {wan_text_encoder_model} /root/comfy/ComfyUI/models/text_encoders/umt5-xxl-encoder-Q6_K.gguf",
        shell=True,
        check=True,
    )

    wan_clip_model = hf_hub_download(
        repo_id="dtarnow/UPscaler",
        filename="RealESRGAN_x2plus.pth",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -s {wan_clip_model} /root/comfy/ComfyUI/models/upscale_models/RealESRGAN_x2plus.pth",
        shell=True,
        check=True,
    )
    
    # download LoRA models - create wanLora subdirectory to match workflow expectations
    os.makedirs("/root/comfy/ComfyUI/models/loras/wanLora", exist_ok=True)
    
    lora_model = hf_hub_download(
        repo_id="Kijai/WanVideo_comfy",
        filename="Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
        cache_dir="/cache",
    )

    subprocess.run(
        f"ln -s {lora_model} /root/comfy/ComfyUI/models/loras/wanLora/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
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
    Path(__file__).parent / "VideoUpscalerAPI.json", "/root/VideoUpscalerAPI.json", copy=True
)

app = modal.App(name="upscaler-wan", image=image)


@app.cls(
    min_containers=0,
    max_containers=1, 
    gpu="L4",
    volumes={"/cache": vol},
    timeout=3600,
    scaledown_window=60 * 2,  # 2 minutes
    enable_memory_snapshot=True,  # snapshot container state for faster cold starts
)
@modal.concurrent(max_inputs=4)
class UpscalerWan:
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
    def infer(self, workflow_path: str = "/root/VideoUpscalerAPI.json"):
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

    # @modal.fastapi_endpoint(method="POST")
    # def api(self, video_filename: str):
    #     from fastapi import Response

    #     # Load the workflow template
    #     workflow_data = json.loads(Path("/root/VideoUpscalerAPI.json").read_text())

    #     # Update the video input - expecting video filename in the request
    #     workflow_data["1479"]["inputs"]["video"] = video_filename

    #     # Give the output video a unique id per client request
    #     client_id = uuid.uuid4().hex
    #     workflow_data["1435"]["inputs"]["filename_prefix"] = f"upscaled_{client_id}"

    #     # Save this updated workflow to a new file
    #     new_workflow_file = f"/tmp/{client_id}.json"
    #     with open(new_workflow_file, "w") as f:
    #         json.dump(workflow_data, f)

    #     # Run inference on the currently running container
    #     video_bytes = self.infer.local(new_workflow_file)

    #     return Response(video_bytes, media_type="video/mp4")

    @modal.method()
    def process_video_bytes(self, video: bytes):
        """Process video bytes and return upscaled video bytes"""
        # Save uploaded video to ComfyUI input directory
        client_id = uuid.uuid4().hex
        video_filename = f"input_{client_id}.mp4"
        input_dir = Path("/root/comfy/ComfyUI/input")
        input_dir.mkdir(exist_ok=True)
        
        video_path = input_dir / video_filename
        video_path.write_bytes(video)

        # Load the workflow template
        workflow_data = json.loads(Path("/root/VideoUpscalerAPI.json").read_text())

        # Update the video input
        workflow_data["1479"]["inputs"]["video"] = video_filename

        # Give the output video a unique id per client request
        workflow_data["1435"]["inputs"]["filename_prefix"] = f"upscaled_{client_id}"

        # Save this updated workflow to a new file
        new_workflow_file = f"/tmp/{client_id}.json"
        with open(new_workflow_file, "w") as f:
            json.dump(workflow_data, f)

        # Run inference on the currently running container
        video_bytes = self.infer.local(new_workflow_file)

        # Clean up input file
        video_path.unlink(missing_ok=True)

        return video_bytes

    @modal.fastapi_endpoint(method="POST", label="upload-and-upscale")
    def upload_and_upscale(self, video: bytes):
        from fastapi import Response

        # Process the video using the local method
        video_bytes = self.process_video_bytes.local(video)
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


# Create a separate FastAPI app for file uploads
@app.function(
    image=image.pip_install("python-multipart"),
    volumes={"/cache": vol},
    min_containers=0,
    scaledown_window=60 * 5,
    timeout=60 * 60,
    # gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow it to scale to 100 concurrent inputs
    max_containers=1,
)
@modal.concurrent(max_inputs=1000)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import Response
    import tempfile

    web_app = FastAPI(title="Video Upscaler API", docs_url="/docs")

    @web_app.post("/upscale")
    async def upscale_video(video: UploadFile = File(...)):
        """Upload a video file and get back an upscaled version"""
        
        # Validate file type
        if not video.content_type or not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Read video content
        video_content = await video.read()
        
        # Call the upscaler using the remote method
        upscaler = UpscalerWan()
        result_bytes = upscaler.process_video_bytes.remote(video_content)
        
        return Response(
            content=result_bytes,
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename=upscaled_{video.filename}"}
        )

    @web_app.get("/")
    def root():
        return {"message": "Video Upscaler API - Use POST /upscale to upload and upscale videos"}

    return web_app
