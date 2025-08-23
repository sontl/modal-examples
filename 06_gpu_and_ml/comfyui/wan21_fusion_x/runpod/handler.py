#!/usr/bin/env python3
"""
RunPod Serverless Handler for ComfyUI Video Upscaler
Converts Modal.com implementation to RunPod serverless worker
"""

import json
import subprocess
import uuid
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional
import runpod
from runpod.serverless.utils import rp_cleanup


def download_models():
    """Download required models if not already present"""
    from huggingface_hub import hf_hub_download
    
    print("Checking and downloading models...")
    
    # Create necessary directories
    os.makedirs("/root/comfy/ComfyUI/models/vae", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/text_encoders", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/clip_vision", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/diffusion_models", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/upscale_models", exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/models/loras/wanLora", exist_ok=True)

    models_to_download = [
        {
            "repo_id": "Kijai/WanVideo_comfy",
            "filename": "Wan2_1-T2V-1_3B_bf16.safetensors",
            "local_path": "/root/comfy/ComfyUI/models/diffusion_models/Wan2_1-T2V-1_3B_bf16.safetensors"
        },
        {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/vae/wan_2.1_vae.safetensors",
            "local_path": "/root/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors"
        },
        {
            "repo_id": "city96/umt5-xxl-encoder-gguf",
            "filename": "umt5-xxl-encoder-Q6_K.gguf",
            "local_path": "/root/comfy/ComfyUI/models/text_encoders/umt5-xxl-encoder-Q6_K.gguf"
        },
        {
            "repo_id": "dtarnow/UPscaler",
            "filename": "RealESRGAN_x2plus.pth",
            "local_path": "/root/comfy/ComfyUI/models/upscale_models/RealESRGAN_x2plus.pth"
        },
        {
            "repo_id": "Kijai/WanVideo_comfy",
            "filename": "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
            "local_path": "/root/comfy/ComfyUI/models/loras/wanLora/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors"
        }
    ]
    
    for model in models_to_download:
        if not os.path.exists(model["local_path"]):
            print(f"Downloading {model['filename']}...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=model["repo_id"],
                    filename=model["filename"],
                    cache_dir="/tmp/hf_cache"
                )
                # Create symlink
                os.makedirs(os.path.dirname(model["local_path"]), exist_ok=True)
                if os.path.exists(model["local_path"]):
                    os.remove(model["local_path"])
                os.symlink(downloaded_path, model["local_path"])
                print(f"✓ Downloaded and linked {model['filename']}")
            except Exception as e:
                print(f"✗ Failed to download {model['filename']}: {e}")
        else:
            print(f"✓ Model already exists: {model['filename']}")


def start_comfy_server(port: int = 8188) -> subprocess.Popen:
    """Start ComfyUI server in background"""
    print(f"Starting ComfyUI server on port {port}...")
    
    cmd = f"comfy launch --background -- --port {port}"
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    import requests
    import time
    
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"http://127.0.0.1:{port}/system_stats", timeout=5)
            if response.status_code == 200:
                print("✓ ComfyUI server started successfully")
                return process
        except:
            pass
        
        if i < max_retries - 1:
            print(f"Waiting for ComfyUI server... ({i+1}/{max_retries})")
            time.sleep(2)
    
    raise Exception("Failed to start ComfyUI server")


def check_server_health(port: int = 8188) -> bool:
    """Check if ComfyUI server is healthy"""
    try:
        import requests
        response = requests.get(f"http://127.0.0.1:{port}/system_stats", timeout=5)
        return response.status_code == 200
    except:
        return False


def process_video_upscale(video_bytes: bytes, job_id: str) -> bytes:
    """Process video upscaling using ComfyUI workflow"""
    
    # Ensure server is running
    if not check_server_health():
        raise Exception("ComfyUI server is not healthy")
    
    # Create temporary directories
    temp_dir = f"/tmp/job_{job_id}"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs("/root/comfy/ComfyUI/input", exist_ok=True)
    
    try:
        # Save input video
        input_filename = f"input_{job_id}.mp4"
        input_path = f"/root/comfy/ComfyUI/input/{input_filename}"
        
        with open(input_path, "wb") as f:
            f.write(video_bytes)
        
        print(f"✓ Saved input video: {input_path}")
        
        # Load and modify workflow
        workflow_data = json.loads(Path("/app/VideoUpscalerAPI.json").read_text())
        
        # Update workflow with input video filename
        workflow_data["1479"]["inputs"]["video"] = input_filename
        
        # Set unique output filename
        output_prefix = f"upscaled_{job_id}"
        workflow_data["1435"]["inputs"]["filename_prefix"] = output_prefix
        
        # Save modified workflow
        workflow_path = f"{temp_dir}/workflow.json"
        with open(workflow_path, "w") as f:
            json.dump(workflow_data, f)
        
        print(f"✓ Created workflow: {workflow_path}")
        
        # Run ComfyUI workflow
        print("🚀 Starting video upscaling...")
        cmd = f"comfy run --workflow {workflow_path} --wait --timeout 1200 --verbose"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"ComfyUI workflow failed: {result.stderr}")
        
        print("✓ Video upscaling completed")
        
        # Find output video
        output_dir = Path("/root/comfy/ComfyUI/output")
        output_files = list(output_dir.glob(f"{output_prefix}*"))
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        output_video = None
        
        for file_path in output_files:
            if file_path.suffix.lower() in video_extensions:
                output_video = file_path
                break
        
        if not output_video or not output_video.exists():
            raise Exception(f"Output video not found. Expected prefix: {output_prefix}")
        
        print(f"✓ Found output video: {output_video}")
        
        # Read output video
        with open(output_video, "rb") as f:
            output_bytes = f.read()
        
        print(f"✓ Read output video ({len(output_bytes)} bytes)")
        
        return output_bytes
        
    finally:
        # Cleanup temporary files
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
            rp_cleanup.clean(folder_list=[temp_dir])
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for video upscaling
    
    Expected input:
    {
        "input": {
            "video_url": "https://example.com/video.mp4",  # URL to video file
            # OR
            "video_base64": "base64_encoded_video_data"    # Base64 encoded video
        }
    }
    """
    
    try:
        job_input = job.get("input", {})
        job_id = job.get("id", str(uuid.uuid4()))
        
        print(f"🎬 Processing video upscale job: {job_id}")
        
        # Get video data
        video_bytes = None
        
        if "video_url" in job_input:
            # Download video from URL
            import requests
            video_url = job_input["video_url"]
            print(f"📥 Downloading video from: {video_url}")
            
            response = requests.get(video_url, timeout=300)
            response.raise_for_status()
            video_bytes = response.content
            
        elif "video_base64" in job_input:
            # Decode base64 video
            import base64
            video_base64 = job_input["video_base64"]
            print("📥 Decoding base64 video data")
            video_bytes = base64.b64decode(video_base64)
            
        else:
            return {"error": "Either 'video_url' or 'video_base64' must be provided"}
        
        if not video_bytes:
            return {"error": "Failed to get video data"}
        
        print(f"✓ Got video data ({len(video_bytes)} bytes)")
        
        # Process video upscaling
        upscaled_video_bytes = process_video_upscale(video_bytes, job_id)
        
        # Return base64 encoded result
        import base64
        result_base64 = base64.b64encode(upscaled_video_bytes).decode('utf-8')
        
        return {
            "output": {
                "upscaled_video_base64": result_base64,
                "original_size_bytes": len(video_bytes),
                "upscaled_size_bytes": len(upscaled_video_bytes),
                "job_id": job_id
            }
        }
        
    except Exception as e:
        print(f"❌ Error in handler: {str(e)}")
        return {"error": str(e)}


# Global server process
comfy_server_process = None

def initialize():
    """Initialize the worker - download models and start ComfyUI server"""
    global comfy_server_process
    
    print("🚀 Initializing ComfyUI Video Upscaler...")
    
    try:
        # Download models
        download_models()
        
        # Start ComfyUI server
        comfy_server_process = start_comfy_server()
        
        print("✅ Initialization complete!")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        raise


if __name__ == "__main__":
    # Initialize on startup
    initialize()
    
    # Start RunPod serverless worker
    print("🎯 Starting RunPod serverless worker...")
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True
    })