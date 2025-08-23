#!/usr/bin/env python3
"""
Standalone model download script for ComfyUI Video Upscaler
Can be run separately to pre-download models
"""

import os
import subprocess
from huggingface_hub import hf_hub_download


def download_models():
    """Download all required models for the video upscaler"""
    
    print("🔄 Starting model download process...")
    
    # Create necessary directories
    directories = [
        "/root/comfy/ComfyUI/models/vae",
        "/root/comfy/ComfyUI/models/text_encoders", 
        "/root/comfy/ComfyUI/models/clip_vision",
        "/root/comfy/ComfyUI/models/diffusion_models",
        "/root/comfy/ComfyUI/models/upscale_models",
        "/root/comfy/ComfyUI/models/loras/wanLora"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

    # Model definitions
    models = [
        {
            "name": "Wan2.1 T2V Model",
            "repo_id": "Kijai/WanVideo_comfy",
            "filename": "Wan2_1-T2V-1_3B_bf16.safetensors",
            "local_path": "/root/comfy/ComfyUI/models/diffusion_models/Wan2_1-T2V-1_3B_bf16.safetensors"
        },
        {
            "name": "Wan2.1 VAE",
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/vae/wan_2.1_vae.safetensors",
            "local_path": "/root/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors"
        },
        {
            "name": "UMT5 Text Encoder",
            "repo_id": "city96/umt5-xxl-encoder-gguf",
            "filename": "umt5-xxl-encoder-Q6_K.gguf",
            "local_path": "/root/comfy/ComfyUI/models/text_encoders/umt5-xxl-encoder-Q6_K.gguf"
        },
        {
            "name": "RealESRGAN Upscaler",
            "repo_id": "dtarnow/UPscaler",
            "filename": "RealESRGAN_x2plus.pth",
            "local_path": "/root/comfy/ComfyUI/models/upscale_models/RealESRGAN_x2plus.pth"
        },
        {
            "name": "Wan2.1 LoRA",
            "repo_id": "Kijai/WanVideo_comfy",
            "filename": "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
            "local_path": "/root/comfy/ComfyUI/models/loras/wanLora/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors"
        }
    ]
    
    # Download each model
    for i, model in enumerate(models, 1):
        print(f"\n📦 [{i}/{len(models)}] Downloading {model['name']}...")
        
        if os.path.exists(model["local_path"]):
            print(f"✓ Model already exists: {model['local_path']}")
            continue
            
        try:
            # Download to cache first
            print(f"   Repository: {model['repo_id']}")
            print(f"   File: {model['filename']}")
            
            downloaded_path = hf_hub_download(
                repo_id=model["repo_id"],
                filename=model["filename"],
                cache_dir="/tmp/hf_cache",
                resume_download=True
            )
            
            # Create symlink to expected location
            os.makedirs(os.path.dirname(model["local_path"]), exist_ok=True)
            
            # Remove existing file/link if it exists
            if os.path.exists(model["local_path"]) or os.path.islink(model["local_path"]):
                os.remove(model["local_path"])
            
            # Create symlink
            os.symlink(downloaded_path, model["local_path"])
            
            print(f"✅ Successfully downloaded and linked: {model['name']}")
            
        except Exception as e:
            print(f"❌ Failed to download {model['name']}: {str(e)}")
            raise
    
    print(f"\n🎉 All models downloaded successfully!")
    
    # Print summary
    print("\n📋 Model Summary:")
    for model in models:
        if os.path.exists(model["local_path"]):
            size = os.path.getsize(model["local_path"]) / (1024**3)  # GB
            print(f"   ✓ {model['name']}: {size:.2f} GB")
        else:
            print(f"   ❌ {model['name']}: Missing")


if __name__ == "__main__":
    download_models()