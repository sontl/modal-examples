#!/usr/bin/env python3
"""
Test script to verify quantization files are properly downloaded
"""

import os
from pathlib import Path

def check_quantization_files():
    """Check if all required quantization files exist"""
    
    base_dir = "/data/weights/MeiGen-MultiTalk"
    quant_dir = f"{base_dir}/quant_models"
    
    required_files = [
        f"{quant_dir}/t5_int8.safetensors",
        f"{quant_dir}/quantization_map_fp8_FusionX.json"
    ]
    
    optional_files = [
        f"{quant_dir}/dit_int8.safetensors",
        f"{quant_dir}/vae_int8.safetensors",
        f"/data/weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
    ]
    
    print("Checking required quantization files:")
    all_required_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✓ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ Missing or empty: {file_path}")
            all_required_exist = False
    
    print("\nChecking optional files:")
    for file_path in optional_files:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✓ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"- Not found: {file_path}")
    
    print(f"\nQuantization directory contents:")
    if os.path.exists(quant_dir):
        for item in Path(quant_dir).iterdir():
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.name} ({size_mb:.1f} MB)")
    else:
        print(f"  Directory does not exist: {quant_dir}")
    
    return all_required_exist

if __name__ == "__main__":
    success = check_quantization_files()
    if success:
        print("\n✅ All required quantization files are present!")
    else:
        print("\n❌ Some required quantization files are missing!")
        exit(1)