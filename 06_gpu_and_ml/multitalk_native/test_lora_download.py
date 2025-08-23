#!/usr/bin/env python3
"""
Test script to verify LoRA files are properly downloaded
"""

import os
from pathlib import Path

def check_lora_files():
    """Check if LoRA files exist"""
    
    lora_file = "/data/weights/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
    quant_map_file = "/data/weights/FusionX_LoRa/quantization_map_int8_FusionX.json"
    
    print("Checking LoRA files:")
    
    all_good = True
    
    # Check LoRA weights file
    if os.path.exists(lora_file) and os.path.getsize(lora_file) > 0:
        size_mb = os.path.getsize(lora_file) / (1024 * 1024)
        print(f"✓ {lora_file} ({size_mb:.1f} MB)")
    else:
        print(f"✗ Missing or empty: {lora_file}")
        all_good = False
        
        # Check if it exists in the old location
        old_lora_file = "/data/weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
        if os.path.exists(old_lora_file):
            size_mb = os.path.getsize(old_lora_file) / (1024 * 1024)
            print(f"! Found in old location: {old_lora_file} ({size_mb:.1f} MB)")
            print("  Consider moving to new location or updating path references")
    
    # Check quantization map file
    if os.path.exists(quant_map_file) and os.path.getsize(quant_map_file) > 0:
        size_kb = os.path.getsize(quant_map_file) / 1024
        print(f"✓ {quant_map_file} ({size_kb:.1f} KB)")
    else:
        print(f"✗ Missing or empty: {quant_map_file}")
        all_good = False
        
    return all_good

def list_weights_directory():
    """List contents of weights directory"""
    weights_dir = "/data/weights"
    
    print(f"\nContents of {weights_dir}:")
    if os.path.exists(weights_dir):
        for item in Path(weights_dir).rglob("*LoRA*"):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item} ({size_mb:.1f} MB)")
    else:
        print(f"  Directory does not exist: {weights_dir}")

if __name__ == "__main__":
    success = check_lora_files()
    list_weights_directory()
    
    if success:
        print("\n✅ LoRA files are present!")
    else:
        print("\n❌ LoRA files are missing!")
        exit(1)