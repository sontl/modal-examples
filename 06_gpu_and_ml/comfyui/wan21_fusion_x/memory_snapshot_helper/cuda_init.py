#!/usr/bin/env python3
"""
CUDA initialization helper for ComfyUI workflows
This ensures CUDA is properly initialized before workflow execution
"""

import os
import sys
import torch
import logging

def ensure_cuda_initialized():
    """Ensure CUDA is properly initialized and available"""
    try:
        # Set CUDA environment
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        # Force CUDA initialization
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.empty_cache()
            
            # Test CUDA functionality
            test_tensor = torch.tensor([1.0]).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            
            print(f"[cuda_init] CUDA initialized successfully: {torch.cuda.device_count()} devices")
            return True
        else:
            print("[cuda_init] WARNING: CUDA not available")
            return False
            
    except Exception as e:
        print(f"[cuda_init] ERROR: Failed to initialize CUDA: {e}")
        return False

if __name__ == "__main__":
    ensure_cuda_initialized()