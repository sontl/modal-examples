#!/usr/bin/env python3
"""
Patch script to handle import issues in MultiTalk
"""

import os
import sys

def patch_kokoro_imports():
    """Patch kokoro imports to handle missing dependencies"""
    
    # Path to the kokoro pipeline file
    pipeline_file = "/app/kokoro/pipeline.py"
    
    if not os.path.exists(pipeline_file):
        print("Kokoro pipeline file not found, skipping patch")
        return
    
    # Read the original file
    with open(pipeline_file, 'r') as f:
        content = f.read()
    
    # Replace the problematic import
    original_import = "from misaki import en, espeak"
    patched_import = """
try:
    from misaki import en, espeak
except ImportError:
    print("Warning: misaki not available, TTS features disabled")
    en = None
    espeak = None
"""
    
    if original_import in content:
        content = content.replace(original_import, patched_import)
        
        # Write the patched file
        with open(pipeline_file, 'w') as f:
            f.write(content)
        
        print("Successfully patched kokoro pipeline imports")
    else:
        print("Import pattern not found, no patch needed")

def patch_generate_multitalk():
    """Patch the main generation script to handle TTS imports gracefully"""
    
    generate_file = "/app/generate_multitalk.py"
    
    if not os.path.exists(generate_file):
        print("generate_multitalk.py not found, skipping patch")
        return
    
    # Read the original file
    with open(generate_file, 'r') as f:
        content = f.read()
    
    # Replace the kokoro import
    original_import = "from kokoro import KPipeline"
    patched_import = """
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Kokoro TTS not available: {e}")
    KPipeline = None
    KOKORO_AVAILABLE = False
"""
    
    if original_import in content:
        content = content.replace(original_import, patched_import)
        
        # Also patch any usage of KPipeline
        content = content.replace(
            "tts_pipeline = KPipeline(",
            "tts_pipeline = KPipeline( if KOKORO_AVAILABLE else None"
        )
        
        # Write the patched file
        with open(generate_file, 'w') as f:
            f.write(content)
        
        print("Successfully patched generate_multitalk.py imports")
    else:
        print("Generate script import pattern not found, no patch needed")

if __name__ == "__main__":
    print("Applying MultiTalk import patches...")
    patch_kokoro_imports()
    patch_generate_multitalk()
    print("Patch application complete")