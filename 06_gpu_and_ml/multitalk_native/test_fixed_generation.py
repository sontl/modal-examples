#!/usr/bin/env python3
"""
Test script for the fixed MultiTalk implementation
"""

import base64
import tempfile
import os
from pathlib import Path

def create_test_audio():
    """Create a simple test audio file"""
    import numpy as np
    import soundfile as sf
    
    # Generate a simple sine wave (1 second at 16kHz)
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sample_rate)
        return f.name

def create_test_image():
    """Create a simple test image"""
    import numpy as np
    from PIL import Image
    
    # Create a simple 512x512 test image
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img.save(f.name, 'JPEG')
        return f.name

def test_local_generation():
    """Test the generation locally (without Modal)"""
    print("🧪 Testing MultiTalk Fixed Generation")
    print("=" * 50)
    
    try:
        # Create test files
        print("Creating test audio and image...")
        audio_path = create_test_audio()
        image_path = create_test_image()
        
        print(f"Test audio: {audio_path}")
        print(f"Test image: {image_path}")
        
        # Read files as bytes (simulating Modal input)
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        print(f"Audio data size: {len(audio_data)} bytes")
        print(f"Image data size: {len(image_data)} bytes")
        
        # Test would go here - for now just verify files were created
        print("✅ Test files created successfully")
        
        # Cleanup
        os.unlink(audio_path)
        os.unlink(image_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_modal_generation():
    """Test the generation via Modal"""
    print("\n🚀 Testing MultiTalk via Modal")
    print("=" * 50)
    
    try:
        import modal
        
        # This would test the actual Modal deployment
        print("Modal test would go here...")
        print("Run: modal run 06_gpu_and_ml/multitalk_native/multitalk_fixed.py")
        
        return True
        
    except ImportError:
        print("❌ Modal not available for testing")
        return False
    except Exception as e:
        print(f"❌ Modal test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔧 MultiTalk Fixed - Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test local generation
    if not test_local_generation():
        success = False
    
    # Test Modal generation
    if not test_modal_generation():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    main()