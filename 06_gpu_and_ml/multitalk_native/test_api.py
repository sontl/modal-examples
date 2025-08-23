#!/usr/bin/env python3
"""
Test script for MultiTalk Fixed FastAPI endpoints
"""

import requests
import base64
import json
from pathlib import Path

# Test configuration
API_BASE_URL = "https://your-modal-app-url.modal.run"  # Replace with actual URL
TEST_AUDIO_PATH = "test_audio.wav"  # Path to test audio file
TEST_IMAGE_PATH = "test_image.jpg"  # Path to test image file

def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Health check passed: {data}")
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_generate_endpoint_binary():
    """Test the /generate endpoint that returns binary video"""
    print("Testing /generate endpoint (binary response)...")
    
    if not Path(TEST_AUDIO_PATH).exists() or not Path(TEST_IMAGE_PATH).exists():
        print(f"❌ Test files not found: {TEST_AUDIO_PATH}, {TEST_IMAGE_PATH}")
        return False
    
    try:
        # Prepare files
        files = {
            'audio': ('test_audio.wav', open(TEST_AUDIO_PATH, 'rb'), 'audio/wav'),
            'image': ('test_image.jpg', open(TEST_IMAGE_PATH, 'rb'), 'image/jpeg')
        }
        
        # Prepare form data
        data = {
            'prompt': 'A person speaking clearly and naturally',
            'sample_steps': 30,
            'use_teacache': True,
            'low_vram': False
        }
        
        print("Sending request...")
        response = requests.post(
            f"{API_BASE_URL}/generate",
            files=files,
            data=data,
            timeout=1800  # 30 minutes
        )
        
        # Close files
        files['audio'][1].close()
        files['image'][1].close()
        
        response.raise_for_status()
        
        # Save the video
        output_path = "generated_video.mp4"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Video generated successfully: {output_path}")
        print(f"   File size: {len(response.content)} bytes")
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

def test_generate_json_endpoint():
    """Test the /generate-json endpoint that returns base64 JSON"""
    print("Testing /generate-json endpoint...")
    
    if not Path(TEST_AUDIO_PATH).exists() or not Path(TEST_IMAGE_PATH).exists():
        print(f"❌ Test files not found: {TEST_AUDIO_PATH}, {TEST_IMAGE_PATH}")
        return False
    
    try:
        # Prepare files
        files = {
            'audio': ('test_audio.wav', open(TEST_AUDIO_PATH, 'rb'), 'audio/wav'),
            'image': ('test_image.jpg', open(TEST_IMAGE_PATH, 'rb'), 'image/jpeg')
        }
        
        # Prepare form data
        data = {
            'prompt': 'A person speaking clearly and naturally',
            'sample_steps': 30,
            'use_teacache': True,
            'low_vram': False
        }
        
        print("Sending request...")
        response = requests.post(
            f"{API_BASE_URL}/generate-json",
            files=files,
            data=data,
            timeout=1800  # 30 minutes
        )
        
        # Close files
        files['audio'][1].close()
        files['image'][1].close()
        
        response.raise_for_status()
        result = response.json()
        
        if result.get("success"):
            # Decode and save video
            video_data = base64.b64decode(result["video_data"])
            output_path = f"generated_video_{result.get('filename', 'output.mp4')}"
            
            with open(output_path, 'wb') as f:
                f.write(video_data)
            
            print(f"✅ Video generated successfully: {output_path}")
            print(f"   File size: {len(video_data)} bytes")
            print(f"   Prompt used: {result.get('prompt')}")
            print(f"   Sample steps: {result.get('sample_steps')}")
            return True
        else:
            print(f"❌ Generation failed: {result.get('error')}")
            if result.get('stdout'):
                print(f"   Output: {result['stdout']}")
            return False
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

def create_test_files():
    """Create minimal test files for testing"""
    print("Creating test files...")
    
    # Create a simple test audio file (1 second of silence)
    try:
        import numpy as np
        import soundfile as sf
        
        # Generate 1 second of silence at 16kHz
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio_data = np.zeros(samples, dtype=np.float32)
        
        sf.write(TEST_AUDIO_PATH, audio_data, sample_rate)
        print(f"✅ Created test audio: {TEST_AUDIO_PATH}")
        
    except ImportError:
        print("❌ soundfile not available, please provide your own test audio file")
    
    # Create a simple test image
    try:
        from PIL import Image
        import numpy as np
        
        # Create a 256x256 test image
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(TEST_IMAGE_PATH)
        print(f"✅ Created test image: {TEST_IMAGE_PATH}")
        
    except ImportError:
        print("❌ PIL not available, please provide your own test image file")

def main():
    """Run all tests"""
    print("MultiTalk Fixed API Test Suite")
    print("=" * 40)
    
    # Update API_BASE_URL before running tests
    if API_BASE_URL == "https://your-modal-app-url.modal.run":
        print("❌ Please update API_BASE_URL with your actual Modal app URL")
        return
    
    # Create test files if they don't exist
    if not Path(TEST_AUDIO_PATH).exists() or not Path(TEST_IMAGE_PATH).exists():
        create_test_files()
    
    # Run tests
    tests = [
        test_health_endpoint,
        test_generate_endpoint_binary,
        test_generate_json_endpoint
    ]
    
    results = []
    for test in tests:
        print("\n" + "-" * 40)
        result = test()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")

if __name__ == "__main__":
    main()