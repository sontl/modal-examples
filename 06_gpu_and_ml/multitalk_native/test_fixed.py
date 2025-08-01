#!/usr/bin/env python3
"""
Test script for MultiTalk Fixed implementation
"""

import base64
import tempfile
from pathlib import Path

import modal

def create_test_audio():
    """Create a simple test audio file"""
    import numpy as np
    import soundfile as sf
    
    # Generate a simple sine wave (1 second at 22050 Hz)
    sample_rate = 22050
    duration = 1.0
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sample_rate)
        return f.name

def create_test_image():
    """Create a simple test image"""
    from PIL import Image
    import numpy as np
    
    # Create a simple 512x512 test image
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img.save(f.name, 'JPEG')
        return f.name

def test_multitalk_fixed():
    """Test MultiTalk Fixed deployment"""
    print("🧪 Testing MultiTalk Fixed...")
    
    try:
        # Get the deployed class
        MultiTalkFixed = modal.Cls.lookup("multitalk-fixed", "MultiTalkFixed")
        
        # Create test files
        print("Creating test files...")
        audio_path = create_test_audio()
        image_path = create_test_image()
        
        print(f"Created test audio: {audio_path}")
        print(f"Created test image: {image_path}")
        
        # Read file data
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        print("Calling MultiTalk generation...")
        
        # Test generation
        result = MultiTalkFixed().generate_video.remote(
            audio_data=audio_data,
            image_data=image_data,
            prompt="A person talking naturally",
            sample_steps=20,  # Reduced for faster testing
            use_teacache=True,
            low_vram=True
        )
        
        print("Generation completed!")
        
        if result.get("success"):
            print("✅ MultiTalk generation successful!")
            
            # Save output video
            video_data = base64.b64decode(result["video_data"])
            output_path = "test_fixed_output.mp4"
            with open(output_path, "wb") as f:
                f.write(video_data)
            print(f"📹 Video saved to {output_path}")
            
        else:
            print(f"❌ Generation failed: {result.get('error')}")
            if result.get('stdout'):
                print(f"Output: {result['stdout']}")
            return False
            
        # Cleanup
        Path(audio_path).unlink()
        Path(image_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_multitalk_fixed()
    if success:
        print("🎉 Test passed!")
    else:
        print("⚠️  Test failed")