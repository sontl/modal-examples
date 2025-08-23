#!/usr/bin/env python3
"""
Deploy and test the fixed MultiTalk implementation
"""

import modal
import base64
import tempfile
import os

def create_test_files():
    """Create simple test files for testing"""
    # Create a simple test audio (sine wave)
    import numpy as np
    import soundfile as sf
    
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sample_rate)
        audio_path = f.name
    
    # Create a simple test image
    from PIL import Image
    import numpy as np
    
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    # Add some structure to make it more face-like
    img_array[200:300, 200:300] = [255, 200, 180]  # Face area
    img_array[220:240, 220:240] = [0, 0, 0]  # Left eye
    img_array[220:240, 280:300] = [0, 0, 0]  # Right eye
    img_array[260:280, 240:280] = [200, 150, 150]  # Mouth area
    
    img = Image.fromarray(img_array)
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img.save(f.name, 'JPEG')
        image_path = f.name
    
    return audio_path, image_path

def test_deployment():
    """Test the MultiTalk deployment"""
    print("🚀 Testing MultiTalk Fixed Deployment")
    print("=" * 50)
    
    try:
        # Create test files
        print("Creating test files...")
        audio_path, image_path = create_test_files()
        
        # Read files as bytes
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        print(f"Audio size: {len(audio_data)} bytes")
        print(f"Image size: {len(image_data)} bytes")
        
        # Get the deployed class
        print("Connecting to deployed MultiTalk...")
        MultiTalkFixed = modal.Cls.lookup("multitalk-fixed", "MultiTalkFixed")
        
        # Test generation
        print("Starting video generation...")
        result = MultiTalkFixed().generate_video.remote(
            audio_data=audio_data,
            image_data=image_data,
            prompt="A person speaking naturally",
            sample_steps=20,  # Reduced for faster testing
            use_teacache=True,
            low_vram=False
        )
        
        print("Generation completed!")
        print(f"Result: {result}")
        
        if result.get("success"):
            # Save the generated video
            video_data = base64.b64decode(result["video_data"])
            output_path = "test_output.mp4"
            
            with open(output_path, "wb") as f:
                f.write(video_data)
            
            print(f"✅ Video saved to {output_path}")
            print(f"Video size: {len(video_data)} bytes")
            
            # Verify the video file
            if os.path.getsize(output_path) > 0:
                print("✅ Video file has content!")
            else:
                print("❌ Video file is empty!")
                
        else:
            print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
            if result.get('stdout'):
                print(f"Stdout: {result['stdout']}")
            if result.get('traceback'):
                print(f"Traceback: {result['traceback']}")
        
        # Cleanup
        os.unlink(audio_path)
        os.unlink(image_path)
        
        return result.get("success", False)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def deploy_app():
    """Deploy the MultiTalk app"""
    print("📦 Deploying MultiTalk Fixed...")
    print("=" * 50)
    
    try:
        # This would deploy the app
        print("Run the following command to deploy:")
        print("modal deploy 06_gpu_and_ml/multitalk_native/multitalk_fixed.py")
        print("\nThen test with:")
        print("python 06_gpu_and_ml/multitalk_native/deploy_fixed.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return False

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test existing deployment
        success = test_deployment()
    else:
        # Show deployment instructions
        success = deploy_app()
    
    if success:
        print("\n✅ Success!")
    else:
        print("\n❌ Failed!")
    
    return success

if __name__ == "__main__":
    main()