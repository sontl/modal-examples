#!/usr/bin/env python3
"""
Test script for MultiTalk Native implementation
"""

import base64
import json
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

def test_single_person_generation():
    """Test single person video generation"""
    print("🧪 Testing single person generation...")
    
    try:
        # Get the deployed class
        MultiTalkNative = modal.Cls.lookup("multitalk-native", "MultiTalkNative")
        
        # Create test files
        audio_path = create_test_audio()
        image_path = create_test_image()
        
        print(f"Created test audio: {audio_path}")
        print(f"Created test image: {image_path}")
        
        # Test generation
        result = MultiTalkNative().generate_single_person.remote(
            audio_path=audio_path,
            reference_image_path=image_path,
            prompt="A person talking naturally",
            sample_steps=20,  # Reduced for faster testing
            use_teacache=True,
            low_vram=True
        )
        
        if result.get("success"):
            print("✅ Single person generation successful!")
            
            # Save output video
            video_data = base64.b64decode(result["video_data"])
            output_path = "test_single_output.mp4"
            with open(output_path, "wb") as f:
                f.write(video_data)
            print(f"📹 Video saved to {output_path}")
            
        else:
            print(f"❌ Generation failed: {result.get('error')}")
            return False
            
        # Cleanup
        Path(audio_path).unlink()
        Path(image_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def test_fusionx_acceleration():
    """Test FusionX LoRA acceleration"""
    print("🧪 Testing FusionX LoRA acceleration...")
    
    try:
        MultiTalkNative = modal.Cls.lookup("multitalk-native", "MultiTalkNative")
        
        audio_path = create_test_audio()
        image_path = create_test_image()
        
        # Test with FusionX LoRA
        result = MultiTalkNative().generate_single_person.remote(
            audio_path=audio_path,
            reference_image_path=image_path,
            prompt="A person talking with FusionX acceleration",
            use_lora="fusionx",
            sample_steps=8,  # FusionX optimal steps
            use_teacache=True
        )
        
        if result.get("success"):
            print("✅ FusionX acceleration test successful!")
            
            video_data = base64.b64decode(result["video_data"])
            output_path = "test_fusionx_output.mp4"
            with open(output_path, "wb") as f:
                f.write(video_data)
            print(f"📹 FusionX video saved to {output_path}")
            
        else:
            print(f"❌ FusionX test failed: {result.get('error')}")
            return False
            
        Path(audio_path).unlink()
        Path(image_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ FusionX test failed with exception: {e}")
        return False

def test_quantization():
    """Test INT8 quantization"""
    print("🧪 Testing INT8 quantization...")
    
    try:
        MultiTalkNative = modal.Cls.lookup("multitalk-native", "MultiTalkNative")
        
        audio_path = create_test_audio()
        image_path = create_test_image()
        
        # Test with quantization
        result = MultiTalkNative().generate_single_person.remote(
            audio_path=audio_path,
            reference_image_path=image_path,
            prompt="A person talking with quantization",
            use_quantization=True,
            low_vram=True,
            sample_steps=20
        )
        
        if result.get("success"):
            print("✅ Quantization test successful!")
            
            video_data = base64.b64decode(result["video_data"])
            output_path = "test_quantized_output.mp4"
            with open(output_path, "wb") as f:
                f.write(video_data)
            print(f"📹 Quantized video saved to {output_path}")
            
        else:
            print(f"❌ Quantization test failed: {result.get('error')}")
            return False
            
        Path(audio_path).unlink()
        Path(image_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Quantization test failed with exception: {e}")
        return False

def test_multi_person_generation():
    """Test multi-person video generation"""
    print("🧪 Testing multi-person generation...")
    
    try:
        MultiTalkNative = modal.Cls.lookup("multitalk-native", "MultiTalkNative")
        
        # Create test files for two people
        audio_path1 = create_test_audio()
        audio_path2 = create_test_audio()
        image_path1 = create_test_image()
        image_path2 = create_test_image()
        
        # Test multi-person generation
        result = MultiTalkNative().generate_multi_person.remote(
            audio_paths=[audio_path1, audio_path2],
            reference_images=[image_path1, image_path2],
            prompts=["Person 1 speaking", "Person 2 responding"],
            sample_steps=20,
            use_teacache=True
        )
        
        if result.get("success"):
            print("✅ Multi-person generation successful!")
            
            video_data = base64.b64decode(result["video_data"])
            output_path = "test_multi_output.mp4"
            with open(output_path, "wb") as f:
                f.write(video_data)
            print(f"📹 Multi-person video saved to {output_path}")
            
        else:
            print(f"❌ Multi-person generation failed: {result.get('error')}")
            return False
            
        # Cleanup
        for path in [audio_path1, audio_path2, image_path1, image_path2]:
            Path(path).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-person test failed with exception: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Running MultiTalk Native tests...\n")
    
    tests = [
        ("Single Person Generation", test_single_person_generation),
        ("FusionX Acceleration", test_fusionx_acceleration),
        ("INT8 Quantization", test_quantization),
        ("Multi-Person Generation", test_multi_person_generation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MultiTalk Native")
    parser.add_argument("--test", choices=[
        "single", "fusionx", "quantization", "multi", "all"
    ], default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test == "single":
        test_single_person_generation()
    elif args.test == "fusionx":
        test_fusionx_acceleration()
    elif args.test == "quantization":
        test_quantization()
    elif args.test == "multi":
        test_multi_person_generation()
    else:
        run_all_tests()