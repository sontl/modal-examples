#!/usr/bin/env python3
"""
Example usage of MultiTalk Fixed FastAPI
"""

import requests
import base64
from pathlib import Path

def generate_video_from_files(
    api_url: str,
    audio_path: str,
    image_path: str,
    prompt: str = "A person talking naturally",
    sample_steps: int = 40,
    output_path: str = "generated_video.mp4"
):
    """
    Generate a talking video from audio and image files
    
    Args:
        api_url: Base URL of the MultiTalk API
        audio_path: Path to audio file
        image_path: Path to reference image
        prompt: Generation prompt
        sample_steps: Number of sampling steps (10-50)
        output_path: Where to save the generated video
    """
    
    # Check if files exist
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print(f"Generating video from:")
    print(f"  Audio: {audio_path}")
    print(f"  Image: {image_path}")
    print(f"  Prompt: {prompt}")
    print(f"  Steps: {sample_steps}")
    
    try:
        # Prepare files for upload
        files = {
            'audio': ('audio.wav', open(audio_path, 'rb'), 'audio/wav'),
            'image': ('image.jpg', open(image_path, 'rb'), 'image/jpeg')
        }
        
        # Prepare form data
        data = {
            'prompt': prompt,
            'sample_steps': sample_steps,
            'use_teacache': True,
            'low_vram': False
        }
        
        print("Sending request to API...")
        
        # Make request to binary endpoint
        response = requests.post(
            f"{api_url}/generate",
            files=files,
            data=data,
            timeout=1800  # 30 minutes timeout
        )
        
        # Close files
        files['audio'][1].close()
        files['image'][1].close()
        
        # Check response
        response.raise_for_status()
        
        # Save video
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Video saved to: {output_path}")
        print(f"   File size: {len(response.content):,} bytes")
        
        return output_path
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out (generation took too long)")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error: {e}")
        print(f"   Response: {response.text}")
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

def generate_video_json_response(
    api_url: str,
    audio_path: str,
    image_path: str,
    prompt: str = "A person talking naturally",
    sample_steps: int = 40
):
    """
    Generate video and get JSON response with base64 data
    
    Returns:
        dict: Response containing success status and video data
    """
    
    try:
        # Prepare files
        files = {
            'audio': ('audio.wav', open(audio_path, 'rb'), 'audio/wav'),
            'image': ('image.jpg', open(image_path, 'rb'), 'image/jpeg')
        }
        
        data = {
            'prompt': prompt,
            'sample_steps': sample_steps,
            'use_teacache': True,
            'low_vram': False
        }
        
        print("Sending request to JSON endpoint...")
        
        response = requests.post(
            f"{api_url}/generate-json",
            files=files,
            data=data,
            timeout=1800
        )
        
        # Close files
        files['audio'][1].close()
        files['image'][1].close()
        
        response.raise_for_status()
        result = response.json()
        
        if result.get("success"):
            print("✅ Generation successful")
            
            # Optionally save the video
            video_data = base64.b64decode(result["video_data"])
            output_path = f"generated_{result.get('filename', 'video.mp4')}"
            
            with open(output_path, 'wb') as f:
                f.write(video_data)
            
            print(f"   Video saved to: {output_path}")
            print(f"   File size: {len(video_data):,} bytes")
            
        else:
            print(f"❌ Generation failed: {result.get('error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

def check_api_health(api_url: str):
    """Check if the API is healthy"""
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ API is healthy: {data}")
        return True
        
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Configuration
    API_URL = "https://your-modal-app-url.modal.run"  # Replace with your actual URL
    AUDIO_FILE = "sample_audio.wav"  # Your audio file
    IMAGE_FILE = "reference_image.jpg"  # Your reference image
    
    # Update the API URL
    if API_URL == "https://your-modal-app-url.modal.run":
        print("❌ Please update API_URL with your actual Modal app URL")
        print("   You can find this URL after deploying with: modal deploy multitalk_fixed.py")
        exit(1)
    
    # Check API health
    print("Checking API health...")
    if not check_api_health(API_URL):
        print("API is not available. Please check your deployment.")
        exit(1)
    
    # Example 1: Generate video with binary response
    print("\n" + "="*50)
    print("Example 1: Binary response")
    try:
        output_file = generate_video_from_files(
            api_url=API_URL,
            audio_path=AUDIO_FILE,
            image_path=IMAGE_FILE,
            prompt="A person speaking clearly and expressively",
            sample_steps=35,
            output_path="output_binary.mp4"
        )
        print(f"Success! Video saved to: {output_file}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Example 2: Generate video with JSON response
    print("\n" + "="*50)
    print("Example 2: JSON response")
    try:
        result = generate_video_json_response(
            api_url=API_URL,
            audio_path=AUDIO_FILE,
            image_path=IMAGE_FILE,
            prompt="A person talking with natural expressions",
            sample_steps=40
        )
        
        if result.get("success"):
            print("JSON generation completed successfully!")
        else:
            print("JSON generation failed")
            
    except Exception as e:
        print(f"Failed: {e}")