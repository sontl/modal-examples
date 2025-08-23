#!/usr/bin/env python3
"""
Test script for RunPod ComfyUI Video Upscaler endpoint
"""

import requests
import base64
import json
import time
import os
from pathlib import Path


def test_with_video_url():
    """Test the endpoint with a video URL"""
    
    # Replace with your RunPod endpoint URL and API key
    ENDPOINT_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
    API_KEY = "YOUR_API_KEY"
    
    # Test video URL (replace with actual video URL)
    VIDEO_URL = "https://example.com/test_video.mp4"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "video_url": VIDEO_URL
        }
    }
    
    print("🚀 Testing video upscaler with URL...")
    print(f"Video URL: {VIDEO_URL}")
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        
        result = response.json()
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        output = result.get("output", {})
        
        print("✅ Success!")
        print(f"Original size: {output.get('original_size_bytes', 0)} bytes")
        print(f"Upscaled size: {output.get('upscaled_size_bytes', 0)} bytes")
        print(f"Job ID: {output.get('job_id', 'N/A')}")
        
        # Save upscaled video
        if "upscaled_video_base64" in output:
            video_data = base64.b64decode(output["upscaled_video_base64"])
            output_path = f"upscaled_output_{output.get('job_id', 'test')}.mp4"
            
            with open(output_path, "wb") as f:
                f.write(video_data)
            
            print(f"💾 Saved upscaled video: {output_path}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_with_local_video(video_path: str):
    """Test the endpoint with a local video file"""
    
    # Replace with your RunPod endpoint URL and API key
    ENDPOINT_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
    API_KEY = "YOUR_API_KEY"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Read and encode video
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    
    video_base64 = base64.b64encode(video_bytes).decode('utf-8')
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "video_base64": video_base64
        }
    }
    
    print("🚀 Testing video upscaler with local file...")
    print(f"Video file: {video_path}")
    print(f"File size: {len(video_bytes)} bytes")
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        
        result = response.json()
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        output = result.get("output", {})
        
        print("✅ Success!")
        print(f"Original size: {output.get('original_size_bytes', 0)} bytes")
        print(f"Upscaled size: {output.get('upscaled_size_bytes', 0)} bytes")
        print(f"Job ID: {output.get('job_id', 'N/A')}")
        
        # Save upscaled video
        if "upscaled_video_base64" in output:
            upscaled_video_data = base64.b64decode(output["upscaled_video_base64"])
            
            # Create output filename
            input_path = Path(video_path)
            output_path = f"upscaled_{input_path.stem}_{output.get('job_id', 'test')}.mp4"
            
            with open(output_path, "wb") as f:
                f.write(upscaled_video_data)
            
            print(f"💾 Saved upscaled video: {output_path}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_async_endpoint():
    """Test the endpoint asynchronously"""
    
    # Replace with your RunPod endpoint URL and API key
    ENDPOINT_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run"
    STATUS_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status"
    API_KEY = "YOUR_API_KEY"
    
    # Test video URL
    VIDEO_URL = "https://example.com/test_video.mp4"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "video_url": VIDEO_URL
        }
    }
    
    print("🚀 Testing async video upscaler...")
    
    try:
        # Submit job
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        job_data = response.json()
        job_id = job_data.get("id")
        
        if not job_id:
            print(f"❌ Failed to get job ID: {job_data}")
            return
        
        print(f"📋 Job submitted: {job_id}")
        print("⏳ Waiting for completion...")
        
        # Poll for completion
        while True:
            status_response = requests.get(f"{STATUS_URL}/{job_id}", headers=headers)
            status_response.raise_for_status()
            
            status_data = status_response.json()
            status = status_data.get("status")
            
            print(f"Status: {status}")
            
            if status == "COMPLETED":
                output = status_data.get("output", {})
                
                print("✅ Job completed!")
                print(f"Original size: {output.get('original_size_bytes', 0)} bytes")
                print(f"Upscaled size: {output.get('upscaled_size_bytes', 0)} bytes")
                
                # Save result
                if "upscaled_video_base64" in output:
                    video_data = base64.b64decode(output["upscaled_video_base64"])
                    output_path = f"async_upscaled_{job_id}.mp4"
                    
                    with open(output_path, "wb") as f:
                        f.write(video_data)
                    
                    print(f"💾 Saved upscaled video: {output_path}")
                
                break
                
            elif status == "FAILED":
                print(f"❌ Job failed: {status_data.get('error', 'Unknown error')}")
                break
                
            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                time.sleep(10)  # Wait 10 seconds before checking again
                
            else:
                print(f"❓ Unknown status: {status}")
                break
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🎬 RunPod ComfyUI Video Upscaler Test")
    print("=" * 50)
    
    # Update these with your actual values
    print("⚠️  Please update the following in this script:")
    print("   - ENDPOINT_URL (your RunPod endpoint)")
    print("   - API_KEY (your RunPod API key)")
    print("   - VIDEO_URL or local video path")
    print()
    
    # Uncomment the test you want to run:
    
    # Test with URL
    # test_with_video_url()
    
    # Test with local file
    # test_with_local_video("path/to/your/video.mp4")
    
    # Test async
    # test_async_endpoint()
    
    print("Please uncomment one of the test functions above to run a test.")