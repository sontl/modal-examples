#!/usr/bin/env python3
"""
Test script for the video upscaler API
"""

import requests
from pathlib import Path

def test_upscaler_api(video_path: str, api_url: str):
    """
    Test the video upscaler API by uploading a video file
    
    Args:
        video_path: Path to the input video file
        api_url: URL of the deployed Modal API endpoint
    """
    
    # Check if video file exists
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"Error: Video file {video_path} not found")
        return
    
    print(f"Uploading video: {video_file.name}")
    print(f"API endpoint: {api_url}")
    
    # Prepare the file for upload
    with open(video_file, 'rb') as f:
        files = {'video': (video_file.name, f, 'video/mp4')}
        
        try:
            # Send POST request to the API
            response = requests.post(f"{api_url}/upscale", files=files, timeout=300)
            
            if response.status_code == 200:
                # Save the upscaled video
                output_path = video_file.parent / f"upscaled_{video_file.name}"
                with open(output_path, 'wb') as output_file:
                    output_file.write(response.content)
                
                print(f"✅ Success! Upscaled video saved to: {output_path}")
                print(f"Original size: {video_file.stat().st_size / (1024*1024):.2f} MB")
                print(f"Upscaled size: {output_path.stat().st_size / (1024*1024):.2f} MB")
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out. Video processing may take longer than expected.")
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python test_upscaler.py <video_path> <api_url>")
        print("Example: python test_upscaler.py input.mp4 https://your-workspace--upscaler-wan-fastapi-app.modal.run")
        sys.exit(1)
    
    video_path = sys.argv[1]
    api_url = sys.argv[2]
    
    test_upscaler_api(video_path, api_url)