#!/usr/bin/env python3
"""
Test script to verify both endpoints work correctly
"""

import requests
from pathlib import Path
import sys

def test_fastapi_endpoint(video_path: str, api_url: str):
    """Test the FastAPI endpoint (/upscale)"""
    print(f"🧪 Testing FastAPI endpoint: {api_url}/upscale")
    
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"❌ Video file {video_path} not found")
        return False
    
    try:
        with open(video_file, 'rb') as f:
            files = {'video': (video_file.name, f, 'video/mp4')}
            response = requests.post(f"{api_url}/upscale", files=files, timeout=300)
        
        if response.status_code == 200:
            output_path = video_file.parent / f"fastapi_upscaled_{video_file.name}"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ FastAPI endpoint test successful! Output: {output_path}")
            return True
        else:
            print(f"❌ FastAPI endpoint failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ FastAPI endpoint error: {e}")
        return False

def test_webhook_endpoint(video_path: str, webhook_url: str):
    """Test the webhook endpoint (upload-and-upscale)"""
    print(f"🧪 Testing webhook endpoint: {webhook_url}")
    
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"❌ Video file {video_path} not found")
        return False
    
    try:
        with open(video_file, 'rb') as f:
            video_bytes = f.read()
        
        response = requests.post(webhook_url, data=video_bytes, 
                               headers={'Content-Type': 'application/octet-stream'}, 
                               timeout=300)
        
        if response.status_code == 200:
            output_path = video_file.parent / f"webhook_upscaled_{video_file.name}"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Webhook endpoint test successful! Output: {output_path}")
            return True
        else:
            print(f"❌ Webhook endpoint failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Webhook endpoint error: {e}")
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: python test_endpoints.py <video_path> <fastapi_url> <webhook_url>")
        print("Example:")
        print("  python test_endpoints.py input.mp4 \\")
        print("    https://workspace--upscaler-wan-fastapi-app.modal.run \\")
        print("    https://workspace--upload-and-upscale.modal.run")
        sys.exit(1)
    
    video_path = sys.argv[1]
    fastapi_url = sys.argv[2]
    webhook_url = sys.argv[3]
    
    print("🚀 Testing Modal Video Upscaler Endpoints")
    print("=" * 50)
    
    # Test both endpoints
    fastapi_success = test_fastapi_endpoint(video_path, fastapi_url)
    print()
    webhook_success = test_webhook_endpoint(video_path, webhook_url)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"FastAPI endpoint: {'✅ PASS' if fastapi_success else '❌ FAIL'}")
    print(f"Webhook endpoint: {'✅ PASS' if webhook_success else '❌ FAIL'}")
    
    if fastapi_success and webhook_success:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())