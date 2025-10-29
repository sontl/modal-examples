#!/usr/bin/env python3
"""
Simple test for FlashVSR endpoint
"""

import requests
import json

# Update with your deployed endpoint URL
ENDPOINT_URL = "https://son-tranlam-1986--flashvsr-endpoint-flashvsrservice-inference.modal.run"

def test_flashvsr():
    """Test FlashVSR endpoint"""
    
    if ENDPOINT_URL == "https://your-flashvsr-endpoint.modal.run":
        print("❌ Please update ENDPOINT_URL with your actual endpoint")
        return
    
    # Test with full model
    test_data = {
        "video_url": "https://d2p7pge43lyniu.cloudfront.net/output/a01411dd-197b-4798-a3e4-44f71a9e624b-u1_962b879b-25f3-4804-b164-42753f5667e3.mp4",
        "model_type": "full",  # or "tiny"
        "output_format": "mp4",
        "scale": 2,
        "seed": 0,
        "sparse_ratio": 2.0,
        "local_range": 11,
        "max_frames": 10
    }
    
    print("🚀 Testing FlashVSR endpoint...")
    print(f"📡 URL: {ENDPOINT_URL}")
    print(f"📊 Request: {json.dumps(test_data, indent=2)}")
    
    try:
        print("\n⏳ Sending request (may take 2-5 minutes for first request)...")
        
        response = requests.post(ENDPOINT_URL, json=test_data, timeout=600)
        
        if response.status_code == 200:
            print("✅ Success!")
            print(f"📁 Response size: {len(response.content):,} bytes")
            
            # Check headers
            processing_time = response.headers.get("X-Processing-Time", "Unknown")
            model_type = response.headers.get("X-Model-Type", "Unknown")
            scale_factor = response.headers.get("X-Scale-Factor", "Unknown")
            frames_processed = response.headers.get("X-Frames-Processed", "Unknown")
            
            print(f"⏱️  Processing time: {processing_time}s")
            print(f"🤖 Model used: FlashVSR {model_type}")
            print(f"📏 Scale factor: {scale_factor}x")
            print(f"🖼️  Frames processed: {frames_processed}")
            
            # Save output
            output_file = f"flashvsr_{model_type}_output.mp4"
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"💾 Saved to: {output_file}")
            
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_flashvsr()