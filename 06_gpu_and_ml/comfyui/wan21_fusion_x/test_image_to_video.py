#!/usr/bin/env python3

import requests
from pathlib import Path

def test_image_to_video_endpoint():
    """Test the form-data image-to-video endpoint"""
    
    # Replace with your actual Modal endpoint URL
    endpoint_url = "https://your-modal-app-url/image-to-video"
    
    # Path to your test image
    image_path = "test_image.jpg"  # Replace with actual image path
    
    if not Path(image_path).exists():
        print(f"Test image not found at {image_path}")
        print("Please provide a test image file")
        return
    
    # Read the image file
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # Optional: Custom prompt
    prompt = "A beautiful sunset with moving clouds and gentle waves"
    
    # Prepare the request
    files = {
        'image': ('test_image.jpg', image_data, 'image/jpeg')
    }
    
    data = {
        'prompt': prompt
    }
    
    print("Sending request to form-data image-to-video endpoint...")
    
    try:
        # Send POST request
        response = requests.post(endpoint_url, files=files, data=data, timeout=300)
        
        if response.status_code == 200:
            # Save the generated video
            output_path = "generated_video_form.mp4"
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Video generated successfully! Saved to {output_path}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    print("Testing image-to-video endpoint...")
    test_image_to_video_endpoint()