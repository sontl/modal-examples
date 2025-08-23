#!/usr/bin/env python3
"""
Usage examples for the fixed MultiTalk implementation
"""

import requests
import json
import base64

def test_single_person_api():
    """Test single person video generation via API"""
    
    # Example API call for single person
    url = "https://your-modal-app-url.modal.run/generate-json"
    
    # You would replace these with actual file paths
    audio_file_path = "path/to/your/audio.wav"
    image_file_path = "path/to/your/image.jpg"
    
    files = {
        'audio': open(audio_file_path, 'rb'),
        'image': open(image_file_path, 'rb')
    }
    
    data = {
        'prompt': 'A person speaking naturally in a professional setting',
        'sample_steps': 40,
        'audio_type': 'single'
    }
    
    response = requests.post(url, files=files, data=data)
    result = response.json()
    
    if result.get('success'):
        # Save the generated video
        video_data = base64.b64decode(result['video_data'])
        with open('output_single.mp4', 'wb') as f:
            f.write(video_data)
        print("✅ Single person video generated successfully!")
    else:
        print(f"❌ Error: {result.get('error')}")

def test_multi_person_api():
    """Test multi-person video generation via API"""
    
    url = "https://your-modal-app-url.modal.run/generate-json"
    
    # You would replace these with actual file paths
    audio_file1_path = "path/to/person1_audio.wav"
    audio_file2_path = "path/to/person2_audio.wav"
    image_file_path = "path/to/your/image.jpg"
    
    files = {
        'audio': open(audio_file1_path, 'rb'),
        'audio_person2': open(audio_file2_path, 'rb'),
        'image': open(image_file_path, 'rb')
    }
    
    # Bounding boxes for two people (optional)
    bbox_person1 = [160, 120, 1280, 1080]  # [x, y, width, height]
    bbox_person2 = [160, 1320, 1280, 2280]
    
    data = {
        'prompt': 'Two people having a conversation in a casual setting',
        'sample_steps': 40,
        'audio_type': 'add',  # or 'para' for parallel speech
        'bbox_person1': json.dumps(bbox_person1),
        'bbox_person2': json.dumps(bbox_person2)
    }
    
    response = requests.post(url, files=files, data=data)
    result = response.json()
    
    if result.get('success'):
        # Save the generated video
        video_data = base64.b64decode(result['video_data'])
        with open('output_multi.mp4', 'wb') as f:
            f.write(video_data)
        print("✅ Multi-person video generated successfully!")
    else:
        print(f"❌ Error: {result.get('error')}")

def test_cli_usage():
    """Show CLI usage examples"""
    
    print("CLI Usage Examples:")
    print("==================")
    
    print("\n1. Single person video generation:")
    print("modal run multitalk_fixed.py --audio-path audio.wav --image-path image.jpg")
    
    print("\n2. Multi-person video generation:")
    print("modal run multitalk_fixed.py --audio-path person1.wav --image-path image.jpg --audio-path-person2 person2.wav --audio-type add")
    
    print("\n3. With custom prompt and settings:")
    print("modal run multitalk_fixed.py --audio-path audio.wav --image-path image.jpg --prompt 'A professional speaker giving a presentation' --sample-steps 30")

if __name__ == "__main__":
    print("MultiTalk Fixed - Usage Examples")
    print("================================")
    
    print("\nThis script shows how to use the fixed MultiTalk implementation.")
    print("The key fixes include:")
    print("- Proper JSON structure matching MultiTalk expectations")
    print("- Support for both single and multi-person scenarios")
    print("- Correct audio_type handling ('single', 'add', 'para')")
    print("- Optional bounding box support for multi-person videos")
    print("- Better error handling and validation")
    
    test_cli_usage()
    
    print("\nFor API usage, uncomment and modify the test functions above.")
    print("Make sure to replace file paths with your actual audio/image files.")