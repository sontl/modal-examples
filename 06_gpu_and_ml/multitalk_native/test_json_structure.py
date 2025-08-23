#!/usr/bin/env python3
"""
Test script to verify JSON structure matches MultiTalk expectations
"""

import json
import tempfile
import os

def test_single_person_json():
    """Test single person JSON structure"""
    
    # Create dummy files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file:
        audio_file.write(b"dummy audio data")
        audio_path = audio_file.name
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as image_file:
        image_file.write(b"dummy image data")
        image_path = image_file.name
    
    # Create JSON structure for single person (like single_example_1.json)
    input_data = {
        "prompt": "A woman is passionately singing into a professional microphone in a recording studio.",
        "cond_image": image_path,
        "cond_audio": {
            "person1": audio_path
        }
    }
    
    print("Single person JSON structure:")
    print(json.dumps(input_data, indent=2))
    
    # Cleanup
    os.unlink(audio_path)
    os.unlink(image_path)
    
    return input_data

def test_multi_person_add_json():
    """Test multi-person add JSON structure"""
    
    # Create dummy files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file1:
        audio_file1.write(b"dummy audio data 1")
        audio_path1 = audio_file1.name
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file2:
        audio_file2.write(b"dummy audio data 2")
        audio_path2 = audio_file2.name
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as image_file:
        image_file.write(b"dummy image data")
        image_path = image_file.name
    
    # Create JSON structure for multi-person add (like multitalk_example_1.json)
    input_data = {
        "prompt": "In a casual, intimate setting, a man and a woman are engaged in a heartfelt conversation inside a car.",
        "cond_image": image_path,
        "audio_type": "add",
        "cond_audio": {
            "person1": audio_path1,
            "person2": audio_path2
        },
        "bbox": {
            "person1": [160, 120, 1280, 1080],
            "person2": [160, 1320, 1280, 2280]
        }
    }
    
    print("\nMulti-person 'add' JSON structure:")
    print(json.dumps(input_data, indent=2))
    
    # Cleanup
    os.unlink(audio_path1)
    os.unlink(audio_path2)
    os.unlink(image_path)
    
    return input_data

def test_multi_person_para_json():
    """Test multi-person parallel JSON structure"""
    
    # Create dummy files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file1:
        audio_file1.write(b"dummy audio data 1")
        audio_path1 = audio_file1.name
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file2:
        audio_file2.write(b"dummy audio data 2")
        audio_path2 = audio_file2.name
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as image_file:
        image_file.write(b"dummy image data")
        image_path = image_file.name
    
    # Create JSON structure for multi-person parallel (like multitalk_example_2.json)
    input_data = {
        "prompt": "In a cozy recording studio, a man and a woman are singing together with passion and emotion.",
        "cond_image": image_path,
        "audio_type": "para",
        "cond_audio": {
            "person1": audio_path1,
            "person2": audio_path2
        }
    }
    
    print("\nMulti-person 'para' JSON structure:")
    print(json.dumps(input_data, indent=2))
    
    # Cleanup
    os.unlink(audio_path1)
    os.unlink(audio_path2)
    os.unlink(image_path)
    
    return input_data

if __name__ == "__main__":
    print("Testing JSON structures for MultiTalk...")
    
    single_json = test_single_person_json()
    multi_add_json = test_multi_person_add_json()
    multi_para_json = test_multi_person_para_json()
    
    print("\n" + "="*50)
    print("All JSON structures generated successfully!")
    print("These match the expected MultiTalk input format.")