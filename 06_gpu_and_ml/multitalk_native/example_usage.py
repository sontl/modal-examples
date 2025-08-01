#!/usr/bin/env python3
"""
Example usage of MultiTalk Native on Modal.com
"""

import base64
import requests
from pathlib import Path

def example_single_person_cli():
    """Example: Single person generation via CLI"""
    print("🎬 Example: Single Person Generation (CLI)")
    print("="*50)
    
    # This would be run from command line:
    cli_command = """
modal run 06_gpu_and_ml/multitalk_native/multitalk_native.py \\
  --audio-path "examples/audio/person1.wav" \\
  --image-path "examples/images/person1.jpg" \\
  --prompt "A person speaking naturally with expressive movements" \\
  --use-lora "fusionx" \\
  --output-path "output_single.mp4"
    """
    
    print("Command to run:")
    print(cli_command)

def example_single_person_python():
    """Example: Single person generation via Python API"""
    print("\n🎬 Example: Single Person Generation (Python API)")
    print("="*50)
    
    code_example = '''
import modal

# Get the deployed MultiTalk class
MultiTalkNative = modal.Cls.lookup("multitalk-native", "MultiTalkNative")

# Generate single person video
result = MultiTalkNative().generate_single_person.remote(
    audio_path="examples/audio/person1.wav",
    reference_image_path="examples/images/person1.jpg",
    prompt="A person speaking naturally with expressive movements",
    use_lora="fusionx",  # Use FusionX for 2x speedup
    use_teacache=True,   # Enable TeaCache for additional speedup
    sample_steps=8       # Optimal for FusionX
)

if result["success"]:
    # Decode and save the video
    import base64
    video_data = base64.b64decode(result["video_data"])
    with open("output_single.mp4", "wb") as f:
        f.write(video_data)
    print(f"Video saved: {result['filename']}")
else:
    print(f"Error: {result['error']}")
    '''
    
    print("Python code:")
    print(code_example)

def example_multi_person_python():
    """Example: Multi-person generation via Python API"""
    print("\n🎬 Example: Multi-Person Generation (Python API)")
    print("="*50)
    
    code_example = '''
import modal

MultiTalkNative = modal.Cls.lookup("multitalk-native", "MultiTalkNative")

# Generate multi-person conversation video
result = MultiTalkNative().generate_multi_person.remote(
    audio_paths=[
        "examples/audio/person1.wav",
        "examples/audio/person2.wav"
    ],
    reference_images=[
        "examples/images/person1.jpg", 
        "examples/images/person2.jpg"
    ],
    prompts=[
        "Person 1 speaking in a friendly conversation",
        "Person 2 responding with enthusiasm"
    ],
    use_lora="lightx2v",  # Use LightX2V for 4x speedup
    use_teacache=True,
    use_apg=True,         # Enable APG for better long video quality
    sample_steps=4        # Optimal for LightX2V
)

if result["success"]:
    video_data = base64.b64decode(result["video_data"])
    with open("output_multi.mp4", "wb") as f:
        f.write(video_data)
    print("Multi-person video generated successfully!")
    '''
    
    print("Python code:")
    print(code_example)

def example_low_vram_setup():
    """Example: Low VRAM configuration"""
    print("\n💾 Example: Low VRAM Configuration")
    print("="*50)
    
    code_example = '''
# For systems with limited VRAM (e.g., RTX 4090 24GB)
result = MultiTalkNative().generate_single_person.remote(
    audio_path="examples/audio/person1.wav",
    reference_image_path="examples/images/person1.jpg",
    prompt="A person talking",
    use_quantization=True,  # Enable INT8 quantization
    low_vram=True,          # Enable low VRAM mode
    sample_steps=20,        # Reduce steps for faster generation
    use_teacache=True       # Enable TeaCache acceleration
)
    '''
    
    print("Python code for low VRAM:")
    print(code_example)

def example_rest_api_usage():
    """Example: REST API usage"""
    print("\n🌐 Example: REST API Usage")
    print("="*50)
    
    # First, show how to start the API server
    print("1. Start the API server:")
    print("modal serve 06_gpu_and_ml/multitalk_native/api.py::fastapi_app")
    
    print("\n2. Use the API with curl:")
    curl_example = '''
# Single person generation
curl -X POST "https://your-modal-app.modal.run/generate/single" \\
  -F "audio=@examples/audio/person1.wav" \\
  -F "image=@examples/images/person1.jpg" \\
  -F "prompt=A person speaking naturally" \\
  -F "use_lora=fusionx" \\
  -F "use_teacache=true"
    '''
    print(curl_example)
    
    print("\n3. Use the API with Python requests:")
    python_api_example = '''
import requests

# Single person generation
with open("examples/audio/person1.wav", "rb") as audio_file, \\
     open("examples/images/person1.jpg", "rb") as image_file:
    
    response = requests.post(
        "https://your-modal-app.modal.run/generate/single",
        files={
            "audio": audio_file,
            "image": image_file
        },
        data={
            "prompt": "A person speaking naturally",
            "use_lora": "fusionx",
            "use_teacache": True,
            "sample_steps": 8
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            # Decode and save video
            import base64
            video_data = base64.b64decode(result["video_data"])
            with open("api_output.mp4", "wb") as f:
                f.write(video_data)
            print("Video generated via API!")
    '''
    print(python_api_example)

def example_gradio_interface():
    """Example: Gradio web interface"""
    print("\n🖥️  Example: Gradio Web Interface")
    print("="*50)
    
    print("1. Start the Gradio interface:")
    print("modal serve 06_gpu_and_ml/multitalk_native/multitalk_native.py::gradio_interface")
    
    print("\n2. Open the provided URL in your browser")
    print("3. Upload audio and image files")
    print("4. Configure generation settings")
    print("5. Click 'Generate Video' and wait for results")

def example_performance_comparison():
    """Example: Performance comparison between different configurations"""
    print("\n⚡ Performance Comparison Examples")
    print("="*50)
    
    configurations = [
        {
            "name": "Standard Quality (Best)",
            "config": {
                "sample_steps": 40,
                "use_lora": None,
                "use_quantization": False,
                "use_teacache": True
            },
            "speed": "1x",
            "memory": "~24GB",
            "quality": "Best"
        },
        {
            "name": "FusionX Acceleration",
            "config": {
                "sample_steps": 8,
                "use_lora": "fusionx",
                "use_quantization": False,
                "use_teacache": True
            },
            "speed": "~2x",
            "memory": "~24GB", 
            "quality": "Very Good"
        },
        {
            "name": "LightX2V Acceleration",
            "config": {
                "sample_steps": 4,
                "use_lora": "lightx2v",
                "use_quantization": False,
                "use_teacache": True
            },
            "speed": "~4x",
            "memory": "~24GB",
            "quality": "Good"
        },
        {
            "name": "Low VRAM Mode",
            "config": {
                "sample_steps": 20,
                "use_lora": None,
                "use_quantization": True,
                "low_vram": True,
                "use_teacache": True
            },
            "speed": "~0.8x",
            "memory": "~12GB",
            "quality": "Good"
        }
    ]
    
    for config in configurations:
        print(f"\n{config['name']}:")
        print(f"  Speed: {config['speed']}")
        print(f"  Memory: {config['memory']}")
        print(f"  Quality: {config['quality']}")
        print(f"  Config: {config['config']}")

def main():
    """Run all examples"""
    print("🚀 MultiTalk Native - Usage Examples")
    print("="*60)
    
    example_single_person_cli()
    example_single_person_python()
    example_multi_person_python()
    example_low_vram_setup()
    example_rest_api_usage()
    example_gradio_interface()
    example_performance_comparison()
    
    print("\n" + "="*60)
    print("📚 For more information, see:")
    print("- README.md for detailed documentation")
    print("- test_multitalk.py for testing examples")
    print("- deploy.py for deployment instructions")
    print("="*60)

if __name__ == "__main__":
    main()