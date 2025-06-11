import modal
import os
import subprocess
from pathlib import Path
import tempfile

# Define the Modal app
app = modal.App("hunyuan-video-avatar")

# Create a custom image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git",
        "wget",
        "curl",
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "libgcc-s1",
        "build-essential"
    ])
    # Install PyTorch first (CUDA version)
    .pip_install([
        "torch==2.4.0",
        "torchvision==0.19.0", 
        "torchaudio==2.4.0"
    ], extra_index_url="https://download.pytorch.org/whl/cu121")
    # Then install other packages that depend on PyTorch
    .pip_install([
        "xformers",
        "transformers>=4.25.0",
        "diffusers>=0.21.0",
        "accelerate>=0.20.0"
    ])
    # Install remaining dependencies
    .pip_install([
        "opencv-python",
        "pillow",
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "omegaconf",
        "safetensors",
        "huggingface-hub",
        "datasets",
        "gradio",
        "imageio",
        "imageio-ffmpeg",
        "decord",
        "av",
        "einops",
        "timm",
        "ftfy",
        "tensorboard",
        "soundfile",
        "librosa",
        "psutil"
    ])
    # Clone the repository and install its requirements
    .run_commands([
        "git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git /app",
        "cd /app && pip install -r requirements.txt || echo 'No requirements.txt found or installation failed, continuing...'"
    ])
)

# Mount for persistent storage of models
volume = modal.Volume.from_name("hunyuan-models", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-80GB",  # Use string format for GPU specification
    volumes={"/models": volume},
    timeout=3600,  # 1 hour timeout
    memory=64*1024,  # 64GB RAM
)
def setup_models():
    """Download and setup required models"""
    import subprocess
    import os
    
    os.chdir("/app")
    
    # Create models directory if it doesn't exist
    os.makedirs("/models", exist_ok=True)
    
    # Download models (adjust based on actual model requirements)
    try:
        # Download HunyuanVideo-Avatar models from Hugging Face
        subprocess.run([
            "python", "-c", 
            "from huggingface_hub import snapshot_download; "
            "snapshot_download('tencent/HunyuanVideo-Avatar', local_dir='/models/hunyuan-avatar')"
        ], check=True)
        
        # Also download base HunyuanVideo model if needed
        subprocess.run([
            "python", "-c", 
            "from huggingface_hub import snapshot_download; "
            "snapshot_download('tencent/HunyuanVideo', local_dir='/models/hunyuan-video')"
        ], check=True)
        
        print("Models downloaded successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading models: {e}")
        print("You may need to manually download the models or check authentication")
        # Continue without failing
        pass
    
    return "Setup complete"

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/models": volume},
    timeout=1800,  # 30 minutes
    memory=64*1024,  # 64GB RAM
)
def generate_avatar_video(
    input_image_url: str,
    audio_url: str = None,
    prompt: str = "A person speaking naturally",
    num_frames: int = 120,
    height: int = 512,
    width: int = 512,
    fps: int = 24
):
    """Generate avatar video from input image and audio/prompt"""
    import requests
    import tempfile
    import os
    import subprocess
    
    os.chdir("/app")
    
    # Create temporary directory for this job
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download input image
        img_response = requests.get(input_image_url)
        input_image_path = os.path.join(temp_dir, "input_image.jpg")
        with open(input_image_path, "wb") as f:
            f.write(img_response.content)
        
        # Download audio if provided
        audio_path = None
        if audio_url:
            audio_response = requests.get(audio_url)
            audio_path = os.path.join(temp_dir, "input_audio.wav")
            with open(audio_path, "wb") as f:
                f.write(audio_response.content)
        
        # Prepare output path
        output_path = os.path.join(temp_dir, "output_video.mp4")
        
        # Build command for HunyuanVideo-Avatar
        cmd = [
            "python", "inference.py",
            "--image_path", input_image_path,
            "--prompt", prompt,
            "--output_path", output_path,
            "--num_frames", str(num_frames),
            "--height", str(height),
            "--width", str(width),
            "--fps", str(fps)
        ]
        
        # Add audio if provided
        if audio_path:
            cmd.extend(["--audio_path", audio_path])
        
        try:
            # Run the inference
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Inference output:", result.stdout)
            
            # Read the output video
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    video_data = f.read()
                return {
                    "success": True,
                    "video_data": video_data,
                    "message": "Video generated successfully!"
                }
            else:
                return {
                    "success": False,
                    "error": "Output video not found",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Inference failed: {e}",
                "stdout": e.stdout,
                "stderr": e.stderr
            }

@app.function(
    image=image,
    volumes={"/models": volume}
)
def create_gradio_interface():
    """Create a Gradio web interface for the avatar generation"""
    import gradio as gr
    import requests
    import base64
    import io
    
    def generate_video_interface(input_image, audio_file, prompt, num_frames, height, width, fps):
        try:
            # Convert image to URL (you'd need to upload to a temporary service or encode)
            # For simplicity, this is a placeholder - you'd need to handle file uploads properly
            
            # Call the generation function
            result = generate_avatar_video.remote(
                input_image_url="placeholder",  # You'd need to handle this properly
                audio_url=None if not audio_file else "placeholder",
                prompt=prompt,
                num_frames=int(num_frames),
                height=int(height),
                width=int(width),
                fps=int(fps)
            )
            
            if result["success"]:
                # Return the video
                return result["video_data"], "Success!"
            else:
                return None, f"Error: {result['error']}"
                
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=generate_video_interface,
        inputs=[
            gr.Image(type="filepath", label="Input Face Image"),
            gr.Audio(type="filepath", label="Audio (optional)"),
            gr.Textbox(value="A person speaking naturally", label="Prompt"),
            gr.Slider(minimum=24, maximum=240, value=120, step=8, label="Number of Frames"),
            gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height"),
            gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width"),
            gr.Slider(minimum=12, maximum=30, value=24, step=1, label="FPS")
        ],
        outputs=[
            gr.Video(label="Generated Avatar Video"),
            gr.Textbox(label="Status")
        ],
        title="HunyuanVideo-Avatar Generator",
        description="Generate talking avatar videos from a face image and audio/prompt"
    )
    
    return interface

@app.local_entrypoint()
def main():
    """Main entry point for local development"""
    print("Setting up HunyuanVideo-Avatar on Modal...")
    
    # Setup models first
    print("Downloading models...")
    setup_result = setup_models.remote()
    print(f"Setup result: {setup_result}")
    
    print("HunyuanVideo-Avatar is ready!")
    print("You can now call the generate_avatar_video function with your inputs.")

# Web interface endpoint
@app.function(
    image=image,
    volumes={"/models": volume}
)
@modal.concurrent(max_inputs=10)
@modal.fastapi_endpoint(method="GET")
def web_interface():
    """Web endpoint to serve the Gradio interface"""
    interface = create_gradio_interface()
    return interface.launch(server_name="0.0.0.0", server_port=8000, share=False)

if __name__ == "__main__":
    # For local testing
    with app.run():
        main()