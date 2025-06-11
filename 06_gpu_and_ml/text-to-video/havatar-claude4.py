import modal
import os
import subprocess
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import io

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

# Create a web image specifically for the Gradio interface
web_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "fastapi[standard]==0.115.4",
    "gradio~=5.7.1",
    "pillow~=10.2.0",
    "pydantic==2.10.6",
    "soundfile~=0.12.1",
    "numpy~=1.24.3"
)

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
    image=web_image,  # Use the web image instead of the main image
    min_containers=1,
    scaledown_window=60 * 20,
    # gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow it to scale to 100 concurrent inputs
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web_interface():
    """Web endpoint to serve the Gradio interface"""
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    
    def generate_video_interface(input_image, audio_file, prompt, num_frames, height, width, fps):
        try:
            # Convert image to temporary file
            if input_image is not None:
                import tempfile
                import os
                from PIL import Image
                import io
                
                # Save image to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    if isinstance(input_image, str):
                        # If it's already a file path
                        input_image_path = input_image
                    else:
                        # If it's PIL Image or numpy array
                        if isinstance(input_image, np.ndarray):
                            input_image = Image.fromarray(input_image)
                        input_image.save(tmp.name)
                        input_image_path = tmp.name

            # Handle audio file
            audio_path = None
            if audio_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    if isinstance(audio_file, tuple):
                        # If it's a tuple (sampling_rate, audio_data)
                        import soundfile as sf
                        sampling_rate, audio_data = audio_file
                        sf.write(tmp.name, audio_data, sampling_rate)
                        audio_path = tmp.name
                    elif hasattr(audio_file, 'name'):
                        # If it's a file object
                        audio_path = audio_file.name
                    else:
                        # If it's a path string
                        audio_path = audio_file
            
            # Call the generation function
            result = generate_avatar_video.remote(
                input_image_url=input_image_path,
                audio_url=audio_path,
                prompt=prompt,
                num_frames=int(num_frames),
                height=int(height),
                width=int(width),
                fps=int(fps)
            )
            
            # Clean up temporary files
            if input_image is not None and os.path.exists(input_image_path):
                os.unlink(input_image_path)
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
            
            if result["success"]:
                # Convert video data to file-like object
                video_bytes = io.BytesIO(result["video_data"])
                return video_bytes, "Success!"
            else:
                return None, f"Error: {result['error']}"
                
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            return None, error_msg
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=generate_video_interface,
        inputs=[
            gr.Image(label="Input Face Image"),
            gr.Audio(label="Audio (optional)"),
            gr.Textbox(value="A person speaking naturally", label="Prompt"),
            gr.Number(value=120, label="Number of Frames"),
            gr.Number(value=512, label="Height"),
            gr.Number(value=512, label="Width"),
            gr.Number(value=24, label="FPS")
        ],
        outputs=[
            gr.Video(label="Generated Avatar Video"),
            gr.Textbox(label="Status")
        ],
        title="HunyuanVideo-Avatar Generator",
        description="Generate talking avatar videos from a face image and audio/prompt",
        theme="soft",
        allow_flagging="never",
    )
    
    # Mount the Gradio app on FastAPI
    app = FastAPI()
    return mount_gradio_app(app=app, blocks=interface, path="/")

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

if __name__ == "__main__":
    # For local testing
    with app.run():
        main()