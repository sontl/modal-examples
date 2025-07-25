import modal
import os
import subprocess
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import io
from urllib.parse import urlparse

# Define the Modal app first
app = modal.App("hunyuan-video-avatar")

# Define constants
MODEL_PATH = "/models"  # where the Volume will appear on our Functions' filesystems

# Define the volume
model_volume = modal.Volume.from_name("hunyuan-models", create_if_missing=True)

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
    # Update the image to set the environment variables for model caching
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster downloads
        "HF_HUB_CACHE": MODEL_PATH,
    })
)

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
    gpu="A100-80GB",
    volumes={MODEL_PATH: model_volume},
    timeout=3600,
    memory=64*1024,
)
def setup_models():
    """Download and setup required models"""
    import subprocess
    import os
    
    os.chdir("/app")
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs("./weights/ckpts/hunyuan-video-t2v-720p/transformers", exist_ok=True)
    
    # Download models
    try:
        subprocess.run([
            "python", "-c", 
            "from huggingface_hub import snapshot_download; "
            "snapshot_download('tencent/HunyuanVideo-Avatar', local_dir='./weights')"
        ], check=True)
        
        print("Models downloaded successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading models: {e}")
        print("You may need to manually download the models or check authentication")
        pass
    
    return "Setup complete"

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={MODEL_PATH: model_volume},
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
    
    # Set environment variables
    os.environ["PYTHONPATH"] = "./"
    os.environ["MODEL_BASE"] = "./weights"
    os.environ["DISABLE_SP"] = "1"  # For single-GPU mode
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU
    
    # Create assets directory if it doesn't exist
    os.makedirs("assets", exist_ok=True)
    
    # Create test.csv in the assets directory
    test_csv_path = os.path.join("assets", "test.csv")
    
    # Create temporary directory for this job
    with tempfile.TemporaryDirectory() as temp_dir:
        # Handle input image - could be URL or local path
        input_image_path = os.path.join(temp_dir, "input_image.jpg")
        
        # Check if input_image_url is a URL or local path
        parsed_url = urlparse(input_image_url)
        if parsed_url.scheme in ('http', 'https'):
            # It's a URL, download it
            img_response = requests.get(input_image_url)
            with open(input_image_path, "wb") as f:
                f.write(img_response.content)
        else:
            # It's a local path, just use it directly
            input_image_path = input_image_url
        
        # Handle audio if provided - similar logic for URL vs local path
        audio_path = None
        if audio_url:
            parsed_audio_url = urlparse(audio_url)
            if parsed_audio_url.scheme in ('http', 'https'):
                # It's a URL, download it
                audio_response = requests.get(audio_url)
                audio_path = os.path.join(temp_dir, "input_audio.wav")
                with open(audio_path, "wb") as f:
                    f.write(audio_response.content)
            else:
                # It's a local path, use it directly
                audio_path = audio_url
        
        # Write to test.csv in the assets directory
        with open(test_csv_path, "w") as f:
            f.write("image_path,audio_path,prompt\n")
            f.write(f"{input_image_path},{audio_path if audio_path else ''},\"{prompt}\"\n")
        
        # Prepare output path
        output_path = os.path.join(temp_dir, "output")
        os.makedirs(output_path, exist_ok=True)
        
        # Define checkpoint path
        checkpoint_path = "./weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
        
        # Build command for HunyuanVideo-Avatar
        cmd = [
            "python3",
            "hymm_sp/sample_gpu_poor.py",
            "--input", test_csv_path,
            "--ckpt", checkpoint_path,
            "--sample-n-frames", str(num_frames),
            "--seed", "128",
            "--image-size", str(width),  # Using width as image size
            "--cfg-scale", "7.5",
            "--infer-steps", "50",
            "--use-deepcache", "1",
            "--flow-shift-eval-video", "5.0",
            "--save-path", output_path,
            "--use-fp8",
            "--infer-min"
        ]
        
        try:
            # Run the inference
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Inference output:", result.stdout)
            
            # Find the output video file
            output_video = None
            for file in os.listdir(output_path):
                if file.endswith(".mp4"):
                    output_video = os.path.join(output_path, file)
                    break
            
            if output_video and os.path.exists(output_video):
                with open(output_video, "rb") as f:
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
        flagging_mode="never"
    )
    
    # Mount the Gradio app on FastAPI
    app = FastAPI()
    return mount_gradio_app(app=app, blocks=interface, path="/")

@app.function(schedule=modal.Period(days=1))  # Optional: Run daily to check/update models
def init_models():
    return setup_models.remote()

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