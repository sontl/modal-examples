import modal
import os
import subprocess

# Define the Modal app
app = modal.App("hunyuan-video-avatar")

# Define the image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11.9")
    .apt_install("git", "wget", "ffmpeg")  # Install system dependencies
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        "pytorch-cuda==12.4",
        "huggingface_hub",
        "diffusers",
        "transformers",
        "numpy",
        "opencv-python",
        "soundfile",
        "librosa",
        "tqdm",
    )  # Install Python dependencies
    .run_commands(
        # Install huggingface-cli
        "pip install huggingface_hub[cli]",
        # Clone the HunyuanVideo-Avatar repository
        "git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git /HunyuanVideo-Avatar",
        # Download model weights
        "cd /HunyuanVideo-Avatar/weights && huggingface-cli download tencent/HunyuanVideo-Avatar --local-dir ."
    )
)

# Define the Modal function
@app.function(
    image=image,
    gpu="A100-80GB",  # Use A100 with 80GB (or H100:96GB for better performance)
    timeout=3600,  # Set timeout to 1 hour to account for long inference times
)
def run_hunyuan_video_avatar(image_path: str, audio_path: str, prompt: str):
    # Change to the repository directory
    os.chdir("/HunyuanVideo-Avatar")

    # Set environment variables
    os.environ["PYTHONPATH"] = "./"
    os.environ["MODEL_BASE"] = "./weights"
    os.environ["DISABLE_SP"] = "1"  # For single-GPU mode
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU

    # Define paths and parameters
    checkpoint_path = "./weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
    output_basepath = "./results"
    os.makedirs(output_basepath, exist_ok=True)

    # Prepare the input CSV file (required by sample_gpu_poor.py)
    input_csv = "assets/test.csv"
    with open(input_csv, "w") as f:
        f.write(f"image_path,audio_path,prompt\n{image_path},{audio_path},{prompt}\n")

    # Run the inference command
    cmd = [
        "python3",
        "hymm_sp/sample_gpu_poor.py",
        "--input",
        input_csv,
        "--ckpt",
        checkpoint_path,
        "--sample-n-frames",
        "129",
        "--seed",
        "128",
        "--image-size",
        "704",
        "--cfg-scale",
        "7.5",
        "--infer-steps",
        "50",
        "--use-deepcache",
        "1",
        "--flow-shift-eval-video",
        "5.0",
        "--save-path",
        output_basepath,
        "--use-fp8",
        "--infer-min",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Inference completed successfully!")
        print(result.stdout)

        # Return the path to the generated video
        output_video = os.path.join(output_basepath, "output.mp4")  # Adjust based on actual output
        if os.path.exists(output_video):
            return output_video
        else:
            raise FileNotFoundError("Generated video not found!")
    except subprocess.CalledProcessError as e:
        print(f"Error during inference: {e.stderr}")
        raise

# Local entrypoint for testing
@app.local_entrypoint()
def main():
    # Example inputs (replace with actual paths or upload to Modal's volume)
    image_path = "assets/image/src1.png"  # Example image path
    audio_path = "assets/audio/4.WAV"     # Example audio path
    prompt = "a person delivering a calm, reassuring message"

    # Call the function
    output = run_hunyuan_video_avatar.remote(image_path, audio_path, prompt)
    print(f"Generated video saved at: {output}")
