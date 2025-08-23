# syntax=docker/dockerfile:1.7
# InfiniteTalk Dockerfile for RunPod and other platforms
# Multi-stage build for optimized image size

###############################################################################
# BUILD STAGE
###############################################################################
ARG CUDA_VERSION=12.1
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

# Install build dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment and install Python packages
ENV VENV_PATH=/opt/venv
RUN python3.10 -m venv ${VENV_PATH} && \
    . ${VENV_PATH}/bin/activate && \
    pip install --upgrade --no-cache-dir pip setuptools wheel

# Install PyTorch and core ML stack
RUN . ${VENV_PATH}/bin/activate && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1 \
        torchvision==0.19.1 \
        torchaudio==2.4.1 \
        xformers==0.0.28

# Install flash-attention
RUN . ${VENV_PATH}/bin/activate && \
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install other dependencies
RUN . ${VENV_PATH}/bin/activate && \
    pip install --no-cache-dir \
        transformers==4.52 \
        diffusers \
        accelerate \
        safetensors \
        opencv-python \
        pillow \
        "numpy<2.0" \
        scipy \
        soundfile \
        moviepy==1.0.3 \
        fastapi \
        python-multipart \
        librosa \
        "misaki[en]" \
        psutil \
        packaging \
        ninja \
        "huggingface_hub[hf_transfer]" \
        easydict \
        ftfy \
        scikit-image \
        loguru \
        optimum-quanto==0.2.6 \
        pyloudnorm \
        "tokenizers>=0.20.3" \
        tqdm \
        imageio \
        imageio-ffmpeg \
        dashscope \
        "uvicorn[standard]" \
        "xfuser>=0.4.1" \
        "gradio>=5.0.0" \
        scenedetect \
        decord

# Clean up pip cache and unnecessary files
RUN . ${VENV_PATH}/bin/activate && \
    find ${VENV_PATH} -name '__pycache__' -type d -exec rm -rf {} + && \
    pip cache purge

# Clone InfiniteTalk repository
WORKDIR /workspace
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git

# Download model weights
RUN . ${VENV_PATH}/bin/activate && \
    mkdir -p /weights && \
    python3 -c "
import os
from huggingface_hub import snapshot_download, hf_hub_download

print('Downloading Wan2.1-I2V-14B-480P...')
snapshot_download(
    repo_id='Wan-AI/Wan2.1-I2V-14B-480P',
    local_dir='/weights/Wan2.1-I2V-14B-480P',
)

print('Downloading chinese-wav2vec2-base...')
snapshot_download(
    repo_id='TencentGameMate/chinese-wav2vec2-base',
    local_dir='/weights/chinese-wav2vec2-base',
)

try:
    hf_hub_download(
        repo_id='TencentGameMate/chinese-wav2vec2-base',
        filename='model.safetensors',
        local_dir='/weights/chinese-wav2vec2-base',
        revision='refs/pr/1',
    )
    print('Downloaded model.safetensors from PR #1')
except Exception as e:
    print(f'Warning: wav2vec2 PR #1 model.safetensors not available: {e}')

print('Downloading InfiniteTalk weights...')
snapshot_download(
    repo_id='MeiGen-AI/InfiniteTalk',
    local_dir='/weights/InfiniteTalk',
)

print('All models downloaded successfully!')
"

###############################################################################
# RUNTIME STAGE
###############################################################################
ARG CUDA_VERSION=12.1
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Install runtime system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.10 \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        espeak \
        espeak-data \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Copy the pre-built Python virtual environment
COPY --from=builder /opt/venv /opt/venv

# Copy InfiniteTalk code and model weights
COPY --from=builder /workspace/InfiniteTalk /app
COPY --from=builder /weights /app/weights

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /app

# Copy FastAPI server script
COPY <<EOF /app/server.py
"""
FastAPI server for InfiniteTalk - compatible with RunPod and other platforms
"""
import os
import json
import tempfile
import subprocess
import base64
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="InfiniteTalk API", version="1.0.0")

class InfiniteTalkRunner:
    def __init__(self):
        self.base_dir = "/app/weights/Wan2.1-I2V-14B-480P"
        self.wav2vec_dir = "/app/weights/chinese-wav2vec2-base"
        self.italk_dir = "/app/weights/InfiniteTalk"
        
        # Set up environment
        os.chdir("/app")
        print("InfiniteTalk server initialized")
        print("Model readiness:", self._all_models_ready())

    def _check_base_model(self) -> bool:
        req = [
            f"{self.base_dir}/diffusion_pytorch_model.safetensors.index.json",
            f"{self.base_dir}/config.json",
        ]
        return all(os.path.exists(p) and os.path.getsize(p) > 0 for p in req)

    def _check_wav2vec(self) -> bool:
        model_file = f"{self.wav2vec_dir}/model.safetensors"
        pytorch_file = f"{self.wav2vec_dir}/pytorch_model.bin"
        cfg = f"{self.wav2vec_dir}/config.json"
        has_model = (
            os.path.exists(model_file) and os.path.getsize(model_file) > 0
        ) or (
            os.path.exists(pytorch_file) and os.path.getsize(pytorch_file) > 0
        )
        return has_model and os.path.exists(cfg) and os.path.getsize(cfg) > 0

    def _check_infinitetalk(self) -> bool:
        path = f"{self.italk_dir}/single/infinitetalk.safetensors"
        return os.path.exists(path) and os.path.getsize(path) > 0

    def _all_models_ready(self) -> bool:
        ok = self._check_base_model() and self._check_wav2vec() and self._check_infinitetalk()
        print(f"Base:{self._check_base_model()} Wav2Vec:{self._check_wav2vec()} InfiniteTalk:{self._check_infinitetalk()}")
        return ok

    def generate_video(
        self,
        audio_data: bytes,
        image_data: bytes,
        prompt: str = "A person talking naturally",
        sample_steps: int = 40,
        use_teacache: bool = True,
        low_vram: bool = False,
        size: str = "infinitetalk-480",
        motion_frame: int = 9,
        mode: str = "streaming",
    ) -> Dict[str, str]:
        try:
            # Save uploaded files
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as af:
                af.write(audio_data)
                audio_path = os.path.abspath(af.name)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as imf:
                imf.write(image_data)
                image_path = os.path.abspath(imf.name)

            # Create input JSON
            input_json = {
                "prompt": prompt,
                "cond_image": image_path,
                "cond_audio": {"person1": audio_path},
            }
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as jf:
                json.dump(input_json, jf, indent=2)
                json_path = jf.name

            # Verify models and script
            if not os.path.exists("/app/generate_infinitetalk.py"):
                return {
                    "success": False,
                    "error": "generate_infinitetalk.py not found",
                    "app_files": [str(p) for p in Path("/app").glob("*")],
                }
            if not self._all_models_ready():
                return {
                    "success": False,
                    "error": "Required model files missing or empty",
                }

            # Build command
            out_name = f"italk_{abs(hash(str(input_json)))}"
            cmd = [
                "python",
                "generate_infinitetalk.py",
                "--ckpt_dir", self.base_dir,
                "--wav2vec_dir", self.wav2vec_dir,
                "--infinitetalk_dir", f"{self.italk_dir}/single/infinitetalk.safetensors",
                "--input_json", json_path,
                "--size", size,
                "--sample_steps", str(sample_steps),
                "--mode", mode,
                "--motion_frame", str(motion_frame),
                "--save_file", out_name,
            ]
            if use_teacache:
                cmd.append("--use_teacache")
            if low_vram:
                cmd.extend(["--num_persistent_param_in_dit", "0"])

            # Set environment
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": "/app",
                "CUDA_VISIBLE_DEVICES": "0",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            })

            print("Command:", " ".join(cmd))
            result = subprocess.run(
                cmd,
                cwd="/app",
                capture_output=True,
                text=True,
                env=env,
                timeout=1800,
            )
            
            print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Return code {result.returncode}",
                    "stderr": result.stderr,
                    "stdout": result.stdout,
                }

            # Find output video
            out_path = f"/app/{out_name}.mp4"
            if not os.path.exists(out_path):
                mp4s = sorted(Path("/app").glob("*.mp4"), key=lambda p: p.stat().st_mtime)
                if mp4s:
                    out_path = str(mp4s[-1])
                else:
                    return {
                        "success": False,
                        "error": f"No MP4 produced at {out_path}",
                        "stdout": result.stdout,
                    }
            
            if os.path.getsize(out_path) == 0:
                return {"success": False, "error": "Output MP4 is empty"}

            # Return as base64
            with open(out_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            # Cleanup
            for p in [audio_path, image_path, json_path, out_path]:
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

            return {
                "success": True,
                "video_data": b64,
                "filename": os.path.basename(out_path),
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Generation timed out"}
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

# Initialize runner
runner = InfiniteTalkRunner()

@app.get("/")
async def root():
    return {"message": "InfiniteTalk API", "status": "ready", "models_ready": runner._all_models_ready()}

@app.get("/health")
async def health():
    return {"status": "healthy", "models_ready": runner._all_models_ready()}

@app.post("/generate")
async def generate(
    audio: UploadFile = File(...),
    image: UploadFile = File(...),
    prompt: str = Form("A person talking naturally"),
    sample_steps: int = Form(40),
    use_teacache: bool = Form(True),
    low_vram: bool = Form(False),
    size: str = Form("infinitetalk-480"),
    motion_frame: int = Form(9),
    mode: str = Form("streaming"),
):
    try:
        audio_bytes = await audio.read()
        image_bytes = await image.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded files")
    
    print("Generating video...")
    res = runner.generate_video(
        audio_data=audio_bytes,
        image_data=image_bytes,
        prompt=prompt,
        sample_steps=sample_steps,
        use_teacache=use_teacache,
        low_vram=low_vram,
        size=size,
        motion_frame=motion_frame,
        mode=mode,
    )
    
    if not res.get("success"):
        return JSONResponse(status_code=500, content=res)
    return res

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Create entrypoint script
COPY <<EOF /app/entrypoint.sh
#!/bin/bash
set -e

echo "Starting InfiniteTalk API server..."
echo "CUDA devices: \$CUDA_VISIBLE_DEVICES"
echo "GPU info:"
nvidia-smi || echo "No GPU info available"

# Start the FastAPI server
exec python /app/server.py
EOF

RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
