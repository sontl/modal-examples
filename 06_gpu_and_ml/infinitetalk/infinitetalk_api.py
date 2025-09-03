"""
InfiniteTalk Modal API (single GPU)
- Builds a Modal image with InfiniteTalk repo and dependencies
- Caches models in a Modal Volume
- Exposes a FastAPI endpoint to generate video from image+audio
"""
import os
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Optional

import modal

# Prebuilt wheel for flash-attn matching torch/cu121 used
FLASH_ATTN_WHL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "espeak",  # for some audio backends
        "espeak-data",
    )
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "xformers==0.0.28",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # core deps from README
        "transformers==4.52",
        "diffusers",
        "accelerate",
        "safetensors",
        "opencv-python",
        "pillow",
        "numpy<2.0",
        "scipy",
        "soundfile",
        "moviepy==1.0.3",
        "fastapi",
        "python-multipart",
        "librosa",
        "misaki[en]",
        "psutil",
        "packaging",
        "ninja",
        "huggingface_hub[hf_transfer]",
        "easydict",
        "ftfy",
        "scikit-image",
        "loguru",
        "optimum-quanto==0.2.6",
        "pyloudnorm",
        FLASH_ATTN_WHL,
        "tokenizers>=0.20.3",
        "tqdm",
        "imageio",
        "imageio-ffmpeg",
        "dashscope",
        "uvicorn[standard]",
        "xfuser>=0.4.1",
        "gradio>=5.0.0",
        "scenedetect",
        "decord"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(
        "mkdir -p /app",
        # Clone InfiniteTalk into /app
        "cd /tmp && git clone https://github.com/MeiGen-AI/InfiniteTalk.git",
        "cp -r /tmp/InfiniteTalk/* /app/",
    )
)

# Pre-download model weights during image build to avoid cold-start downloads
def hf_download():
    """
    Download all required InfiniteTalk model assets during image build.
    Files are placed under /weights to be baked into the image layer.
    """
    import os
    from huggingface_hub import snapshot_download, hf_hub_download

    os.makedirs("/weights/Wan2.1-I2V-14B-480P", exist_ok=True)
    os.makedirs("/weights/chinese-wav2vec2-base", exist_ok=True)
    os.makedirs("/weights/InfiniteTalk", exist_ok=True)

    # Base video model
    snapshot_download(
        repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
        local_dir="/weights/Wan2.1-I2V-14B-480P",
    )

    # wav2vec2 base, plus attempt PR #1 model.safetensors
    snapshot_download(
        repo_id="TencentGameMate/chinese-wav2vec2-base",
        local_dir="/weights/chinese-wav2vec2-base",
    )
    safep = "/weights/chinese-wav2vec2-base/model.safetensors"
    if not os.path.exists(safep):
        try:
            hf_hub_download(
                repo_id="TencentGameMate/chinese-wav2vec2-base",
                filename="model.safetensors",
                local_dir="/weights/chinese-wav2vec2-base",
                revision="refs/pr/1",
            )
        except Exception as e:
            print("Warning: wav2vec2 PR #1 model.safetensors not available:", e)

    # InfiniteTalk weights (contains single/infinitetalk.safetensors)
    snapshot_download(
        repo_id="MeiGen-AI/InfiniteTalk",
        local_dir="/weights/InfiniteTalk",
    )

# Extend image to include huggingface_hub and run the download at build time
image = (
    image.pip_install("huggingface_hub[hf_transfer]>=0.34.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(hf_download)
)

# Shared cache for models
vol = modal.Volume.from_name("infinitetalk-cache", create_if_missing=True)
app = modal.App(name="infinitetalk-api", image=image)


@app.cls(
    gpu="L40S",
    volumes={"/data": vol},
    timeout=3600,
    scaledown_window=60,
)
class InfiniteTalkRunner:
    base_dir: str = modal.parameter(default="/weights/Wan2.1-I2V-14B-480P")
    wav2vec_dir: str = modal.parameter(default="/weights/chinese-wav2vec2-base")
    italk_dir: str = modal.parameter(default="/weights/InfiniteTalk")

    @modal.enter()
    def setup(self):
        import sys
        print("Setting up InfiniteTalk environment...")
        sys.path.insert(0, "/app")
        os.chdir("/app")
        print("Files in /app:", [p.name for p in Path("/app").glob("*")])
        # Models should already be baked into the image under /weights
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
            os.path.exists(model_file)
            and os.path.getsize(model_file) > 0
        ) or (
            os.path.exists(pytorch_file)
            and os.path.getsize(pytorch_file) > 0
        )
        return has_model and os.path.exists(cfg) and os.path.getsize(cfg) > 0

    def _check_infinitetalk(self) -> bool:
        # single-gpu weight file expected path
        path = f"{self.italk_dir}/single/infinitetalk.safetensors"
        return os.path.exists(path) and os.path.getsize(path) > 0

    def _all_models_ready(self) -> bool:
        ok = self._check_base_model() and self._check_wav2vec() and self._check_infinitetalk()
        print(
            f"Base:{self._check_base_model()} Wav2Vec:{self._check_wav2vec()} InfiniteTalk:{self._check_infinitetalk()}"
        )
        return ok

    def _download_models(self):
        # No-op: models are downloaded at image build time into /weights
        print("Download skipped; models baked into image at /weights.")

    def _verify_all(self) -> bool:
        files = [
            f"{self.base_dir}/diffusion_pytorch_model.safetensors.index.json",
            f"{self.italk_dir}/single/infinitetalk.safetensors",
        ]
        # wav2vec either file
        w1 = f"{self.wav2vec_dir}/model.safetensors"
        w2 = f"{self.wav2vec_dir}/pytorch_model.bin"
        if os.path.exists(w1):
            files.append(w1)
        elif os.path.exists(w2):
            files.append(w2)
        else:
            print("Missing wav2vec model.safetensors/pytorch_model.bin")
            return False
        ok = True
        for p in files:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                print("✓", p)
            else:
                print("✗ Missing or empty:", p)
                ok = False
        return ok

    @modal.method()
    def generate_video(
        self,
        audio_data: bytes,
        image_data: bytes,
        prompt: str = "A person talking naturally",
        sample_steps: int = 40,
        use_teacache: bool = True,
        low_vram: bool = False,
        size: str = "infinitetalk-480",  # or infinitetalk-720
        motion_frame: int = 9,
        mode: str = "streaming",  # or "clip"
    ) -> Dict[str, str]:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as af:
                af.write(audio_data)
                audio_path = os.path.abspath(af.name)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as imf:
                imf.write(image_data)
                image_path = os.path.abspath(imf.name)

            # Create minimal JSON input matching InfiniteTalk examples
            input_json = {
                "prompt": prompt,
                "cond_video": image_path,
                # Single-person format expected by InfiniteTalk examples
                "cond_audio": {"person1": audio_path},
            }
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as jf:
                json.dump(input_json, jf, indent=2)
                json_path = jf.name

            if not os.path.exists("/app/generate_infinitetalk.py"):
                return {
                    "success": False,
                    "error": "generate_infinitetalk.py not found",
                    "app_files": [str(p) for p in Path("/app").glob("*")],
                }
            if not self._verify_all():
                return {
                    "success": False,
                    "error": "Required model files missing or empty",
                }

            out_name = f"italk_{abs(hash(str(input_json))) }"
            cmd = [
                "python",
                "generate_infinitetalk.py",
                "--ckpt_dir",
                self.base_dir,
                "--wav2vec_dir",
                self.wav2vec_dir,
                "--infinitetalk_dir",
                f"{self.italk_dir}/single/infinitetalk.safetensors",
                "--input_json",
                json_path,
                "--size",
                size,
                "--sample_steps",
                str(sample_steps),
                "--mode",
                mode,
                "--motion_frame",
                str(motion_frame),
                "--save_file",
                out_name,
            ]
            if use_teacache:
                cmd.append("--use_teacache")
            if low_vram:
                cmd.extend(["--num_persistent_param_in_dit", "0"])

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
            print("STDOUT:\n", result.stdout)
            if result.stderr:
                print("STDERR:\n", result.stderr)
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Return code {result.returncode}",
                    "stderr": result.stderr,
                    "stdout": result.stdout,
                }

            # InfiniteTalk saves as {out_name}.mp4 in /app
            out_path = f"/app/{out_name}.mp4"
            if not os.path.exists(out_path):
                # Fallback: pick latest mp4
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

            # Return as base64 to keep API simple
            import base64
            with open(out_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            # Cleanup temp inputs and output
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


# FastAPI app wrapper
@app.function(image=image, volumes={"/data": vol}, timeout=1800)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.responses import JSONResponse

    web = FastAPI(title="InfiniteTalk API", version="1.0.0")
    runner = InfiniteTalkRunner()

    @web.get("/")
    async def root():
        return {"message": "InfiniteTalk API (single GPU)", "default_size": "infinitetalk-480"}

    @web.post("/generate")
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
        res = runner.generate_video.remote(
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

    return web
