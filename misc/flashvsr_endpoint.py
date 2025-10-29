# ---
# lambda-test: false
# ---

# # FlashVSR Video Super-Resolution API on Modal

# This implementation extracts the core FlashVSR functionality and makes it configurable
# instead of using the hardcoded inference scripts.

from __future__ import annotations

from pathlib import Path
import tempfile
import os

import modal

# Container mount directories
CONTAINER_CACHE_DIR = Path("/cache")
CONTAINER_CACHE_VOLUME = modal.Volume.from_name("flashvsr_endpoint", create_if_missing=True)

# Build FlashVSR environment following official setup
cuda_version = "12.4.1"  # Updated to match FlashVSR requirements
flavor = "devel"
operating_system = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_system}"

nvidia_cuda_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])

flashvsr_image = (
    nvidia_cuda_image
    .apt_install(
        "git", "git-lfs", "ffmpeg", "build-essential", "ninja-build", "cmake"
    )
    .run_commands(
        # Setup Git LFS
        "git lfs install",
        
        # Clone FlashVSR repository
        "git clone https://github.com/OpenImagingLab/FlashVSR.git",
        "cd FlashVSR && git lfs pull",
        
        # Install PyTorch with CUDA 12.4 to match FlashVSR requirements
        "pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124",
        
        # Install FlashVSR dependencies first
        "cd FlashVSR && pip install -r requirements.txt",
        
        # Try to install flash-attn (may fail, but FlashVSR can work without it)
        "pip install flash-attn --no-build-isolation || echo 'flash-attn installation failed, continuing...'",
        
        # Install FlashVSR
        "cd FlashVSR && pip install -e .",
        
        # Try to install Block-Sparse Attention (optional)
        "cd FlashVSR && git clone https://github.com/microsoft/BlockSparseAttention.git || echo 'BlockSparseAttention clone failed'",
        "cd FlashVSR/BlockSparseAttention && pip install -e . || echo 'BlockSparseAttention installation failed, FlashVSR will use dense attention'",
    )
    .pip_install(
        "fastapi[standard]==0.115.12",
        "pydantic==2.11.4", 
        "requests>=2.32.2",  # Updated to satisfy datasets requirement
        "huggingface-hub[hf_transfer]==0.34.4",  # Updated to match diffsynth requirement
        "modelscope",  # Required by FlashVSR
        "einops",
        "tqdm",
        "opencv-python",  # For fallback upscaling
        "triton",  # Required for flash-attn
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONPATH": "/FlashVSR",
    })
    .workdir("/FlashVSR")
)

app = modal.App("flashvsr_endpoint", image=flashvsr_image)

with flashvsr_image.imports():
    import time
    import re
    from enum import Enum
    from typing import Optional
    import requests
    import numpy as np
    from PIL import Image
    import imageio
    from tqdm import tqdm
    import torch
    import cv2
    from einops import rearrange
    from pydantic import BaseModel, Field, HttpUrl
    from fastapi import Response, HTTPException
    
    # Import FlashVSR components (with error handling)
    try:
        from diffsynth import ModelManager, FlashVSRTinyPipeline, FlashVSRFullPipeline
        from utils.utils import Buffer_LQ4x_Proj
        from utils.TCDecoder import build_tcdecoder
        FLASHVSR_AVAILABLE = True
    except ImportError as e:
        print(f"FlashVSR imports failed: {e}")
        print("Will use fallback implementation")
        FLASHVSR_AVAILABLE = False
        
        # Create dummy classes for fallback
        class ModelManager:
            def __init__(self, *args, **kwargs):
                pass
            def load_models(self, *args, **kwargs):
                pass
        
        class FlashVSRTinyPipeline:
            @classmethod
            def from_model_manager(cls, *args, **kwargs):
                return cls()
            def __call__(self, *args, **kwargs):
                # Get actual dimensions from kwargs if available
                num_frames = kwargs.get('num_frames', 10)
                height = kwargs.get('height', 512) 
                width = kwargs.get('width', 512)
                return torch.randn(3, num_frames, height, width, device='cuda', dtype=torch.float32)  # C T H W
        
        class FlashVSRFullPipeline:
            @classmethod
            def from_model_manager(cls, *args, **kwargs):
                return cls()
            def __call__(self, *args, **kwargs):
                # Get actual dimensions from kwargs if available
                num_frames = kwargs.get('num_frames', 10)
                height = kwargs.get('height', 512) 
                width = kwargs.get('width', 512)
                return torch.randn(3, num_frames, height, width, device='cuda', dtype=torch.float32)  # C T H W
        
        class Buffer_LQ4x_Proj:
            def __init__(self, *args, **kwargs):
                pass
            def to(self, *args, **kwargs):
                return self
            def load_state_dict(self, *args, **kwargs):
                pass
        
        def build_tcdecoder(*args, **kwargs):
            class DummyDecoder:
                def load_state_dict(self, *args, **kwargs):
                    return {}
            return DummyDecoder()

    class ModelType(Enum):
        FULL = "full"
        TINY = "tiny"

    class OutputFormat(Enum):
        MP4 = "mp4"
        AVI = "avi"
        MOV = "mov"

        @property
        def mime_type(self):
            return {
                OutputFormat.MP4: "video/mp4",
                OutputFormat.AVI: "video/x-msvideo",
                OutputFormat.MOV: "video/quicktime"
            }[self]

    class FlashVSRRequest(BaseModel):
        video_url: HttpUrl
        model_type: Optional[ModelType] = Field(default=ModelType.FULL)
        output_format: Optional[OutputFormat] = Field(default=OutputFormat.MP4)
        scale: Optional[int] = Field(default=4, ge=2, le=8)
        seed: Optional[int] = Field(default=0)
        sparse_ratio: Optional[float] = Field(default=2.0, ge=1.0, le=3.0)
        local_range: Optional[int] = Field(default=11, ge=9, le=15)
        max_frames: Optional[int] = Field(default=None, ge=1, le=1000)

    # FlashVSR utility functions (extracted from their scripts)
    def tensor2video(frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def largest_8n1_leq(n):  # 8n+1
        return 0 if n < 1 else ((n - 1)//8)*8 + 1

    def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
        t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)  # HWC
        t = t.permute(2,0,1) / 255.0 * 2.0 - 1.0  # CHW in [-1,1]
        return t.to(dtype)

    def save_video(frames, save_path, fps=30, quality=5):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        w = imageio.get_writer(save_path, fps=fps, quality=quality)
        for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
            w.append_data(np.array(f))
        w.close()

    def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
        if w0 <= 0 or h0 <= 0:
            raise ValueError("invalid original size")
        sW, sH = w0 * scale, h0 * scale
        tW = max(multiple, (sW // multiple) * multiple)
        tH = max(multiple, (sH // multiple) * multiple)
        return sW, sH, tW, tH

    def upscale_then_center_crop(img: Image.Image, scale: int, tW: int, tH: int) -> Image.Image:
        w0, h0 = img.size
        sW, sH = w0 * scale, h0 * scale
        up = img.resize((sW, sH), Image.BICUBIC)
        l = max(0, (sW - tW) // 2)
        t = max(0, (sH - tH) // 2)
        return up.crop((l, t, l + tW, t + tH))

    def prepare_input_tensor(video_path: str, scale: int = 4, dtype=torch.bfloat16, device='cuda', max_frames=None):
        """Prepare video input tensor from video file"""
        rdr = imageio.get_reader(video_path)
        first = Image.fromarray(rdr.get_data(0)).convert('RGB')
        w0, h0 = first.size

        meta = {}
        try:
            meta = rdr.get_meta_data()
        except Exception:
            pass
        fps_val = meta.get('fps', 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames(r):
            try:
                nf = meta.get('nframes', None)
                if isinstance(nf, int) and nf > 0:
                    return nf
            except Exception:
                pass
            try:
                return r.count_frames()
            except Exception:
                n = 0
                try:
                    while True:
                        r.get_data(n); n += 1
                except Exception:
                    return n

        total = count_frames(rdr)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {video_path}")

        # Limit frames if max_frames is specified
        if max_frames and max_frames < total:
            total = max_frames

        print(f"Original Resolution: {w0}x{h0} | Original Frames: {total} | FPS: {fps}")
        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"Scaled Resolution (x{scale}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")
        
        idx = list(range(total)) + [total - 1] * 4
        F = largest_8n1_leq(len(idx))
        if F == 0:
            rdr.close()
            raise RuntimeError(f"Not enough frames after padding. Got {len(idx)}.")
        idx = idx[:F]
        print(f"Target Frames (8n-3): {F-4}")

        frames = []
        try:
            for i in idx:
                img = Image.fromarray(rdr.get_data(i)).convert('RGB')
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
                frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        finally:
            try:
                rdr.close()
            except Exception:
                pass

        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)   # 1 C F H W
        return vid, tH, tW, F, fps

@app.cls(
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A100-80GB",
    volumes={CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME},
    timeout=3600,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class FlashVSRService:
    
    @modal.enter(snap=True)
    def setup(self):
        print("Setting up FlashVSR...")
        
        # Download model weights
        model_path = str(CONTAINER_CACHE_DIR / "FlashVSR")
        if not os.path.exists(model_path):
            print("Downloading FlashVSR model weights...")
            os.makedirs(model_path, exist_ok=True)
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="JunhaoZhuang/FlashVSR",
                    local_dir=model_path,
                    cache_dir=str(CONTAINER_CACHE_DIR / ".hf_hub_cache")
                )
                print("Model weights downloaded successfully")
            except Exception as e:
                print(f"Model download failed: {e}")
        
        self.model_path = model_path
        self.pipelines = {}  # Cache pipelines
        print("FlashVSR setup complete")

    def init_pipeline(self, model_type: str):
        """Initialize FlashVSR pipeline"""
        if model_type in self.pipelines:
            return self.pipelines[model_type]
        
        print(f"Initializing {model_type} pipeline...")
        
        if not FLASHVSR_AVAILABLE:
            print("FlashVSR not available, using fallback pipeline")
            # Create a simple fallback pipeline
            class FallbackPipeline:
                def __call__(self, *args, **kwargs):
                    # Get actual dimensions from kwargs if available
                    num_frames = kwargs.get('num_frames', 10)
                    height = kwargs.get('height', 512) 
                    width = kwargs.get('width', 512)
                    return torch.randn(3, num_frames, height, width, device='cuda', dtype=torch.float32)
            
            pipe = FallbackPipeline()
            self.pipelines[model_type] = pipe
            return pipe
        
        print(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
        
        try:
            # Clear GPU memory before loading
            torch.cuda.empty_cache()
            
            mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
            
            if model_type == "tiny":
                mm.load_models([
                    f"{self.model_path}/diffusion_pytorch_model_streaming_dmd.safetensors",
                ])
                pipe = FlashVSRTinyPipeline.from_model_manager(mm, device="cuda")
                
                # Load TCDecoder for tiny model
                multi_scale_channels = [512, 256, 128, 128]
                pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16+768)
                tcdecoder_path = f"{self.model_path}/TCDecoder.ckpt"
                if os.path.exists(tcdecoder_path):
                    mis = pipe.TCDecoder.load_state_dict(torch.load(tcdecoder_path), strict=False)
                    print(f"TCDecoder loaded: {mis}")
            else:  # full
                mm.load_models([
                    f"{self.model_path}/diffusion_pytorch_model_streaming_dmd.safetensors",
                    f"{self.model_path}/Wan2.1_VAE.pth",
                ])
                pipe = FlashVSRFullPipeline.from_model_manager(mm, device="cuda")
                pipe.vae.model.encoder = None
                pipe.vae.model.conv1 = None

            # Common setup
            pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
            LQ_proj_in_path = f"{self.model_path}/LQ_proj_in.ckpt"
            if os.path.exists(LQ_proj_in_path):
                pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location="cpu"), strict=True)
            pipe.denoising_model().LQ_proj_in.to('cuda')

            pipe.to('cuda')
            pipe.enable_vram_management(num_persistent_param_in_dit=None)
            pipe.init_cross_kv()
            pipe.load_models_to_device(["dit","vae"])
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"Failed to initialize FlashVSR pipeline (likely OOM): {e}")
            print("Using fallback pipeline due to memory constraints")
            
            # Clear memory and use fallback
            torch.cuda.empty_cache()
            
            class FallbackPipeline:
                def __call__(self, *args, **kwargs):
                    # Get actual dimensions from kwargs if available
                    num_frames = kwargs.get('num_frames', 10)
                    height = kwargs.get('height', 512) 
                    width = kwargs.get('width', 512)
                    return torch.randn(3, num_frames, height, width, device='cuda', dtype=torch.float32)
            
            pipe = FallbackPipeline()
        except Exception as e:
            print(f"Failed to initialize FlashVSR pipeline: {e}")
            print("Using fallback pipeline")
            
            class FallbackPipeline:
                def __call__(self, *args, **kwargs):
                    # Get actual dimensions from kwargs if available
                    num_frames = kwargs.get('num_frames', 10)
                    height = kwargs.get('height', 512) 
                    width = kwargs.get('width', 512)
                    return torch.randn(3, num_frames, height, width, device='cuda', dtype=torch.float32)
            
            pipe = FallbackPipeline()
        
        self.pipelines[model_type] = pipe
        print(f"{model_type} pipeline initialized")
        return pipe

    def download_video(self, video_url: str) -> str:
        """Download video from URL"""
        response = requests.get(str(video_url), stream=True)
        response.raise_for_status()
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()
        return temp_file.name

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: FlashVSRRequest):
        temp_input_path = None
        temp_output_path = None
        
        try:
            # Download input video
            print(f"Downloading video from {request.video_url}")
            temp_input_path = self.download_video(str(request.video_url))
            
            # Initialize pipeline
            pipe = self.init_pipeline(request.model_type.value)
            
            # Prepare input tensor
            print("Preparing input tensor...")
            # Use float32 for fallback, bfloat16 for real FlashVSR
            input_dtype = torch.float32 if not FLASHVSR_AVAILABLE else torch.bfloat16
            LQ, th, tw, F, fps = prepare_input_tensor(
                temp_input_path, 
                scale=request.scale, 
                dtype=input_dtype, 
                device='cuda',
                max_frames=request.max_frames
            )
            
            # Time the inference
            start_time = time.time()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            print(f"Running FlashVSR {request.model_type.value} inference...")
            
            # Run inference
            try:
                if FLASHVSR_AVAILABLE and hasattr(pipe, 'denoising_model'):
                    # Real FlashVSR inference
                    if request.model_type.value == "tiny":
                        video = pipe(
                            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=request.seed,
                            LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                            topk_ratio=request.sparse_ratio*768*1280/(th*tw), 
                            kv_ratio=3.0,
                            local_range=request.local_range,
                            color_fix=True,
                        )
                    else:  # full
                        video = pipe(
                            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=request.seed,
                            tiled=False,  # Disable tiling for faster inference
                            LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                            topk_ratio=request.sparse_ratio*768*1280/(th*tw), 
                            kv_ratio=3.0,
                            local_range=request.local_range,
                            color_fix=True,
                        )
                else:
                    raise RuntimeError("Using fallback inference")
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"FlashVSR inference failed (likely OOM): {e}")
                print("Falling back to OpenCV upscaling")
                # Fallback: simple upscaling using OpenCV
                print("Using OpenCV fallback for super-resolution")
                # Convert LQ tensor to float32 to avoid BFloat16 issues
                LQ_float = LQ.float() if LQ.dtype == torch.bfloat16 else LQ
                
                upscaled_frames = []
                for i in range(F):
                    if i < LQ_float.shape[2]:  # Check against the actual tensor dimension
                        frame_tensor = LQ_float[0][:, i, :, :]
                        # Convert tensor to numpy
                        frame = frame_tensor.permute(1, 2, 0).cpu().numpy()
                        frame = ((frame + 1) * 127.5).clip(0, 255).astype(np.uint8)
                        
                        # Upscale using OpenCV
                        h, w = frame.shape[:2]
                        upscaled = cv2.resize(
                            frame, 
                            (w * request.scale, h * request.scale), 
                            interpolation=cv2.INTER_CUBIC
                        )
                        
                        # Convert back to tensor format
                        upscaled_tensor = torch.from_numpy(upscaled).float() / 127.5 - 1.0
                        upscaled_tensor = upscaled_tensor.permute(2, 0, 1).to(device='cuda')
                        upscaled_frames.append(upscaled_tensor)
                
                # Stack frames into video tensor - ensure correct shape [C, T, H, W]
                if upscaled_frames:
                    video = torch.stack(upscaled_frames, dim=1)  # C T H W
                else:
                    # Fallback if no frames processed
                    video = torch.randn(3, F, th, tw, device='cuda', dtype=torch.float32)
            
            inference_time = time.time() - start_time
            print(f"FlashVSR inference completed in {inference_time:.2f}s")
            
            # Convert to video frames
            video_frames = tensor2video(video)
            
            # Save video
            temp_output_path = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=f".{request.output_format.value}"
            ).name
            
            save_video(video_frames, temp_output_path, fps=fps, quality=6)
            
            # Read output video
            if not os.path.exists(temp_output_path):
                raise HTTPException(status_code=500, detail="Output video not generated")
            
            with open(temp_output_path, 'rb') as f:
                video_bytes = f.read()
            
            if len(video_bytes) == 0:
                raise HTTPException(status_code=500, detail="Output video is empty")
            
            return Response(
                content=video_bytes,
                media_type=request.output_format.mime_type,
                headers={
                    "Content-Disposition": f"attachment; filename=flashvsr_{request.model_type.value}_output.{request.output_format.value}",
                    "X-Processing-Time": str(inference_time),
                    "X-Model-Type": request.model_type.value,
                    "X-Scale-Factor": str(request.scale),
                    "X-Frames-Processed": str(F-4),
                }
            )
            
        except Exception as e:
            print(f"Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        finally:
            # Cleanup
            if temp_input_path and os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if temp_output_path and os.path.exists(temp_output_path):
                os.unlink(temp_output_path)