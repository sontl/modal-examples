"""
Fixed MultiTalk implementation for Modal.com
Handles dependency issues and avoids TTS imports
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

import modal

flash_attn_release = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)

# Build a minimal Modal image with proper dependencies
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
        "espeak",
        "espeak-data"
    )
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1", 
        "torchaudio==2.4.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "xformers==0.0.28",
        extra_options="--index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "transformers==4.52",
        "diffusers",
        "accelerate",
        "safetensors",
        "opencv-python",
        "pillow",
        "numpy<2.0",  # Pin numpy version for compatibility
        "scipy",
        "soundfile",
        "moviepy",
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
        flash_attn_release,
        "tokenizers>=0.20.3",
        "tqdm",
        "imageio",
        "dashscope",
        "imageio-ffmpeg",
        "gradio>=5.0.0",
       " xfuser>=0.4.1"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(
        "mkdir -p /app",
        "cd /tmp && git clone https://github.com/MeiGen-AI/MultiTalk.git",
        "cp -r /tmp/MultiTalk/* /app/"
    )
)

vol = modal.Volume.from_name("multitalk-cache", create_if_missing=True)
app = modal.App(name="multitalk-fixed", image=image)

@app.cls(
    gpu="L40S",
    volumes={"/data": vol},
    timeout=3600,
    scaledown_window=60
)
class MultiTalkFixed:
    base_dir: str = modal.parameter(default="/data/weights/Wan2.1-I2V-14B-480P")
    wav2vec_dir: str = modal.parameter(default="/data/weights/chinese-wav2vec2-base")
        
    def _check_base_model_complete(self):
        """Check if base model is completely downloaded"""
        required_files = [
            f"{self.base_dir}/diffusion_pytorch_model.safetensors.index.json",
            f"{self.base_dir}/config.json"
        ]
        return all(os.path.exists(f) and os.path.getsize(f) > 0 for f in required_files)
    
    def _check_wav2vec_complete(self):
        """Check if wav2vec model is completely downloaded"""
        model_file = f"{self.wav2vec_dir}/model.safetensors"
        pytorch_file = f"{self.wav2vec_dir}/pytorch_model.bin"
        config_file = f"{self.wav2vec_dir}/config.json"
        
        # Either model.safetensors or pytorch_model.bin should exist, plus config
        has_model = (os.path.exists(model_file) and os.path.getsize(model_file) > 0) or \
                   (os.path.exists(pytorch_file) and os.path.getsize(pytorch_file) > 0)
        has_config = os.path.exists(config_file) and os.path.getsize(config_file) > 0
        
        return has_model and has_config
    
    def _check_multitalk_complete(self):
        """Check if MultiTalk weights are properly set up"""
        multitalk_file = f"{self.base_dir}/multitalk.safetensors"
        quant_t5_file = "/data/weights/MeiGen-MultiTalk/quant_models/t5_int8.safetensors"
        lora_file = "/data/weights/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
        
        # Check the main multitalk file and quantization files
        has_multitalk = os.path.exists(multitalk_file) and os.path.getsize(multitalk_file) > 0
        has_quant = os.path.exists(quant_t5_file) and os.path.getsize(quant_t5_file) > 0
        
        # LoRA is optional, but if we're using it, it should exist
        has_lora = os.path.exists(lora_file) and os.path.getsize(lora_file) > 0
        
        return has_multitalk and has_quant and has_lora

    @modal.enter()
    def setup(self):
        """Setup the environment and download models if needed"""
        import os
        import sys
        from pathlib import Path
        
        print("Setting up MultiTalk environment...")
        
        # Add app to Python path
        sys.path.insert(0, "/app")
        os.chdir("/app")
        
        # List available files in /app
        app_files = list(Path("/app").glob("*"))
        print(f"Files in /app: {[f.name for f in app_files]}")
        
        # Patch the problematic imports before they're loaded
        self._patch_imports()
        
        # Check if models are already downloaded and complete
        base_complete = self._check_base_model_complete()
        wav2vec_complete = self._check_wav2vec_complete()
        multitalk_complete = self._check_multitalk_complete()
        
        print(f"Base model complete: {base_complete} ({self.base_dir})")
        print(f"Wav2vec model complete: {wav2vec_complete} ({self.wav2vec_dir})")
        print(f"MultiTalk weights complete: {multitalk_complete}")
        
        if not base_complete or not wav2vec_complete or not multitalk_complete:
            print("Some models missing or incomplete, downloading...")
            self._download_models()
        else:
            print("All models already complete!")
            
        print("MultiTalk environment ready")
    
    def _patch_imports(self):
        """Patch problematic imports in MultiTalk code"""
        
        # Patch generate_multitalk.py to skip TTS imports
        generate_file = "/app/generate_multitalk.py"
        if os.path.exists(generate_file):
            with open(generate_file, 'r') as f:
                content = f.read()
            
            # Comment out the kokoro import and related TTS code
            patches = [
                ("from kokoro import KPipeline", "# from kokoro import KPipeline  # Disabled for Modal deployment"),
                ("import kokoro", "# import kokoro  # Disabled for Modal deployment"),
                ("tts_pipeline = KPipeline(", "# tts_pipeline = KPipeline(  # TTS disabled"),
                ("tts_pipeline(", "# tts_pipeline(  # TTS disabled"),
            ]
            
            for old, new in patches:
                content = content.replace(old, new)
            
            with open(generate_file, 'w') as f:
                f.write(content)
            
            print("Patched generate_multitalk.py imports")
        else:
            print(f"Warning: {generate_file} not found for patching")
        
        # Create a dummy kokoro module to prevent import errors
        os.makedirs("/app/kokoro", exist_ok=True)
        with open("/app/kokoro/__init__.py", 'w') as f:
            f.write("""
# Dummy kokoro module for Modal deployment
class KPipeline:
    def __init__(self, *args, **kwargs):
        print("Warning: TTS functionality disabled in Modal deployment")
        pass
    
    def __call__(self, *args, **kwargs):
        return None

# Export the class
__all__ = ['KPipeline']
""")
        
        print("Created dummy kokoro module")
        
        # Also patch any other files that might import kokoro
        for py_file in Path("/app").glob("*.py"):
            if py_file.name == "generate_multitalk.py":
                continue  # Already patched above
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                if "kokoro" in content or "KPipeline" in content:
                    # Apply similar patches
                    for old, new in patches:
                        content = content.replace(old, new)
                    
                    with open(py_file, 'w') as f:
                        f.write(content)
                    
                    print(f"Patched {py_file.name}")
            except Exception as e:
                print(f"Warning: Could not patch {py_file.name}: {e}")
    
    def _download_models(self):
        """Download models on first use"""
        from huggingface_hub import snapshot_download, hf_hub_download
        import os
        import shutil
        
        # Create directories
        os.makedirs("/data/weights", exist_ok=True)
        os.makedirs("/data/cache", exist_ok=True)
        
        try:
            # Only download models that don't exist or are incomplete
            base_model_complete = self._check_base_model_complete()
            wav2vec_complete = self._check_wav2vec_complete()
            multitalk_complete = self._check_multitalk_complete()
            
            if not base_model_complete:
                print("Downloading Wan2.1-I2V-14B-480P...")
                snapshot_download(
                    repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
                    local_dir=self.base_dir,
                    cache_dir="/data/cache"
                )
            else:
                print("Base model already complete, skipping download")
            
            if not wav2vec_complete:
                print("Downloading chinese-wav2vec2-base...")
                # Try multiple approaches to get the model.safetensors file
                try:
                    # First try the main branch
                    snapshot_download(
                        repo_id="TencentGameMate/chinese-wav2vec2-base", 
                        local_dir=self.wav2vec_dir,
                        cache_dir="/data/cache"
                    )
                    
                    # Check if model.safetensors exists, if not try PR branch
                    model_file = f"{self.wav2vec_dir}/model.safetensors"
                    if not os.path.exists(model_file):
                        print("model.safetensors not found in main branch, trying PR #1...")
                        try:
                            hf_hub_download(
                                repo_id="TencentGameMate/chinese-wav2vec2-base",
                                filename="model.safetensors",
                                local_dir=self.wav2vec_dir,
                                cache_dir="/data/cache",
                                revision="refs/pr/1"
                            )
                        except Exception as pr_error:
                            print(f"Failed to download from PR #1: {pr_error}")
                            # Try downloading pytorch_model.bin and converting if needed
                            try:
                                print("Trying to download pytorch_model.bin as fallback...")
                                hf_hub_download(
                                    repo_id="TencentGameMate/chinese-wav2vec2-base",
                                    filename="pytorch_model.bin",
                                    local_dir=self.wav2vec_dir,
                                    cache_dir="/data/cache"
                                )
                                print("Downloaded pytorch_model.bin - MultiTalk should handle conversion")
                            except Exception as bin_error:
                                print(f"Failed to download pytorch_model.bin: {bin_error}")
                                raise
                        
                except Exception as e:
                    print(f"Error downloading wav2vec model: {e}")
                    raise
            else:
                print("Wav2vec model already complete, skipping download")
            
            if not multitalk_complete:
                print("Downloading MeiGen-MultiTalk...")
                multitalk_dir = "/data/weights/MeiGen-MultiTalk"
                snapshot_download(
                    repo_id="MeiGen-AI/MeiGen-MultiTalk",
                    local_dir=multitalk_dir, 
                    cache_dir="/data/cache"
                )
                
                # Download quantization models specifically
                print("Downloading quantization models...")
                quant_dir = f"{multitalk_dir}/quant_models"
                os.makedirs(quant_dir, exist_ok=True)
                
                # Download specific quantization files
                quant_files = [
                    "t5_int8.safetensors",
                    "quantization_map_fp8_FusionX.json",
                    "dit_int8.safetensors",  # Add other potential quantization files
                    "vae_int8.safetensors"
                ]
                
                for quant_file in quant_files:
                    try:
                        print(f"Downloading {quant_file}...")
                        hf_hub_download(
                            repo_id="MeiGen-AI/MeiGen-MultiTalk",
                            filename=f"quant_models/{quant_file}",
                            local_dir=multitalk_dir,
                            cache_dir="/data/cache"
                        )
                        print(f"✓ Downloaded {quant_file}")
                    except Exception as e:
                        print(f"Warning: Could not download {quant_file}: {e}")
                        # For optional files, continue without failing
                        if quant_file == "t5_int8.safetensors":
                            # This is critical, re-raise the error
                            raise
                
                # Download LoRA weights from the correct repository
                try:
                    print("Downloading LoRA weights from vrgamedevgirl84/Wan14BT2VFusioniX...")
                    hf_hub_download(
                        repo_id="vrgamedevgirl84/Wan14BT2VFusioniX",
                        filename="FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors",
                        local_dir="/data/weights",
                        local_dir_use_symlinks=False,
                        cache_dir="/data/cache"
                    )
                    print("✓ Downloaded LoRA weights from vrgamedevgirl84 repo")
                except Exception as e:
                    print(f"Warning: Could not download LoRA weights from vrgamedevgirl84 repo: {e}")
                    # Try fallback from original repo
                    try:
                        print("Trying fallback download from MeiGen-AI repo...")
                        hf_hub_download(
                            repo_id="MeiGen-AI/MeiGen-MultiTalk",
                            filename="Wan2.1_I2V_14B_FusionX_LoRA.safetensors",
                            local_dir="/data/weights",
                            cache_dir="/data/cache"
                        )
                        print("✓ Downloaded LoRA weights from MeiGen-AI repo")
                    except Exception as fallback_error:
                        print(f"Warning: Could not download LoRA weights from either repo: {fallback_error}")
                
                # Setup MultiTalk files in base model directory
                self._setup_multitalk_files(multitalk_dir)
            else:
                print("MultiTalk weights already complete, skipping download")
            
            # Final verification
            self._verify_all_models()
            print("Model download and setup complete!")
            
        except Exception as e:
            print(f"Error downloading models: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _setup_multitalk_files(self, multitalk_dir):
        """Setup MultiTalk files in the base model directory"""
        import shutil
        
        index_file = f"{self.base_dir}/diffusion_pytorch_model.safetensors.index.json"
        multitalk_file = f"{self.base_dir}/multitalk.safetensors"
        
        # Backup original index file if it exists
        if os.path.exists(index_file):
            backup_file = f"{index_file}_original"
            if not os.path.exists(backup_file):
                shutil.copy2(index_file, backup_file)
                print(f"Backed up original index file to {backup_file}")
        
        # Copy MultiTalk files
        multitalk_index = f"{multitalk_dir}/diffusion_pytorch_model.safetensors.index.json"
        multitalk_weights = f"{multitalk_dir}/multitalk.safetensors"
        
        if os.path.exists(multitalk_index):
            shutil.copy2(multitalk_index, index_file)
            print(f"Copied MultiTalk index file to {index_file}")
        else:
            print(f"Warning: MultiTalk index file not found at {multitalk_index}")
        
        if os.path.exists(multitalk_weights):
            shutil.copy2(multitalk_weights, multitalk_file)
            print(f"Copied MultiTalk weights to {multitalk_file}")
        else:
            print(f"Warning: MultiTalk weights not found at {multitalk_weights}")
    
    def _verify_all_models(self):
        """Verify all required model files exist"""
        required_files = [
            f"{self.base_dir}/diffusion_pytorch_model.safetensors.index.json",
            f"{self.base_dir}/multitalk.safetensors",
            "/data/weights/MeiGen-MultiTalk/quant_models/t5_int8.safetensors",
            "/data/weights/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
        ]
        
        # For wav2vec, check for either model.safetensors or pytorch_model.bin
        wav2vec_model = f"{self.wav2vec_dir}/model.safetensors"
        wav2vec_pytorch = f"{self.wav2vec_dir}/pytorch_model.bin"
        
        if os.path.exists(wav2vec_model):
            required_files.append(wav2vec_model)
        elif os.path.exists(wav2vec_pytorch):
            required_files.append(wav2vec_pytorch)
        else:
            print(f"✗ Missing: Neither {wav2vec_model} nor {wav2vec_pytorch} found")
            return False
        
        all_good = True
        for file_path in required_files:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                print(f"✓ {file_path} ({os.path.getsize(file_path)} bytes)")
            else:
                print(f"✗ Missing or empty: {file_path}")
                all_good = False
        
        return all_good
    
    def _validate_input_json(self, input_data: dict):
        """Validate that the input JSON has the correct structure for MultiTalk"""
        
        # Required fields
        required_fields = ["prompt", "cond_image", "cond_audio"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate cond_audio structure
        cond_audio = input_data["cond_audio"]
        if not isinstance(cond_audio, dict):
            raise ValueError("cond_audio must be a dictionary")
        
        if "person1" not in cond_audio:
            raise ValueError("cond_audio must contain 'person1' key")
        
        # Validate files exist
        if not os.path.exists(input_data["cond_image"]):
            raise ValueError(f"Image file does not exist: {input_data['cond_image']}")
        
        if not os.path.exists(cond_audio["person1"]):
            raise ValueError(f"Audio file for person1 does not exist: {cond_audio['person1']}")
        
        # If multi-person, validate person2
        if "person2" in cond_audio:
            if not os.path.exists(cond_audio["person2"]):
                raise ValueError(f"Audio file for person2 does not exist: {cond_audio['person2']}")
            
            # audio_type should be specified for multi-person
            if "audio_type" not in input_data:
                print("Warning: audio_type not specified for multi-person audio, defaulting to 'add'")
                input_data["audio_type"] = "add"
            elif input_data["audio_type"] not in ["add", "para"]:
                raise ValueError(f"Invalid audio_type for multi-person: {input_data['audio_type']}")
        
        print("✓ Input JSON structure validation passed")
    
    @modal.method()
    def generate_video(
        self,
        audio_data: bytes,
        image_data: bytes,
        prompt: str = "A person talking naturally",
        sample_steps: int = 40,
        use_teacache: bool = True,
        low_vram: bool = False,
        use_lora: bool = False,
        use_quantization: bool = False,
        audio_data_person2: Optional[bytes] = None,
        audio_type: str = "single",  # "single", "add", or "para"
        bbox_person1: Optional[list] = None,
        bbox_person2: Optional[list] = None
    ) -> Dict[str, str]:
        """Generate video from audio and image data"""
        
        try:
            # Save input files
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file:
                audio_file.write(audio_data)
                audio_path = os.path.abspath(audio_file.name)
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as image_file:
                image_file.write(image_data)
                image_path = os.path.abspath(image_file.name)

            print(f"Audio file: {audio_path} ({os.path.getsize(audio_path)} bytes)")
            print(f"Image file: {image_path} ({os.path.getsize(image_path)} bytes)")
            
            # Handle second person audio if provided
            audio_path_person2 = None
            if audio_data_person2:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file2:
                    audio_file2.write(audio_data_person2)
                    audio_path_person2 = os.path.abspath(audio_file2.name)
                print(f"Audio file person2: {audio_path_person2} ({os.path.getsize(audio_path_person2)} bytes)")
            
            # Create input JSON for MultiTalk based on the expected structure
            input_data = {
                "prompt": prompt,
                "cond_image": image_path
            }
            
            # Handle different audio configurations
            if audio_data_person2 and audio_type in ["add", "para"]:
                # Multi-person audio
                input_data["audio_type"] = audio_type
                input_data["cond_audio"] = {
                    "person1": audio_path,
                    "person2": audio_path_person2
                }
                
                # Add bounding boxes if provided
                if bbox_person1 and bbox_person2:
                    input_data["bbox"] = {
                        "person1": bbox_person1,
                        "person2": bbox_person2
                    }
            else:
                # Single person audio (default format)
                input_data["cond_audio"] = {
                    "person1": audio_path
                }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as json_file:
                json.dump(input_data, json_file, indent=2)
                json_path = json_file.name
            
            # Validate JSON structure
            self._validate_input_json(input_data)
            
            # Debug: Print the JSON content
            print(f"Generated JSON structure:")
            print(json.dumps(input_data, indent=2))
            print(f"JSON file path: {json_path}")
            print(f"Audio type: {audio_type}")
            print(f"Has second person audio: {audio_data_person2 is not None}")
            
            # Create output filename
            output_filename = f"output_{abs(hash(str(input_data)))}.mp4"
            output_path = f"/app/{output_filename}"
            
            # Verify required files exist before running
            if not os.path.exists("/app/generate_multitalk.py"):
                return {
                    "success": False,
                    "error": "generate_multitalk.py not found in /app",
                    "available_files": [str(f) for f in Path("/app").glob("*")]
                }
            
            # Use our verification methods
            if not self._verify_all_models():
                return {
                    "success": False,
                    "error": "Required model files are missing or incomplete",
                    "suggestion": "Models may need to be re-downloaded"
                }
            
            # Build the proper MultiTalk command
            save_name = output_filename.replace('.mp4', '')  # Remove extension as script adds it
            cmd = [
                "python", "generate_multitalk.py",
                "--ckpt_dir", self.base_dir,
                "--wav2vec_dir", self.wav2vec_dir,
                "--input_json", json_path,
                "--sample_steps", str(sample_steps),
                "--mode", "streaming",
                "--save_file", save_name,
                "--size", "multitalk-480",
            ]
            
            print(f"Command to execute: {' '.join(cmd)}")
            print(f"Sample steps parameter: {sample_steps}")
            
            # Add optional flags
            if use_teacache:
                cmd.append("--use_teacache")
            
            if low_vram:
                cmd.extend(["--num_persistent_param_in_dit", "0"])

            if use_lora:
                cmd.extend(["--lora_dir", "/data/weights/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"])
                cmd.extend(["--lora_scale", "1"])
                cmd.extend(["--sample_text_guide_scale", "1.0"])
                cmd.extend(["--sample_audio_guide_scale", "2.0"])
                cmd.extend(["--sample_shift", "2"])

            if use_quantization:
                cmd.extend(["--quant", "int8"])
                cmd.extend(["--quant_dir", "/data/weights/MeiGen-MultiTalk"])

            print(f"Running MultiTalk generation...")
            print(f"Command: {' '.join(cmd)}")
            print(f"Working directory: /app")
            print(f"Input JSON: {json_path}")
            
            # Set up environment variables
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": "/app",
                "CUDA_VISIBLE_DEVICES": "0",  # Ensure GPU is visible
                "HF_HUB_ENABLE_HF_TRANSFER": "1"
            })
            
            # Run generation
            result = subprocess.run(
                cmd,
                cwd="/app",
                capture_output=True,
                text=True,
                env=env,
                timeout=1800  # 30 minutes
            )
            
            print(f"Command output: {result.stdout}")
            if result.stderr:
                print(f"Command errors: {result.stderr}")
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Generation failed with return code {result.returncode}: {result.stderr}",
                    "stdout": result.stdout
                }
            
            # Check if output video was created
            if not os.path.exists(output_path):
                # Try to find any generated video files
                possible_outputs = list(Path("/app").glob("*.mp4"))
                if possible_outputs:
                    output_path = str(possible_outputs[-1])  # Use the most recent one
                    print(f"Using output file: {output_path}")
                else:
                    return {
                        "success": False,
                        "error": f"No output video file found at {output_path}",
                        "stdout": result.stdout,
                        "available_files": [str(f) for f in Path("/app").glob("*")]
                    }
            
            # Check if file has content
            if os.path.getsize(output_path) == 0:
                return {
                    "success": False,
                    "error": f"Output video file is empty (0 bytes)",
                    "stdout": result.stdout
                }
            
            # Read video file and return as base64
            with open(output_path, "rb") as f:
                import base64
                video_data = base64.b64encode(f.read()).decode()
            
            print(f"Output video size: {os.path.getsize(output_path)} bytes")
            print(f"Base64 data size: {len(video_data)} characters")
            
            # Cleanup
            os.unlink(audio_path)
            if audio_path_person2:
                os.unlink(audio_path_person2)
            os.unlink(image_path)
            os.unlink(json_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
            
            return {
                "success": True,
                "video_data": video_data,
                "filename": os.path.basename(output_path)
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Generation timed out"}
        except Exception as e:
            import traceback
            return {
                "success": False, 
                "error": f"Unexpected error: {str(e)}",
                "traceback": traceback.format_exc()
            }

# FastAPI web interface
@app.function(
    image=image,
    volumes={"/data": vol}
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI interface for MultiTalk"""
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.responses import Response
    import base64
    
    web_app = FastAPI(title="MultiTalk Fixed API", version="1.0.0")
    multitalk = MultiTalkFixed()
    
    @web_app.get("/")
    async def root():
        return {
            "message": "MultiTalk Fixed API",
            "endpoints": {
                "generate": "/generate - POST with audio and image files",
                "health": "/health - GET health check"
            }
        }
    
    @web_app.get("/health")
    async def health():
        return {"status": "healthy", "service": "multitalk-fixed"}
    
    @web_app.post("/generate")
    async def generate_video(
        audio: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)"),
        image: UploadFile = File(..., description="Reference image (JPG, PNG, etc.)"),
        prompt: str = Form(default="A person talking naturally", description="Generation prompt"),
        sample_steps: int = Form(default=40, description="Number of sampling steps (10-50)"),
        use_teacache: bool = Form(default=True, description="Use TeaCache optimization"),
        low_vram: bool = Form(default=False, description="Enable low VRAM mode"),
        use_lora: bool = Form(default=False, description="Enable LoRA acceleration"),
        use_quantization: bool = Form(default=False, description="Enable INT8 quantization"),
        audio_person2: Optional[UploadFile] = File(None, description="Second person audio (optional)"),
        audio_type: str = Form(default="single", description="Audio type: single, add, or para"),
        bbox_person1: Optional[str] = Form(None, description="Bounding box for person1 as JSON array [x,y,w,h]"),
        bbox_person2: Optional[str] = Form(None, description="Bounding box for person2 as JSON array [x,y,w,h]")
    ):
        """Generate talking video from audio and reference image"""
        
        try:
            # Validate inputs
            if not audio.filename or not image.filename:
                raise HTTPException(status_code=400, detail="Both audio and image files are required")
            
            if sample_steps < 4 or sample_steps > 50:
                raise HTTPException(status_code=400, detail="Sample steps must be between 4 and 50")
            
            # Read file data
            audio_data = await audio.read()
            image_data = await image.read()
            
            # Handle second person audio
            audio_data_person2 = None
            if audio_person2:
                audio_data_person2 = await audio_person2.read()
            
            # Parse bounding boxes if provided
            bbox_person1_parsed = None
            bbox_person2_parsed = None
            if bbox_person1:
                try:
                    bbox_person1_parsed = json.loads(bbox_person1)
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid bbox_person1 JSON format")
            
            if bbox_person2:
                try:
                    bbox_person2_parsed = json.loads(bbox_person2)
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid bbox_person2 JSON format")
            
            # Validate file sizes (optional)
            if len(audio_data) > 50 * 1024 * 1024:  # 50MB limit
                raise HTTPException(status_code=400, detail="Audio file too large (max 50MB)")
            
            if audio_data_person2 and len(audio_data_person2) > 50 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Second audio file too large (max 50MB)")
            
            if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
            
            # Validate audio_type
            if audio_type not in ["single", "add", "para"]:
                raise HTTPException(status_code=400, detail="audio_type must be 'single', 'add', or 'para'")
            
            # Generate video
            result = multitalk.generate_video.remote(
                audio_data=audio_data,
                image_data=image_data,
                prompt=prompt,
                sample_steps=sample_steps,
                use_teacache=use_teacache,
                low_vram=low_vram,
                use_lora=use_lora,
                use_quantization=use_quantization,
                audio_data_person2=audio_data_person2,
                audio_type=audio_type,
                bbox_person1=bbox_person1_parsed,
                bbox_person2=bbox_person2_parsed
            )
            
            if result.get("success"):
                # Return video as binary response
                video_data = base64.b64decode(result["video_data"])
                
                return Response(
                    content=video_data,
                    media_type="video/mp4",
                    headers={
                        "Content-Disposition": f"attachment; filename={result.get('filename', 'generated_video.mp4')}"
                    }
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": result.get("error", "Unknown error"),
                        "stdout": result.get("stdout", "")
                    }
                )
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    @web_app.post("/generate-json")
    async def generate_video_json(
        audio: UploadFile = File(...),
        image: UploadFile = File(...),
        prompt: str = Form(default="A person talking naturally"),
        sample_steps: int = Form(default=40),
        use_teacache: bool = Form(default=True),
        low_vram: bool = Form(default=False),
        use_lora: bool = Form(default=False),
        use_quantization: bool = Form(default=False),
        audio_person2: Optional[UploadFile] = File(None),
        audio_type: str = Form(default="single"),
        bbox_person1: Optional[str] = Form(None),
        bbox_person2: Optional[str] = Form(None)
    ):
        """Generate video and return as base64 JSON response"""
        
        try:
            # Validate inputs
            if not audio.filename or not image.filename:
                raise HTTPException(status_code=400, detail="Both audio and image files are required")
            
            # Read file data
            audio_data = await audio.read()
            image_data = await image.read()
            
            # Handle second person audio
            audio_data_person2 = None
            if audio_person2:
                audio_data_person2 = await audio_person2.read()
            
            # Parse bounding boxes if provided
            bbox_person1_parsed = None
            bbox_person2_parsed = None
            if bbox_person1:
                try:
                    bbox_person1_parsed = json.loads(bbox_person1)
                except json.JSONDecodeError:
                    return {"success": False, "error": "Invalid bbox_person1 JSON format"}
            
            if bbox_person2:
                try:
                    bbox_person2_parsed = json.loads(bbox_person2)
                except json.JSONDecodeError:
                    return {"success": False, "error": "Invalid bbox_person2 JSON format"}
            
            # Generate video
            result = multitalk.generate_video.remote(
                audio_data=audio_data,
                image_data=image_data,
                prompt=prompt,
                sample_steps=sample_steps,
                use_teacache=use_teacache,
                low_vram=low_vram,
                use_lora=use_lora,
                use_quantization=use_quantization,
                audio_data_person2=audio_data_person2,
                audio_type=audio_type,
                bbox_person1=bbox_person1_parsed,
                bbox_person2=bbox_person2_parsed
            )
            
            if result.get("success"):
                return {
                    "success": True,
                    "video_data": result["video_data"],
                    "filename": result.get("filename", "generated_video.mp4"),
                    "prompt": prompt,
                    "sample_steps": sample_steps
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "stdout": result.get("stdout", "")
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Internal server error: {str(e)}"
            }
    
    return web_app

# CLI interface
@app.local_entrypoint()
def main(
    audio_path: str,
    image_path: str,
    prompt: str = "A person talking naturally",
    sample_steps: int = 40,
    output_path: str = "output.mp4",
    audio_path_person2: Optional[str] = None,
    audio_type: str = "single"
):
    """CLI interface for MultiTalk generation"""
    
    # Read input files
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    # Read second audio file if provided
    audio_data_person2 = None
    if audio_path_person2:
        with open(audio_path_person2, 'rb') as f:
            audio_data_person2 = f.read()
    
    multitalk = MultiTalkFixed()
    
    result = multitalk.generate_video.remote(
        audio_data=audio_data,
        image_data=image_data,
        prompt=prompt,
        sample_steps=sample_steps,
        audio_data_person2=audio_data_person2,
        audio_type=audio_type
    )
    
    if result.get("success"):
        import base64
        video_data = base64.b64decode(result["video_data"])
        with open(output_path, "wb") as f:
            f.write(video_data)
        print(f"✅ Video saved to {output_path}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
        if result.get('stdout'):
            print(f"Output: {result['stdout']}")