"""
FastAPI wrapper for MultiTalk Native
Provides REST API endpoints for video generation
"""

import base64
import tempfile
from typing import Dict, List, Optional

import modal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# Import the MultiTalk class
from multitalk_native import MultiTalkNative, app as multitalk_app

# Create FastAPI app
web_app = FastAPI(
    title="MultiTalk Native API",
    description="Audio-driven multi-person conversational video generation",
    version="1.0.0"
)

@web_app.post("/generate/single")
async def generate_single_person_video(
    audio: UploadFile = File(..., description="Audio file (WAV/MP3)"),
    image: UploadFile = File(..., description="Reference image (JPG/PNG)"),
    prompt: str = Form("A person talking naturally", description="Generation prompt"),
    use_lora: Optional[str] = Form(None, description="LoRA type: fusionx, lightx2v, or none"),
    use_quantization: bool = Form(False, description="Enable INT8 quantization"),
    low_vram: bool = Form(False, description="Enable low VRAM mode"),
    sample_steps: int = Form(40, description="Number of sampling steps"),
    use_teacache: bool = Form(True, description="Enable TeaCache acceleration"),
    use_apg: bool = Form(False, description="Enable APG for color error reduction")
):
    """Generate single-person video from audio and reference image"""
    
    try:
        # Save uploaded files to temporary locations
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio.filename.split('.')[-1]}") as audio_file:
            audio_content = await audio.read()
            audio_file.write(audio_content)
            audio_path = audio_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image.filename.split('.')[-1]}") as image_file:
            image_content = await image.read()
            image_file.write(image_content)
            image_path = image_file.name
        
        # Get MultiTalk instance
        multitalk = MultiTalkNative()
        
        # Adjust sample steps for LoRA
        if use_lora == "fusionx" and sample_steps > 8:
            sample_steps = 8
        elif use_lora == "lightx2v" and sample_steps > 4:
            sample_steps = 4
        
        # Generate video
        result = multitalk.generate_single_person.remote(
            audio_path=audio_path,
            reference_image_path=image_path,
            prompt=prompt,
            use_lora=use_lora if use_lora and use_lora != "none" else None,
            use_quantization=use_quantization,
            low_vram=low_vram,
            sample_steps=sample_steps,
            use_teacache=use_teacache,
            use_apg=use_apg
        )
        
        if result.get("success"):
            return JSONResponse({
                "success": True,
                "video_data": result["video_data"],
                "filename": result["filename"],
                "message": "Video generated successfully"
            })
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/generate/multi")
async def generate_multi_person_video(
    audio_files: List[UploadFile] = File(..., description="Audio files for each person"),
    image_files: List[UploadFile] = File(..., description="Reference images for each person"),
    prompts: List[str] = Form(..., description="Prompts for each person"),
    use_lora: Optional[str] = Form(None, description="LoRA type: fusionx, lightx2v, or none"),
    use_quantization: bool = Form(False, description="Enable INT8 quantization"),
    low_vram: bool = Form(False, description="Enable low VRAM mode"),
    sample_steps: int = Form(40, description="Number of sampling steps"),
    use_teacache: bool = Form(True, description="Enable TeaCache acceleration"),
    use_apg: bool = Form(False, description="Enable APG for color error reduction")
):
    """Generate multi-person video from multiple audio files and reference images"""
    
    if len(audio_files) != len(image_files) or len(audio_files) != len(prompts):
        raise HTTPException(
            status_code=400,
            detail="Number of audio files, images, and prompts must match"
        )
    
    try:
        # Save uploaded files
        audio_paths = []
        image_paths = []
        
        for i, (audio, image) in enumerate(zip(audio_files, image_files)):
            # Save audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio.filename.split('.')[-1]}") as audio_file:
                audio_content = await audio.read()
                audio_file.write(audio_content)
                audio_paths.append(audio_file.name)
            
            # Save image
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image.filename.split('.')[-1]}") as image_file:
                image_content = await image.read()
                image_file.write(image_content)
                image_paths.append(image_file.name)
        
        # Get MultiTalk instance
        multitalk = MultiTalkNative()
        
        # Adjust sample steps for LoRA
        if use_lora == "fusionx" and sample_steps > 8:
            sample_steps = 8
        elif use_lora == "lightx2v" and sample_steps > 4:
            sample_steps = 4
        
        # Generate video
        result = multitalk.generate_multi_person.remote(
            audio_paths=audio_paths,
            reference_images=image_paths,
            prompts=prompts,
            use_lora=use_lora if use_lora and use_lora != "none" else None,
            use_quantization=use_quantization,
            low_vram=low_vram,
            sample_steps=sample_steps,
            use_teacache=use_teacache,
            use_apg=use_apg
        )
        
        if result.get("success"):
            return JSONResponse({
                "success": True,
                "video_data": result["video_data"],
                "filename": result["filename"],
                "message": "Multi-person video generated successfully"
            })
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "MultiTalk Native API"}

@web_app.get("/models/info")
async def get_model_info():
    """Get information about available models and configurations"""
    return {
        "base_model": "Wan2.1-I2V-14B-480P",
        "audio_encoder": "chinese-wav2vec2-base",
        "lora_options": ["fusionx", "lightx2v"],
        "resolutions": ["multitalk-480", "multitalk-720"],
        "features": {
            "single_person": True,
            "multi_person": True,
            "lora_acceleration": True,
            "quantization": True,
            "low_vram": True,
            "teacache": True,
            "apg": True
        }
    }

# Mount the FastAPI app to Modal
@multitalk_app.function(
    image=multitalk_app.image.pip_install("fastapi", "python-multipart"),
    gpu="L40S"
)
@modal.web_server(8080, startup_timeout=300)
def fastapi_app():
    return web_app

if __name__ == "__main__":
    # For local development
    import uvicorn
    uvicorn.run(web_app, host="0.0.0.0", port=8080)