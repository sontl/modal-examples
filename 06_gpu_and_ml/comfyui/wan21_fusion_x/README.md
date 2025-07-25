# Video Upscaler with Modal and ComfyUI

This project provides a video upscaling service using ComfyUI's WAN2.1 model deployed on Modal.com.

## Features

- Video upscaling using RealESRGAN and WAN2.1 models
- FastAPI endpoint for easy video upload
- Memory snapshots for faster cold starts
- Automatic model downloading from Hugging Face

## Setup

1. Install Modal CLI:
```bash
pip install modal
modal setup
```

2. Deploy the application:
```bash
modal deploy upscaler-wan.py
```

## Usage

### Method 1: FastAPI Web Interface

After deployment, Modal will provide you with URLs like:
- Main FastAPI app: `https://your-workspace--upscaler-wan-fastapi-app.modal.run`
- Direct webhook endpoint: `https://your-workspace--upload-and-upscale.modal.run`
- ComfyUI interface: `https://your-workspace--upscaler-wan-ui.modal.run`

**Option 1: Use the FastAPI endpoint (recommended):**

```bash
curl -X POST \
  -F "video=@your_video.mp4" \
  https://your-workspace--upscaler-wan-fastapi-app.modal.run/upscale \
  -o upscaled_video.mp4
```

**Option 2: Use the direct webhook endpoint:**

```bash
curl -X POST \
  --data-binary @your_video.mp4 \
  -H "Content-Type: application/octet-stream" \
  https://your-workspace--upload-and-upscale.modal.run \
  -o upscaled_video.mp4
```

### Method 2: Python Script

Use the provided test script for the FastAPI endpoint:

```bash
python test_upscaler.py input_video.mp4 https://your-workspace--upscaler-wan-fastapi-app.modal.run
```

Or test both endpoints:

```bash
python test_endpoints.py input_video.mp4 \
  https://your-workspace--upscaler-wan-fastapi-app.modal.run \
  https://your-workspace--upload-and-upscale.modal.run
```

### Method 3: Direct API Call

You can also call the upscaler methods directly:

```python
import modal

# Get the deployed class
upscaler_cls = modal.Cls.from_name("upscaler-wan", "UpscalerWan")

# Read your video file
with open("input_video.mp4", "rb") as f:
    video_bytes = f.read()

# Process the video
result = upscaler_cls().process_video_bytes.remote(video_bytes)

# Save the result
with open("upscaled_video.mp4", "wb") as f:
    f.write(result)
```

## API Endpoints

### POST /upscale
Upload a video file for upscaling.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: video file

**Response:**
- Content-Type: video/mp4
- Body: Upscaled video file

**Example:**
```python
import requests

with open("input.mp4", "rb") as f:
    files = {"video": ("input.mp4", f, "video/mp4")}
    response = requests.post("https://your-endpoint/upscale", files=files)
    
with open("output.mp4", "wb") as f:
    f.write(response.content)
```

## Configuration

The workflow uses the following models:
- **Upscale Model**: RealESRGAN_x2plus.pth
- **Diffusion Model**: Wan2_1-T2V-1_3B_bf16.safetensors
- **VAE**: wan_2.1_vae.safetensors
- **Text Encoder**: umt5-xxl-encoder-Q6_K.gguf
- **LoRA**: Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors

## Workflow Parameters

The ComfyUI workflow includes:
- **Upscaling**: 2x upscaling with RealESRGAN
- **Resize**: 1.5x additional scaling
- **Video Processing**: 30 FPS output, H.264 encoding
- **Quality**: CRF 10 for high quality output

## Troubleshooting

1. **Memory Issues**: The app uses L4 GPU by default. For larger videos, you might need to upgrade to A100.

2. **Timeout Issues**: Video processing can take time. The default timeout is 1200 seconds (20 minutes).

3. **Model Loading**: First run will be slower as models are downloaded. Subsequent runs use memory snapshots for faster startup.

4. **File Size Limits**: Modal has request size limits. For very large videos, consider using Modal Volumes for file transfer.

## Development

To modify the workflow:
1. Edit `VideoUpscalerAPI.json` with your ComfyUI workflow
2. Update model downloads in `hf_download()` function if needed
3. Redeploy with `modal deploy upscaler-wan.py`

## Cost Optimization

- Uses memory snapshots to reduce cold start times
- Automatically scales down when not in use
- Consider using spot instances for batch processing