# WanFusionX: Image-to-Video and Video Upscaling with Modal and ComfyUI

This project provides both image-to-video generation and video upscaling services using ComfyUI's WAN2.1 models deployed on Modal.com.

## Features

- **Image-to-Video Generation**: Convert static images to animated videos using WanFusionX
- **Video Upscaling**: Enhance video quality using RealESRGAN and WAN2.1 models
- FastAPI endpoints for easy file upload
- Memory snapshots for faster cold starts
- Automatic model downloading from Hugging Face
- Customizable prompts for image-to-video generation

## Setup

1. Install Modal CLI:
```bash
pip install modal
modal setup
```

2. Deploy the applications:

**For Image-to-Video Generation:**
```bash
modal deploy wan.py
```

**For Video Upscaling:**
```bash
modal deploy upscaler-wan.py
```

## Usage

## Image-to-Video Generation

### Method 1: FastAPI Endpoint

After deploying `wan.py`, Modal will provide you with URLs like:
- FastAPI app: `https://your-workspace--wan21-fusion-x-fastapi-app.modal.run`
- Image-to-Video endpoint: `https://your-workspace--wan21-fusion-x-fastapi-app.modal.run/image-to-video`
- API docs: `https://your-workspace--wan21-fusion-x-fastapi-app.modal.run/docs`
- ComfyUI interface: `https://your-workspace--wan-fusion-x-ui.modal.run`

**Upload an image and generate video:**

```bash
curl -X POST \
  -F "image=@your_image.jpg" \
  -F "prompt=A beautiful sunset with moving clouds and gentle waves" \
  https://your-workspace--wan21-fusion-x-fastapi-app.modal.run/image-to-video \
  -o generated_video.mp4
```

### Method 2: Python Script

Use the provided test script:

```bash
python test_image_to_video.py
```

### Method 3: Web Interface

Open `image_to_video_interface.html` in your browser for a user-friendly interface to upload images and generate videos.

### Method 4: Direct API Call

```python
import modal

# Get the deployed class
wan_cls = modal.Cls.from_name("wan21-fusion-x", "WanFusionX")

# Read your image file
with open("input_image.jpg", "rb") as f:
    image_bytes = f.read()

# Generate video
result = wan_cls().process_image_bytes.remote(
    image_bytes, 
    prompt="Your custom prompt here"
)

# Save the result
with open("generated_video.mp4", "wb") as f:
    f.write(result)
```

## Video Upscaling

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

### POST /image-to-video
Convert an image to an animated video.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: 
  - `image` (required): Image file (JPG, PNG, etc.)
  - `prompt` (optional): Text description of desired animation

**Response:**
- Content-Type: video/mp4
- Body: Generated video file

**Example:**
```python
import requests

with open("input.jpg", "rb") as f:
    files = {"image": ("input.jpg", f, "image/jpeg")}
    data = {"prompt": "The person's eyes glow with magical energy"}
    response = requests.post("https://your-endpoint/image-to-video", files=files, data=data)
    
with open("output.mp4", "wb") as f:
    f.write(response.content)
```

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