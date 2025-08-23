# RunPod ComfyUI Video Upscaler

This directory contains the RunPod serverless deployment for the ComfyUI video upscaler, converted from the original Modal.com implementation.

## Overview

The video upscaler uses the Wan2.1 model through ComfyUI to upscale videos with enhanced quality. It's designed to run as a RunPod serverless worker that can process video files and return upscaled results.

## Files

- `Dockerfile` - Container definition for the RunPod worker
- `handler.py` - Main RunPod serverless handler
- `download_models.py` - Standalone script to download required models
- `requirements.txt` - Python dependencies
- `test_endpoint.py` - Test script for the deployed endpoint
- `VideoUpscalerAPI.json` - ComfyUI workflow definition (copied from parent directory)

## Quick Start

### 1. Build and Push Docker Image

```bash
# Build the Docker image
docker build --platform linux/amd64 -t your-username/comfyui-video-upscaler:latest .

# Push to Docker Hub (or your preferred registry)
docker push your-username/comfyui-video-upscaler:latest
```

### 2. Deploy to RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/serverless)
2. Create a new template:
   - **Container Image**: `your-username/comfyui-video-upscaler:latest`
   - **Container Disk**: 20GB minimum
   - **GPU**: L4 or better (A100 recommended for faster processing)
   - **Environment Variables**: 
     - `HF_HUB_ENABLE_HF_TRANSFER=1`
     - `RUNPOD_DEBUG_LEVEL=INFO`

3. Create an endpoint using the template
4. Note your endpoint ID and API key

### 3. Test the Endpoint

Update `test_endpoint.py` with your endpoint details and run:

```bash
python test_endpoint.py
```

## API Usage

### Input Format

The handler accepts two input methods:

#### Method 1: Video URL
```json
{
  "input": {
    "video_url": "https://example.com/video.mp4"
  }
}
```

#### Method 2: Base64 Encoded Video
```json
{
  "input": {
    "video_base64": "base64_encoded_video_data"
  }
}
```

### Output Format

```json
{
  "output": {
    "upscaled_video_base64": "base64_encoded_upscaled_video",
    "original_size_bytes": 1234567,
    "upscaled_size_bytes": 2345678,
    "job_id": "unique_job_id"
  }
}
```

### Error Format

```json
{
  "error": "Error description"
}
```

## cURL Examples

### Synchronous Request
```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "video_url": "https://example.com/video.mp4"
    }
  }'
```

### Asynchronous Request
```bash
# Submit job
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "video_url": "https://example.com/video.mp4"
    }
  }'

# Check status (replace JOB_ID with returned ID)
curl -X GET "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Python Client Example

```python
import runpod
import base64

# Initialize client
runpod.api_key = "YOUR_API_KEY"
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Method 1: Using video URL
result = endpoint.run_sync({
    "video_url": "https://example.com/video.mp4"
})

# Method 2: Using local video file
with open("input_video.mp4", "rb") as f:
    video_data = f.read()

video_base64 = base64.b64encode(video_data).decode('utf-8')

result = endpoint.run_sync({
    "video_base64": video_base64
})

# Save upscaled video
if "output" in result and "upscaled_video_base64" in result["output"]:
    upscaled_data = base64.b64decode(result["output"]["upscaled_video_base64"])
    
    with open("upscaled_video.mp4", "wb") as f:
        f.write(upscaled_data)
    
    print("Upscaled video saved!")
```

## Model Information

The upscaler uses the following models (automatically downloaded):

- **Wan2.1 T2V Model**: Main diffusion model for video processing
- **Wan2.1 VAE**: Variational autoencoder for encoding/decoding
- **UMT5 Text Encoder**: Text understanding for prompts
- **RealESRGAN Upscaler**: Image upscaling model
- **Wan2.1 LoRA**: Fine-tuning weights for enhanced quality

Total model size: ~15-20GB

## Performance Notes

- **Cold Start**: First request may take 2-3 minutes due to model loading
- **Warm Requests**: Subsequent requests typically complete in 30-60 seconds
- **GPU Requirements**: L4 minimum, A100 recommended for best performance
- **Memory**: 16GB+ VRAM recommended
- **Storage**: 25GB+ container disk for models and temporary files

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase GPU memory or use smaller input videos
2. **Timeout**: Increase timeout settings for longer videos
3. **Model Download Fails**: Check internet connectivity and HuggingFace access

### Debug Mode

Set environment variable for more verbose logging:
```bash
RUNPOD_DEBUG_LEVEL=DEBUG
```

### Manual Model Download

If models fail to download automatically:
```bash
python download_models.py
```

## Differences from Modal.com Version

- Uses RunPod serverless instead of Modal
- Models downloaded at runtime (can be pre-downloaded in Docker build)
- Base64 encoding for video input/output instead of direct file handling
- Simplified error handling and logging
- No memory snapshot functionality (RunPod handles cold starts differently)

## Cost Optimization

- Use spot instances when available
- Set appropriate idle timeout (5-10 minutes)
- Consider pre-downloading models in Docker image for faster cold starts
- Use appropriate GPU tier for your use case

## Support

For issues specific to this RunPod implementation, please check:
1. RunPod documentation: https://docs.runpod.io/
2. ComfyUI documentation: https://github.com/comfyanonymous/ComfyUI
3. Original Modal implementation in parent directory