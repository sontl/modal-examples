# Video2X Modal API

This example provides a Modal API service for upscaling videos using [Video2X](https://github.com/k4yt3x/video2x). It wraps the Video2X CLI into a scalable, GPU-accelerated web endpoint.

## Features

- **Automated Build**: Compiles Video2X and its dependencies (Vulkan, Rust, Just) within a Modal container.
- **GPU Acceleration**: Utilizes NVIDIA A10G GPUs for high-performance upscaling.
- **Persistent Caching**: Uses a Modal Volume (`video2x_cache`) to store downloaded models and avoid redundant downloads.
- **Flexible Drivers**: Supports multiple Video2X backends (e.g., `realesrgan`, `waifu2x`).

## Deployment

To deploy the service as a persistent web endpoint:

```bash
modal deploy video2x_endpoint.py
```

*Note: The first deployment takes ~10-15 minutes because it builds Video2X from source. Subsequent deployments are nearly instant.*

## Usage

### 1. Running locally via Modal CLI
You can trigger the upscaler directly from your terminal using the built-in `local_entrypoint`:

```bash
modal run video2x_endpoint.py
```

### 2. Using the REST API
Once deployed, you can send POST requests to the web endpoint.

**Endpoint URL:** `https://son-tranlam-1986--video2x-endpoint-video2xservice-upscale.modal.run`

**Example Request (curl):**
```bash
curl -X POST https://son-tranlam-1986--video2x-endpoint-video2xservice-upscale.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",
    "scale": "2x",
    "driver": "realesrgan"
  }' \
  --output upscaled_video.mp4
```

### API Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `video_url` | String | (Required) | Publicly accessible URL of the video to upscale. |
| `scale` | String | `"2x"` | Upscaling factor. Options: `"2x"`, `"4x"`. |
| `driver` | String | `"realesrgan"` | Video2X driver to use (e.g., `realesrgan`, `waifu2x`). |
| `process_timeout` | Integer | `1800` | Maximum processing time in seconds. |

## Infrastructure Details

- **GPU**: NVIDIA A10G (24GB VRAM)
- **Base Image**: CUDA 12.6.3 on Ubuntu 24.04
- **Storage**: Modal Volume mounted at `/cache` for XDG_CACHE persistence.
