# Video2X Video Upscaling API on Modal

This project provides a high-performance video upscaling API using **Video2X 6.2.0** deployed on Modal. It leverages NVIDIA T4 GPUs and Vulkan acceleration to upscale videos using state-of-the-art algorithms like **Real-ESRGAN**.

## 🚀 Key Features

- **High Speed**: Average processing speed of **~8 FPS** for 4x upscaling on a Tesla T4.
- **Vulkan Accelerated**: Optimized for GPU performance using Vulkan.
- **GPU Snapshots**: Enabled for near-instant cold starts.
- **Configurable**: Full control over scaling factor, models, codecs, and quality settings.

## 🛠️ Deployment

To deploy this API to your Modal account:

```bash
modal deploy modal-examples/misc/video2x_endpoint.py
```

This will create a persistent web endpoint.

## 📖 API Usage

### Endpoint
`POST /upscale`

### Request Body (JSON)

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `video_url` | `string` | **Required** | Publicly accessible URL of the video to upscale. |
| `scale` | `string` | `"4x"` | Upscaling factor: `"2x"`, `"3x"`, or `"4x"`. |
| `processor` | `string` | `"realesrgan"` | Algorithm to use: `"realesrgan"` or `"libplacebo"`. |
| `model` | `string` | `"realesr-animevideov3"` | RealESRGAN model: `realesr-animevideov3`, `realesrgan-plus-anime`, `realesrgan-plus`. |
| `codec` | `string` | `"libx264"` | Output video codec (e.g., `libx264`, `libx265`, `av1`). |
| `preset` | `string` | `"slow"` | Encoder preset: `ultrafast`, `fast`, `medium`, `slow`, etc. |
| `crf` | `integer` | `20` | Quality factor (0-51, lower = better quality). |

### Example Curl Request

```bash
curl -X POST https://your-modal-dev-endpoint.modal.run/upscale \
     -H "Content-Type: application/json" \
     -d '{
       "video_url": "https://example.com/input.mp4",
       "scale": "4x",
       "model": "realesr-animevideov3",
       "crf": 20
     }' \
     --output upscaled_video.mp4
```

### Python Client Example

```python
import requests

def upscale_video(url):
    api_url = "https://your-modal-endpoint.modal.run/upscale"
    payload = {
        "video_url": url,
        "scale": "4x",
        "model": "realesr-animevideov3",
        "crf": 18
    }
    
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        with open("upscaled.mp4", "wb") as f:
            f.write(response.content)
        print("Upscaling complete!")
    else:
        print(f"Error: {response.text}")
```

## 🏗️ Technical Implementation

- **Base Image**: CUDA 12.1.1 on Ubuntu 22.04.
- **Architecture**: Dynamically find and configure NVIDIA Vulkan ICD at runtime to ensure compatibility with Modal's driver injection.
- **Storage**: Uses Modal Volumes for model caching and ephemeral storage for processing jobs.
- **Snapshotting**: Uses `enable_gpu_snapshot` to capture initialized Vulkan drivers and state, reducing startup latency.

## 🧪 Benchmark Results (T4 GPU)

| Input | Output | Frames | Time | Avg. FPS |
| :--- | :--- | :--- | :--- | :--- |
| 360p (10s) | 1440p (4x) | 298 | ~37s | **8.05 FPS** |

*Note: Performance may vary based on video complexity and Modal's dynamic scaling.*
