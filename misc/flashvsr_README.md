# FlashVSR Video Super-Resolution API

A Modal endpoint for FlashVSR video super-resolution using the official inference scripts.

## Features

- **Real FlashVSR**: Uses official `infer_flashvsr_full.py` and `infer_flashvsr_tiny.py` scripts
- **Two Models**: Full model (best quality) and Tiny model (faster)
- **A100 Optimized**: Designed for NVIDIA A100 GPUs
- **Block-Sparse Attention**: Includes LCSA for efficient inference

## Deployment

```bash
modal deploy misc/flashvsr_endpoint.py
```

## Usage

### API Request

```json
{
  "video_url": "https://example.com/video.mp4",
  "model_type": "full",  // "full" or "tiny"
  "output_format": "mp4",
  "max_frames": 100
}
```

### cURL Example

```bash
curl -X POST "https://your-endpoint.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    "model_type": "full",
    "max_frames": 10
  }' \
  --output flashvsr_output.mp4
```

## Testing

```bash
# Update ENDPOINT_URL in the test file first
python misc/test_flashvsr.py
```

## Model Types

- **Full Model**: Best quality, slower processing
- **Tiny Model**: Faster processing, good quality

## Files

- `misc/flashvsr_endpoint.py` - Main Modal endpoint
- `misc/test_flashvsr.py` - Test script
- `misc/flashvsr_README.md` - This documentation