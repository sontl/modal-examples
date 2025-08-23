# InfiniteTalk Docker Deployment

This Docker setup allows you to run InfiniteTalk on platforms like RunPod, Vast.ai, or any GPU-enabled Docker environment, as an alternative to Modal.

## Features

- **CUDA 12.1 Support**: Optimized for modern GPUs
- **Pre-downloaded Models**: All required models are downloaded during build
- **FastAPI Server**: RESTful API compatible with RunPod and other platforms
- **Memory Efficient**: Supports low VRAM mode for smaller GPUs
- **Multiple Resolutions**: Supports both 480P and 720P generation

## Requirements

- NVIDIA GPU with at least 16GB VRAM (recommended)
- Docker with NVIDIA Container Toolkit
- ~50GB disk space for models and dependencies

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository (if not already done)
git clone https://github.com/MeiGen-AI/InfiniteTalk.git
cd InfiniteTalk

# Copy the Dockerfile and docker-compose.yml to the InfiniteTalk directory
# (or create them in the same directory as this README)

# Build and run
docker-compose up --build
```

### Option 2: Docker Build & Run

```bash
# Build the image (this will take 20-30 minutes due to model downloads)
docker build -t infinitetalk .

# Run the container
docker run -d \
  --name infinitetalk \
  --gpus all \
  -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=0 \
  infinitetalk
```

## API Usage

Once running, the API will be available at `http://localhost:8000`

### Health Check
```bash
curl http://localhost:8000/health
```

### Generate Video
```bash
curl -X POST "http://localhost:8000/generate" \
  -F "audio=@your_audio.wav" \
  -F "image=@your_image.jpg" \
  -F "prompt=A person talking naturally" \
  -F "sample_steps=40" \
  -F "size=infinitetalk-480"
```

### API Parameters

- **audio**: Audio file (WAV format recommended)
- **image**: Image file (JPG/PNG)
- **prompt**: Text description (default: "A person talking naturally")
- **sample_steps**: Number of diffusion steps (default: 40)
- **use_teacache**: Enable TeaCache acceleration (default: true)
- **low_vram**: Enable low VRAM mode (default: false)
- **size**: Resolution - "infinitetalk-480" or "infinitetalk-720" (default: 480)
- **motion_frame**: Motion frame parameter (default: 9)
- **mode**: Generation mode - "streaming" or "clip" (default: streaming)

## RunPod Deployment

1. Create a new RunPod template with this Docker image
2. Set the container disk size to at least 50GB
3. Expose port 8000
4. Use a GPU with at least 16GB VRAM (RTX 4090, A100, etc.)

### RunPod Template Settings
```
Container Image: your-registry/infinitetalk:latest
Container Disk: 50GB
Expose HTTP Ports: 8000
GPU: RTX 4090 or better
```

## Performance Notes

- **First run**: Model loading takes 2-3 minutes
- **Generation time**: 30-60 seconds per video depending on length and GPU
- **Memory usage**: ~14-16GB VRAM for 480P, ~18-20GB for 720P
- **TeaCache**: Significantly speeds up generation (enabled by default)

## Troubleshooting

### Out of Memory
- Set `low_vram=true` in API calls
- Use 480P instead of 720P
- Reduce `sample_steps` to 20-30

### Slow Generation
- Ensure `use_teacache=true`
- Check GPU utilization with `nvidia-smi`
- Verify CUDA is properly configured

### Model Loading Issues
- Check container logs: `docker logs infinitetalk`
- Ensure sufficient disk space (50GB+)
- Verify internet connection during build

## Model Information

The following models are automatically downloaded during build:

- **Wan2.1-I2V-14B-480P**: Base video generation model (~25GB)
- **chinese-wav2vec2-base**: Audio processing model (~1GB)
- **InfiniteTalk**: Fine-tuned weights (~500MB)

## Development

To modify the server or add features:

1. Edit `server.py` in the Dockerfile
2. Rebuild the image: `docker build -t infinitetalk .`
3. Test locally before deploying

## License

This Docker setup follows the same license as the original InfiniteTalk project.
