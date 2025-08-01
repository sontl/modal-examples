# MultiTalk Native - Modal.com Implementation

A native implementation of MultiTalk for Modal.com that supports audio-driven multi-person conversational video generation with single GPU inference.

## Features

- ✅ Single GPU inference (L40S)
- ✅ Single-person video generation
- ✅ Multi-person video generation  
- ✅ FusionX LoRA acceleration (8 steps)
- ✅ LightX2V LoRA acceleration (4 steps)
- ✅ INT8 quantization support
- ✅ Low VRAM mode
- ✅ TeaCache acceleration
- ✅ APG for color error reduction
- ✅ Gradio web interface
- ✅ CLI interface

## Quick Start

### Deploy to Modal

```bash
modal deploy 06_gpu_and_ml/multitalk_native/multitalk_native.py
```

### CLI Usage

```bash
# Single person generation
modal run 06_gpu_and_ml/multitalk_native/multitalk_native.py \
  --audio-path "path/to/audio.wav" \
  --image-path "path/to/reference.jpg" \
  --prompt "A person talking naturally" \
  --output-path "output.mp4"

# With FusionX LoRA acceleration
modal run 06_gpu_and_ml/multitalk_native/multitalk_native.py \
  --audio-path "path/to/audio.wav" \
  --image-path "path/to/reference.jpg" \
  --use-lora "fusionx" \
  --output-path "output_fusionx.mp4"

# With quantization for lower memory usage
modal run 06_gpu_and_ml/multitalk_native/multitalk_native.py \
  --audio-path "path/to/audio.wav" \
  --image-path "path/to/reference.jpg" \
  --use-quantization \
  --output-path "output_quantized.mp4"
```

### Web Interface

Access the Gradio interface at the deployed web server URL after running:

```bash
modal serve 06_gpu_and_ml/multitalk_native/multitalk_native.py::gradio_interface
```

### Python API Usage

```python
import modal

# Get the deployed class
MultiTalkNative = modal.Cls.lookup("multitalk-native", "MultiTalkNative")

# Single person generation
result = MultiTalkNative().generate_single_person.remote(
    audio_path="path/to/audio.wav",
    reference_image_path="path/to/reference.jpg", 
    prompt="A person talking naturally",
    use_lora="fusionx",  # or "lightx2v" or None
    use_quantization=True,
    low_vram=True
)

# Multi-person generation
result = MultiTalkNative().generate_multi_person.remote(
    audio_paths=["audio1.wav", "audio2.wav"],
    reference_images=["person1.jpg", "person2.jpg"],
    prompts=["Person 1 speaking", "Person 2 responding"],
    use_teacache=True,
    use_apg=True
)

if result["success"]:
    # Decode and save video
    import base64
    video_data = base64.b64decode(result["video_data"])
    with open("output.mp4", "wb") as f:
        f.write(video_data)
```

## Configuration Options

### Generation Modes
- `streaming`: Long video generation (default)
- `clip`: Short video generation

### LoRA Acceleration
- `fusionx`: FusionX LoRA (8 steps, ~2x faster)
- `lightx2v`: LightX2V LoRA (4 steps, ~4x faster)

### Optimization Options
- `use_teacache`: TeaCache acceleration (~2-3x speedup)
- `use_quantization`: INT8 quantization (lower memory)
- `low_vram`: Very low VRAM mode
- `use_apg`: APG for color error reduction in long videos

### Resolution
- `multitalk-480`: 480P output (default)
- `multitalk-720`: 720P output (requires more VRAM)

## Performance Comparison

| Configuration | Memory Usage | Speed | Quality |
|---------------|--------------|-------|---------|
| Standard (40 steps) | ~24GB | 1x | Best |
| FusionX LoRA (8 steps) | ~24GB | ~2x | Very Good |
| LightX2V LoRA (4 steps) | ~24GB | ~4x | Good |
| Quantized + Low VRAM | ~12GB | 0.8x | Good |
| TeaCache Enabled | Same | +2-3x | Same |

## Input Formats

### Single Person
```json
{
  "audio_path": "path/to/audio.wav",
  "reference_image": "path/to/reference.jpg", 
  "prompt": "A person talking naturally",
  "mode": "single"
}
```

### Multi-Person
```json
{
  "audio_paths": ["audio1.wav", "audio2.wav"],
  "reference_images": ["person1.jpg", "person2.jpg"],
  "prompts": ["Person 1 speaking", "Person 2 responding"],
  "mode": "multi"
}
```

## Model Requirements

The implementation automatically downloads these models:
- Wan2.1-I2V-14B-480P (base model)
- chinese-wav2vec2-base (audio encoder)
- MeiGen-MultiTalk (audio condition weights)
- FusionX LoRA (acceleration)
- LightX2V LoRA (acceleration)
- Kokoro-82M (TTS, optional)

Total download size: ~28GB

## Troubleshooting

### Out of Memory Errors
- Enable `low_vram=True`
- Use `use_quantization=True`
- Reduce `sample_steps` to 20-30

### Slow Generation
- Enable `use_teacache=True`
- Use LoRA acceleration (`fusionx` or `lightx2v`)
- Use `mode="clip"` for shorter videos

### Quality Issues
- Increase `sample_steps` to 50-60
- Disable LoRA for best quality
- Enable `use_apg=True` for long videos

## License

Apache 2.0 License - see the original MultiTalk repository for details.