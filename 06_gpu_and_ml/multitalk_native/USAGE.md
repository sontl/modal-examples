# MultiTalk Native - Usage Guide

## 🎉 Deployment Successful!

Your MultiTalk Native implementation has been successfully deployed to Modal.com!

## 🌐 Web Interface

Access the Gradio web interface at:
**https://sontl--multitalk-minimal-gradio-interface.modal.run**

Simply upload:
- An audio file (WAV/MP3)
- A reference image (JPG/PNG)
- Enter a prompt (e.g., "A person talking naturally")
- Click "Generate Video"

## 🖥️ CLI Usage

### Basic Generation
```bash
modal run 06_gpu_and_ml/multitalk_native/multitalk_minimal.py \
  --audio-path "path/to/your/audio.wav" \
  --image-path "path/to/your/reference.jpg" \
  --prompt "A person talking naturally" \
  --output-path "output.mp4"
```

### With Custom Settings
```bash
modal run 06_gpu_and_ml/multitalk_native/multitalk_minimal.py \
  --audio-path "audio.wav" \
  --image-path "reference.jpg" \
  --prompt "A person speaking with expressive movements" \
  --sample-steps 30 \
  --output-path "custom_output.mp4"
```

## 🧪 Testing

Run the test script to verify everything works:
```bash
python 06_gpu_and_ml/multitalk_native/test_minimal.py
```

This will:
1. Create synthetic test audio and image
2. Call the MultiTalk generation
3. Save the output video
4. Clean up test files

## 📊 Performance

The minimal version includes:
- ✅ Single GPU inference (L40S)
- ✅ Model download on first use (avoids build timeouts)
- ✅ TeaCache acceleration
- ✅ Low VRAM mode
- ✅ 480P video generation
- ✅ Gradio web interface
- ✅ CLI interface

**First run**: ~5-10 minutes (model download + generation)
**Subsequent runs**: ~2-5 minutes (generation only)

## 🔧 Configuration Options

### Sample Steps
- `10-20`: Fast, lower quality
- `30-40`: Balanced (default)
- `50+`: Slower, higher quality

### Modes
- `use_teacache=True`: 2-3x speedup (recommended)
- `low_vram=True`: Reduces memory usage for smaller GPUs

## 📝 Input Requirements

### Audio Files
- Format: WAV, MP3
- Duration: 1-30 seconds recommended
- Quality: 16kHz+ sample rate preferred

### Reference Images
- Format: JPG, PNG
- Resolution: 512x512 or higher
- Content: Clear face/person for best results

## 🚀 Next Steps

To add advanced features (LoRA acceleration, multi-person, quantization), you can extend the minimal implementation or use the full `multitalk_native.py` version once the basic setup is working.

## 🐛 Troubleshooting

### Common Issues

1. **First run is slow**: Models are downloading (~28GB). This is normal.
2. **Out of memory**: Enable `low_vram=True`
3. **Generation fails**: Check audio/image file formats and sizes
4. **Web interface not loading**: Wait a few minutes for cold start

### Getting Help

Check the logs in Modal dashboard:
https://modal.com/apps/sontl/main/deployed/multitalk-minimal

## 🎯 Success!

You now have a working MultiTalk implementation that can generate talking videos from audio and reference images. The system automatically handles:

- Model downloading and caching
- GPU memory management  
- File I/O and cleanup
- Error handling and logging

Enjoy creating talking videos! 🎬