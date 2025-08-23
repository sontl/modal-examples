# MultiTalk Fixed Implementation

This is a fixed implementation of MultiTalk for Modal.com that resolves the JSON structure issues and provides proper support for both single and multi-person video generation.

## Key Fixes

### 1. JSON Structure Fix
The original error was caused by incorrect JSON structure. MultiTalk expects:

**Single Person (correct):**
```json
{
  "prompt": "A person talking naturally",
  "cond_image": "/path/to/image.jpg",
  "cond_audio": {
    "person1": "/path/to/audio.wav"
  }
}
```

**Multi-Person Add (correct):**
```json
{
  "prompt": "Two people having a conversation",
  "cond_image": "/path/to/image.jpg",
  "audio_type": "add",
  "cond_audio": {
    "person1": "/path/to/audio1.wav",
    "person2": "/path/to/audio2.wav"
  },
  "bbox": {
    "person1": [160, 120, 1280, 1080],
    "person2": [160, 1320, 1280, 2280]
  }
}
```

**Multi-Person Parallel (correct):**
```json
{
  "prompt": "Two people singing together",
  "cond_image": "/path/to/image.jpg",
  "audio_type": "para",
  "cond_audio": {
    "person1": "/path/to/audio1.wav",
    "person2": "/path/to/audio2.wav"
  }
}
```

### 2. Audio Type Support
- `single`: Single person talking (default)
- `add`: Two people with additive audio (conversation)
- `para`: Two people with parallel audio (singing together)

### 3. Bounding Box Support
Optional bounding boxes for multi-person videos to specify where each person should appear in the frame.

## Usage

### CLI Usage

**Single Person:**
```bash
modal run multitalk_fixed.py --audio-path audio.wav --image-path image.jpg
```

**Multi-Person Conversation:**
```bash
modal run multitalk_fixed.py \
  --audio-path person1.wav \
  --image-path image.jpg \
  --audio-path-person2 person2.wav \
  --audio-type add
```

**Multi-Person Singing:**
```bash
modal run multitalk_fixed.py \
  --audio-path person1.wav \
  --image-path image.jpg \
  --audio-path-person2 person2.wav \
  --audio-type para
```

### API Usage

**Single Person via FastAPI:**
```python
import requests

files = {
    'audio': open('audio.wav', 'rb'),
    'image': open('image.jpg', 'rb')
}

data = {
    'prompt': 'A person speaking naturally',
    'sample_steps': 40,
    'audio_type': 'single'
}

response = requests.post('https://your-app.modal.run/generate', files=files, data=data)
```

**Multi-Person via FastAPI:**
```python
import requests
import json

files = {
    'audio': open('person1.wav', 'rb'),
    'audio_person2': open('person2.wav', 'rb'),
    'image': open('image.jpg', 'rb')
}

data = {
    'prompt': 'Two people having a conversation',
    'sample_steps': 40,
    'audio_type': 'add',
    'bbox_person1': json.dumps([160, 120, 1280, 1080]),
    'bbox_person2': json.dumps([160, 1320, 1280, 2280])
}

response = requests.post('https://your-app.modal.run/generate', files=files, data=data)
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /generate` - Generate video, return as binary MP4
- `POST /generate-json` - Generate video, return as base64 JSON

## Parameters

### Required
- `audio`: Audio file for person 1
- `image`: Reference image

### Optional
- `audio_person2`: Audio file for person 2 (multi-person only)
- `prompt`: Generation prompt (default: "A person talking naturally")
- `sample_steps`: Number of sampling steps 4-50 (default: 40)
- `audio_type`: "single", "add", or "para" (default: "single")
- `bbox_person1`: Bounding box for person 1 as JSON array [x,y,w,h]
- `bbox_person2`: Bounding box for person 2 as JSON array [x,y,w,h]
- `use_teacache`: Enable TeaCache optimization (default: true)
- `low_vram`: Enable low VRAM mode (default: false)

## Error Resolution

The original error:
```
TypeError: string indices must be integers
```

Was caused by the script expecting `input_data['cond_audio']['person1']` but receiving a string instead of a dictionary structure. This is now fixed by properly structuring the JSON input.

## Model Requirements

The implementation automatically downloads required models:
- Wan2.1-I2V-14B-480P (base model)
- chinese-wav2vec2-base (audio processing)
- MeiGen-MultiTalk (MultiTalk weights)

Models are cached in Modal volumes for faster subsequent runs.

## Testing

Run the test script to verify JSON structures:
```bash
python test_json_structure.py
```

This will show the correct JSON formats for all supported scenarios.