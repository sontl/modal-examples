# YuE Music Generation API Documentation

Complete REST API reference for **YuE2 Full-Song Generation, Zero-Shot Covers, Audio Extension, and SheetSage2 Audio Transcription** deployed on [Modal](https://modal.com) with NVIDIA A10G (24GB VRAM).

---

## Table of Contents
1. [Overview & Base URL](#overview--base-url)
2. [Authentication](#authentication)
3. [Endpoints Summary](#endpoints-summary)
4. [Endpoint Reference](#endpoint-reference)
   - [1. Health Check (`GET /health`)](#1-health-check-get-health)
   - [2. Generate Song Audio (`POST /generate_audio`)](#2-generate-song-audio-post-generate_audio)
   - [3. Generate Song JSON (`POST /generate`)](#3-generate-song-json-post-generate)
   - [4. Generate Song Asynchronous (`POST /generate_async`)](#4-generate-song-asynchronous-post-generate_async)
   - [5. Zero-Shot Cover Audio (`POST /cover_audio`)](#5-zero-shot-cover-audio-post-cover_audio)
   - [6. Zero-Shot Cover JSON (`POST /cover`)](#6-zero-shot-cover-json-post-cover)
   - [7. Extend Song Audio (`POST /extend_audio`)](#7-extend-song-audio-post-extend_audio)
   - [8. Extend Song JSON (`POST /extend`)](#8-extend-song-json-post-extend)
   - [9. Transcribe Audio to ABC Score (`POST /transcribe`)](#9-transcribe-audio-to-abc-score-post-transcribe)
5. [Client Code Examples](#client-code-examples)
   - [Python Client](#python-client)
   - [TypeScript / JavaScript Client](#typescript--javascript-client)
   - [cURL](#curl)
6. [Field Definitions & Prompting Tips](#field-definitions--prompting-tips)

---

## Overview & Base URL

When deployed via `modal deploy misc/yue_endpoint.py`, Modal exposes dedicated URLs for each endpoint based on your workspace name:

```text
Base Pattern: https://<workspace>--yue-music-generator-<service>-<endpoint>.modal.run
```

For workspace `riyantibakijan1`:
- **YuEService Endpoints**: `https://riyantibakijan1--yue-music-generator-yueservice-<endpoint_name>.us-east.modal.run`
- **Standalone Endpoints**: `https://riyantibakijan1--yue-music-generator-<endpoint_name>.us-east.modal.run`

All requests and responses use `application/json` (except direct audio streaming endpoints which return binary audio formats `audio/mpeg`, `audio/wav`, or `audio/flac`).

---

## Authentication

The API provides built-in token authentication via FastAPI dependencies supporting both `Authorization: Bearer <token>` and `X-API-Key: <token>` headers.

### Open vs. Protected Mode
- **Protected Mode**: When the Modal Secret named `yue-api-secret` (containing key `YUE_API_KEY`) is created, authentication is **strictly enforced**. Any request lacking a valid token receives `401 Unauthorized`.
- **Open Mode**: If `yue-api-secret` does not exist, endpoints remain open for rapid local development and testing. The `/health` endpoint always indicates whether authentication is active via `"auth_enabled": true/false`.

### How to Create and Enable an Auth Token

#### Step 1: Generate or Pick a Secret Token
Choose any strong secret string, or generate one in your terminal:
```bash
# Generate a secure 32-character hexadecimal token
openssl rand -hex 16
# e.g.: a3f1c8e9b042d7658f12a9e04bc53de8
```

#### Step 2: Create the Modal Secret
Run the `modal secret create` command:
```bash
modal secret create yue-api-secret YUE_API_KEY="a3f1c8e9b042d7658f12a9e04bc53de8"
```
*(Alternatively, in the [Modal Web Dashboard](https://modal.com/secrets), click **Create Secret** → select **Custom Secret** → name it `yue-api-secret` → add key `YUE_API_KEY` with your token value.)*

#### Step 3: Deploy to Apply the Secret
Re-run deployment so Modal attaches the newly created secret to your container instances:
```bash
modal deploy misc/yue_endpoint.py
```

#### Step 4: Verify Auth Status
Check the public `/health` endpoint:
```bash
curl https://riyantibakijan1--yue-music-generator-health.us-east.modal.run
```
Response:
```json
{
  "status": "healthy",
  "auth_enabled": true,
  ...
}
```

### Passing the Auth Token in Requests

You can use either standard header format:

```http
Authorization: Bearer <YOUR_API_TOKEN>
```
or
```http
X-API-Key: <YOUR_API_TOKEN>
```

---

## Endpoints Summary

| Endpoint | Method | Input Type | Output Type | Description |
| :--- | :---: | :--- | :--- | :--- |
| [`/health`](#1-health-check-get-health) | `GET` | None | JSON | Health status, active models, and capabilities |
| [`/generate_audio`](#2-generate-song-audio-post-generate_audio) | `POST` | JSON | Binary Audio Stream | Direct audio streaming from lyrics & style prompt |
| [`/generate`](#3-generate-song-json-post-generate) | `POST` | JSON | JSON | Returns base64 audio + generated ABC score plan |
| [`/generate_async`](#4-generate-song-asynchronous-post-generate_async) | `POST` | JSON | JSON | Enqueues background task; delivers to `webhook_url` |
| [`/cover_audio`](#5-zero-shot-cover-audio-post-cover_audio) | `POST` | JSON | Binary Audio Stream | Re-renders existing audio or ABC score in a new genre |
| [`/cover`](#6-zero-shot-cover-json-post-cover) | `POST` | JSON | JSON | Returns base64 cover audio + extracted ABC melody |
| [`/extend_audio`](#7-extend-song-audio-post-extend_audio) | `POST` | JSON | Binary Audio Stream | Extends song with new sections (streams audio) |
| [`/extend`](#8-extend-song-json-post-extend) | `POST` | JSON | JSON | Extends song with new sections (returns base64 JSON) |
| [`/transcribe`](#9-transcribe-audio-to-abc-score-post-transcribe) | `POST` | JSON | JSON | SheetSage2 audio-to-ABC transcription |

---

## Endpoint Reference

### 1. Health Check (`GET /health`)
**URL**: `https://<workspace>--yue-music-generator-health.us-east.modal.run`

#### Response (`200 OK`)
```json
{
  "status": "healthy",
  "service": "yue-music-generator",
  "models": {
    "generation": "m-a-p/YuE2-3B",
    "decoder": "m-a-p/YuE2-Vae",
    "transcription": "m-a-p/SheetSage2"
  },
  "capabilities": [
    "text-to-full-song",
    "zero-shot-cover (audio or ABC)",
    "song-extension",
    "audio-to-abc-transcription"
  ],
  "gpu_target": "NVIDIA A10G (24GB VRAM)",
  "output_sample_rate": 48000,
  "supported_formats": ["mp3", "wav", "flac"],
  "supported_cot_modes": ["full", "melody", "off"]
}
```

---

### 2. Generate Song Audio (`POST /generate_audio`)
**URL**: `https://<workspace>--yue-music-generator-yueservice-generate-audio.us-east.modal.run`

Generates full-song music and directly returns the binary audio file.

#### Request Body (`application/json`)
```json
{
  "lyrics": "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n[Chorus]\nLet the day come into view\nEvery road begins with you",
  "style": "English, warm piano pop, expressive female voice, acoustic piano, 88 BPM",
  "cot": "full",
  "seed": 831001,
  "output_format": "mp3",
  "cfg_scale": 1.0,
  "request_id": "req_123"
}
```

#### Request Parameters
| Field | Type | Required | Default | Description |
| :--- | :--- | :---: | :--- | :--- |
| `lyrics` | string | **Yes** | — | Lyrics formatted with section tags like `[Verse]`, `[Chorus]` |
| `style` | string | **Yes** | — | Style description: genre, vocal character, tempo (BPM) |
| `cot` | string | No | `"full"` | Symbolic planning: `"full"` (melody + chords), `"melody"`, or `"off"` |
| `seed` | integer | No | `831001` | Seed for deterministic reproduction |
| `output_format` | string | No | `"mp3"` | `"mp3"` (320kbps), `"wav"`, or `"flac"` (24-bit 48kHz) |
| `abc` | string | No | `null` | Pre-existing ABC notation score (optional direct control) |
| `cfg_scale` | float | No | `1.0` | Guidance scale (1.0 - 5.0) |
| `request_id` | string | No | auto | Client tracking ID |

#### Response Headers
- `Content-Type`: `audio/mpeg` (or `audio/wav`, `audio/flac`)
- `Content-Disposition`: `inline; filename="req_123.mp3"`
- `X-Duration-Seconds`: `"34.52"`
- `X-Sample-Rate`: `"48000"`
- `X-Request-ID`: `"req_123"`

---

### 3. Generate Song JSON (`POST /generate`)
**URL**: `https://<workspace>--yue-music-generator-yueservice-generate.us-east.modal.run`

Same generation inputs as above, but returns base64-encoded audio and full score metadata.

#### Response Body (`200 OK`)
```json
{
  "success": true,
  "message": "Song generated successfully.",
  "request_id": "yue_1792594275543",
  "audio_base64": "//uQZAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAAYAAAG...",
  "output_format": "mp3",
  "sample_rate": 48000,
  "duration_seconds": 34.68,
  "score_abc": "X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=88\nV: Vocal ...\nK:C\n...",
  "timing": {
    "abc": { "seconds": 5.1 },
    "semantic": { "seconds": 11.2 },
    "nar_seconds": 4.7,
    "vae_seconds": 0.9,
    "e2e_seconds": 32.87
  },
  "truncated": {
    "abc": false,
    "semantic": false
  }
}
```

---

### 4. Generate Song Asynchronous (`POST /generate_async`)
**URL**: `https://<workspace>--yue-music-generator-yueservice-generate-async.us-east.modal.run`

Enqueues the generation job as a non-blocking background task.

#### Request Body
```json
{
  "lyrics": "[Verse]\nDriving down the highway...",
  "style": "Synthwave, retro 80s drums, energetic vocals",
  "webhook_url": "https://your-api.com/webhooks/music-ready",
  "request_id": "order_789"
}
```

#### Immediate Response (`200 OK`)
```json
{
  "success": true,
  "task_id": "ta-01M5FER56KYDMKAWF61BH6YRNR",
  "request_id": "order_789",
  "message": "Generation task enqueued. Results will be delivered to webhook_url upon completion."
}
```

#### Webhook Payload (Delivered via HTTP POST to `webhook_url`)
Contains the identical schema as the JSON response (`audio_base64`, `score_abc`, `duration_seconds`, `request_id`, etc.).

---

### 5. Zero-Shot Cover Audio (`POST /cover_audio`)
**URL**: `https://<workspace>--yue-music-generator-yueservice-cover-audio.us-east.modal.run`

Transcribes the melody from a reference audio file (or provided score) and re-synthesizes the entire song in a new genre/style. Returns a binary audio stream.

#### Request Body
```json
{
  "audio_url": "https://example.com/reference_piano_ballad.mp3",
  "style": "Heavy metal, distorted electric guitars, fast double-kick drums, energetic rock vocals, 130 BPM",
  "lyrics": "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n[Chorus]\nLet the day come into view",
  "cot": "melody",
  "output_format": "mp3"
}
```

#### Request Parameters
| Field | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `style` | string | **Yes** | Target genre/timbre/instrumentation for the cover |
| `lyrics` | string | **Yes** | Lyrics to sing, matching the reference melody |
| `audio_url` | string | Optional* | Public URL of source audio to transcribe |
| `audio_base64` | string | Optional* | Base64-encoded source audio to transcribe |
| `score_abc` | string | Optional* | Pre-extracted ABC melody score (bypasses transcription) |
| `cot` | string | No (`"melody"`) | Keep as `"melody"` to give accompaniment harmonic freedom |
| `output_format` | string | No (`"mp3"`) | `"mp3"`, `"wav"`, or `"flac"` |

*\*At least one of `audio_url`, `audio_base64`, or `score_abc` must be supplied.*

---

### 6. Zero-Shot Cover JSON (`POST /cover`)
**URL**: `https://<workspace>--yue-music-generator-yueservice-cover.us-east.modal.run`

Same input as `/cover_audio`, but returns the structured JSON response containing `audio_base64` and the extracted `score_abc`.

---

### 7. Extend Song Audio (`POST /extend_audio`)
**URL**: `https://<workspace>--yue-music-generator-yueservice-extend-audio.us-east.modal.run`

Extends a previously generated song with new verses/choruses/outro while preserving musical themes and chord progressions. Returns a binary audio stream.

#### Request Body
```json
{
  "original_score_abc": "X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=88\n...",
  "style": "English, warm piano pop, expressive female voice, acoustic piano, 88 BPM",
  "lyrics": "[Verse 1]\nNeon fades along the lane\n[Chorus]\nLet the day come into view\n[Verse 2]\nMorning has a sky to clear\nSunlight drying every tear\n[Outro]\nWe will sing beyond the night",
  "cot": "full",
  "output_format": "mp3"
}
```

#### Request Parameters
| Field | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `lyrics` | string | **Yes** | Complete extended lyrics (original + new sections) |
| `style` | string | **Yes** | Musical style prompt (usually same style as original) |
| `original_score_abc` | string | Optional* | ABC score returned by previous generation |
| `original_audio_url` | string | Optional* | Audio URL of original track if ABC was not retained |
| `cot` | string | No (`"full"`) | `"full"` (preserves harmony) or `"melody"` |

---

### 8. Extend Song JSON (`POST /extend`)
**URL**: `https://<workspace>--yue-music-generator-yueservice-extend.us-east.modal.run`

Same input as `/extend_audio`, returning JSON with `audio_base64`, `score_abc`, and timing metrics.

---

### 9. Transcribe Audio to ABC Score (`POST /transcribe`)
**URL**: `https://<workspace>--yue-music-generator-transcribe.us-east.modal.run`

Extracts lead melodies or full lead-sheets into editable ABC score notation using SheetSage2.

#### Request Body
```json
{
  "audio_url": "https://example.com/source_song.mp3",
  "melody_only": true
}
```

#### Response Body (`200 OK`)
```json
{
  "success": true,
  "score_abc": "X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=88\nV: Vocal clef=treble name=\"Vocal Melody\"\n...",
  "warnings": []
}
```

---

## Client Code Examples

### Python Client

```python
import requests
from pathlib import Path

BASE_URL = "https://riyantibakijan1--yue-music-generator"

class YuEMusicClient:
    def __init__(self, workspace: str = "riyantibakijan1", api_key: str | None = None):
        self.host = f"https://{workspace}--yue-music-generator"
        self.api_key = api_key

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate_song(self, style: str, lyrics: str, output_path: str = "song.mp3", output_format: str = "mp3"):
        """Generate music and save directly to audio file."""
        url = f"{self.host}-yueservice-generate-audio.us-east.modal.run"
        payload = {
            "style": style,
            "lyrics": lyrics,
            "cot": "full",
            "output_format": output_format,
        }
        print(f"Generating song via {url}...")
        response = requests.post(url, json=payload, headers=self._headers(), timeout=600)
        response.raise_for_status()

        Path(output_path).write_bytes(response.content)
        print(f"Saved: {output_path} (Duration: {response.headers.get('X-Duration-Seconds')}s)")

    def create_cover(self, audio_url: str, new_style: str, lyrics: str, output_path: str = "cover.mp3"):
        """Create a zero-shot genre cover from an existing audio URL."""
        url = f"{self.host}-yueservice-cover-audio.us-east.modal.run"
        payload = {
            "audio_url": audio_url,
            "style": new_style,
            "lyrics": lyrics,
            "cot": "melody",
            "output_format": "mp3",
        }
        print(f"Generating cover via {url}...")
        response = requests.post(url, json=payload, headers=self._headers(), timeout=600)
        response.raise_for_status()

        Path(output_path).write_bytes(response.content)
        print(f"Saved cover: {output_path}")

    def transcribe(self, audio_url: str) -> str:
        """Transcribe audio URL to ABC notation score."""
        url = f"{self.host}-transcribe.us-east.modal.run"
        response = requests.post(url, json={"audio_url": audio_url, "melody_only": True}, headers=self._headers(), timeout=300)
        response.raise_for_status()
        return response.json()["score_abc"]

# Example usage
if __name__ == "__main__":
    # Pass api_key if you configured yue-api-secret, otherwise None
    client = YuEMusicClient("riyantibakijan1", api_key=None)
    
    # 1. Text to Music
    client.generate_song(
        style="English, synthwave, retro 80s drums, driving bassline, male vocal",
        lyrics="[Verse]\nRacing through the city glow\n[Chorus]\nNever letting go",
        output_path="synthwave.mp3"
    )
```

---

### TypeScript / JavaScript Client

```typescript
import fs from "fs/promises";

interface GenerateParams {
  style: string;
  lyrics: string;
  cot?: "full" | "melody" | "off";
  output_format?: "mp3" | "wav" | "flac";
}

interface CoverParams {
  audio_url?: string;
  audio_base64?: string;
  score_abc?: string;
  style: string;
  lyrics: string;
  output_format?: "mp3" | "wav" | "flac";
}

export class YuEMusicClient {
  private workspace: string;
  private apiKey?: string;

  constructor(workspace: string = "riyantibakijan1", apiKey?: string) {
    this.workspace = workspace;
    this.apiKey = apiKey;
  }

  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  /**
   * Generates a song and returns the audio as a Buffer.
   */
  async generateAudio(params: GenerateParams): Promise<Buffer> {
    const url = `https://${this.workspace}--yue-music-generator-yueservice-generate-audio.us-east.modal.run`;
    const response = await fetch(url, {
      method: "POST",
      headers: this.getHeaders(),
      body: JSON.stringify({
        cot: "full",
        output_format: "mp3",
        ...params,
      }),
    });

    if (!response.ok) {
      throw new Error(`Generation failed: ${response.statusText} (${await response.text()})`);
    }

    const arrayBuffer = await response.arrayBuffer();
    return Buffer.from(arrayBuffer);
  }

  /**
   * Generates a zero-shot cover of an existing song.
   */
  async coverAudio(params: CoverParams): Promise<Buffer> {
    const url = `https://${this.workspace}--yue-music-generator-yueservice-cover-audio.us-east.modal.run`;
    const response = await fetch(url, {
      method: "POST",
      headers: this.getHeaders(),
      body: JSON.stringify({
        cot: "melody",
        output_format: "mp3",
        ...params,
      }),
    });

    if (!response.ok) {
      throw new Error(`Cover failed: ${response.statusText}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    return Buffer.from(arrayBuffer);
  }
}

// Example usage
async function run() {
  const client = new YuEMusicClient("riyantibakijan1");

  const audioBuffer = await client.generateAudio({
    style: "English, acoustic folk, mellow guitar, gentle male voice, 85 BPM",
    lyrics: "[Verse]\nWalking down the quiet street\n[Chorus]\nTake me where the wind can go",
  });

  await fs.writeFile("output.mp3", audioBuffer);
  console.log("Audio written to output.mp3");
}
```

---

### cURL

#### 1. Generate MP3 Directly
```bash
curl -X POST "https://riyantibakijan1--yue-music-generator-yueservice-generate-audio.us-east.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "style": "English, warm piano pop, expressive female voice, 88 BPM",
    "lyrics": "[Verse]\nNeon fades along the lane\n[Chorus]\nLet the day come into view",
    "output_format": "mp3"
  }' \
  --output "song.mp3"
```

#### 2. Generate Cover from Audio URL
```bash
curl -X POST "https://riyantibakijan1--yue-music-generator-yueservice-cover-audio.us-east.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/reference_track.mp3",
    "style": "Cyberpunk synth, aggressive synths, distorted vocals, 128 BPM",
    "lyrics": "[Verse]\nNeon fades along the lane\n[Chorus]\nLet the day come into view",
    "output_format": "mp3"
  }' \
  --output "cyberpunk_cover.mp3"
```

---

## Field Definitions & Prompting Tips

### Section Tags
Format `lyrics` with square brackets for natural lyrical structure:
- `[Intro]`
- `[Verse]` / `[Verse 1]` / `[Verse 2]`
- `[Pre-Chorus]`
- `[Chorus]`
- `[Bridge]`
- `[Outro]`

### Style Prompt Formula
For highest fidelity generations, format `style` with:
$$\text{[Language]}, \text{[Genre/Subgenre]}, \text{[Vocal Timbre]}, \text{[Primary Instruments]}, \text{[Tempo/BPM]}$$

*Example*: `"English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, 88 BPM"`
