# Gemini Image API

OpenAI-compatible REST API for Google Gemini image generation.

## Features

- **Text-to-Image** - Generate images from text prompts
- **Image-to-Image** - Edit/transform images with natural language
- **OpenAI Compatible** - Drop-in replacement for OpenAI's image API
- **Base64 Output** - Returns images as base64 encoded strings
- **Aspect Ratio Control** - Support for landscape, portrait, and square formats
- **Multi-Image** - Generate up to 4 images per request

## Installation

```bash
# Clone or download the project
cd gemini-image-api

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### 1. Get Google Cookies

1. Go to [gemini.google.com](https://gemini.google.com) and login
2. Press **F12** → Open DevTools
3. Go to **Application** → **Cookies** → `https://gemini.google.com`
4. Copy these cookie values:
   - `__Secure-1PSID` (required)
   - `__Secure-1PSIDTS` (optional but recommended)

### 2. Edit config.json

```bash
cp config.example.json config.json
```

```json
{
    "api_key": "sk-your-secret-key",
    "cookie_1PSID": "your__Secure-1PSID_value",
    "cookie_1PSIDTS": "your__Secure-1PSIDTS_value",
    "host": "0.0.0.0",
    "port": 8000,
    "default_model": "gemini-2.5-flash",
    "timeout": 120
}
```

## Usage

### Start the Server

```bash
python3 api_server.py
```

Server runs at: `http://localhost:8000`

### API Documentation

Once running, visit: `http://localhost:8000/docs` for interactive Swagger UI.

---

## API Endpoints

### POST `/v1/images/generations`

Generate images from text prompts.

**Request:**
```json
{
    "model": "gemini-2.5-flash",
    "prompt": "A cute cat playing with yarn",
    "n": 1,
    "size": "landscape",
    "seed": 12345
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Text description of the image |
| `model` | string | No | gemini-2.5-flash | Model to use |
| `n` | integer | No | 1 | Number of images (1-4) |
| `size` | string | No | null | Aspect ratio (see below) |
| `seed` | integer | No | null | Seed for generation |
| `response_format` | string | No | b64_json | Output format |

**Size Options:**

| Value | Aspect Ratio | Description |
|-------|--------------|-------------|
| `landscape` or `16:9` | 16:9 | Wide cinematic |
| `portrait` or `9:16` | 9:16 | Tall vertical |
| `square` or `1:1` | 1:1 | Square |

**Response:**
```json
{
    "created": 1704067200,
    "data": [
        {
            "b64_json": "iVBORw0KGgoAAAANSUhEUgAA...",
            "revised_prompt": "A cute cat playing with yarn"
        }
    ]
}
```

---

### POST `/v1/images/edits`

Edit an existing image with a text prompt.

**Request:** (multipart/form-data)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | file | Yes | Image file to edit |
| `prompt` | string | Yes | Edit instruction |
| `model` | string | No | Model to use |
| `n` | integer | No | Number of results (1-4) |
| `size` | string | No | Aspect ratio |

**Response:**
```json
{
    "created": 1704067200,
    "data": [
        {
            "b64_json": "iVBORw0KGgoAAAANSUhEUgAA...",
            "revised_prompt": "Edit instruction"
        }
    ]
}
```

---

### GET `/v1/models`

List available models.

**Response:**
```json
{
    "data": [
        {"id": "gemini-2.5-flash", "object": "model", "owned_by": "google"},
        {"id": "gemini-2.5-pro", "object": "model", "owned_by": "google"},
        {"id": "gemini-3.0-pro", "object": "model", "owned_by": "google"}
    ]
}
```

---

### GET `/health`

Health check endpoint.

---

## Examples

### Generate Image (curl)

```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Authorization: Bearer sk-demo" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A beautiful sunset over mountains", "n": 1}'
```

### Generate and Save to PNG

```bash
curl -s -X POST http://localhost:8000/v1/images/generations \
  -H "Authorization: Bearer sk-demo" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cute cat", "size": "square"}' \
  | jq -r '.data[0].b64_json' \
  | base64 -d > output.png
```

### Generate Landscape Image

```bash
curl -s -X POST http://localhost:8000/v1/images/generations \
  -H "Authorization: Bearer sk-demo" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Mountain landscape at golden hour", "size": "landscape"}' \
  | jq -r '.data[0].b64_json' \
  | base64 -d > landscape.png
```

### Edit Image

```bash
curl -X POST http://localhost:8000/v1/images/edits \
  -H "Authorization: Bearer sk-demo" \
  -F "image=@input.png" \
  -F "prompt=Add a rainbow in the sky" \
  -F "n=1" \
  | jq -r '.data[0].b64_json' \
  | base64 -d > edited.png
```

### Python Example

```python
import requests
import base64

response = requests.post(
    "http://localhost:8000/v1/images/generations",
    headers={
        "Authorization": "Bearer sk-demo",
        "Content-Type": "application/json"
    },
    json={
        "prompt": "A futuristic cityscape",
        "size": "landscape",
        "n": 1
    }
)

data = response.json()
image_b64 = data["data"][0]["b64_json"]

# Save to file
with open("output.png", "wb") as f:
    f.write(base64.b64decode(image_b64))
```

---

## Available Models

| Model | Description |
|-------|-------------|
| `gemini-2.5-flash` | Fast model (default) |
| `gemini-2.5-pro` | Pro model |
| `gemini-3.0-pro` | Latest Pro model |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Cookie expired" | Refresh gemini.google.com and update cookies in config.json |
| "No images generated" | Try adding "generate" to your prompt |
| "Connection refused" | Make sure server is running (`python3 api_server.py`) |
| Keychain popup (macOS) | Click "Deny" - manual cookies in config.json are used |
| Wrong aspect ratio | Gemini may not always follow size hints exactly |

---

## Limitations

- **Cookie Expiration**: `__Secure-1PSIDTS` expires frequently (hours to days)
- **No Official API**: Uses web interface, may break with Google updates
- **Aspect Ratio**: Size parameter uses prompt hints, not guaranteed
- **Rate Limits**: Subject to Google's web interface limits
- **Region Restrictions**: Image generation may not work in all regions

---

## Project Structure

```
gemini-image-api/
├── api_server.py        # Main API server
├── config.json          # Your configuration (git-ignored)
├── config.example.json  # Template configuration
├── requirements.txt     # Python dependencies
├── test_generate.py     # Test script
└── gemini_webapi/       # Gemini client library
    ├── client.py
    ├── constants.py
    ├── exceptions.py
    ├── types/
    └── utils/
```