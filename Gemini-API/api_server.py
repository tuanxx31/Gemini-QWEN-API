"""
OpenAI-Compatible Gemini Image Generation API Server

Endpoints:
- POST /v1/images/generations  (text-to-image)
- POST /v1/images/edits        (image-to-image)

Usage:
    python api_server.py

API Format matches OpenAI's image generation API.
"""

import asyncio
import base64
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import Gemini client
from gemini_webapi import GeminiClient

# Load config
# CONFIG_PATH = Path(__file__).parent / "config.json"

def load_config() -> Dict:
    """Load configuration from environment variables with fallback to config.json"""
    config = {}
    
    # Try to load from config.json first (for backward compatibility)
    # if CONFIG_PATH.exists():
    #     try:
    #         with open(CONFIG_PATH, "r") as f:
    #             config = json.load(f)
    #     except Exception as e:
    #         print(f"[WARNING] Failed to load config.json: {e}")
    
    # Override with environment variables if they exist
    config["api_key"] = os.getenv("GEMINI_API_KEY", config.get("api_key", "sk-demo"))
    config["cookie_1PSID"] = os.getenv("GEMINI_COOKIE_1PSID", config.get("cookie_1PSID", ""))
    config["cookie_1PSIDTS"] = os.getenv("GEMINI_COOKIE_1PSIDTS", config.get("cookie_1PSIDTS", ""))
    config["host"] = os.getenv("GEMINI_HOST", config.get("host", "0.0.0.0"))
    config["port"] = int(os.getenv("GEMINI_PORT", str(config.get("port", 8000))))
    config["default_model"] = os.getenv("GEMINI_DEFAULT_MODEL", config.get("default_model", "gemini-2.5-flash"))
    config["timeout"] = int(os.getenv("GEMINI_TIMEOUT", str(config.get("timeout", 120))))
    
    return config

CONFIG = load_config()
print(CONFIG)
# Global client instance
gemini_client: Optional[GeminiClient] = None


# ============ Request/Response Models ============

class ImageGenerateRequest(BaseModel):
    model: str = Field(default="gemini-2.5-flash", description="Model to use")
    prompt: str = Field(..., description="Text prompt for image generation")
    n: int = Field(default=1, ge=1, le=4, description="Number of images (1-4)")
    size: Optional[str] = Field(
        default=None,
        description="Aspect ratio: 'landscape' or '16:9' | 'portrait' or '9:16' | 'square' or '1:1'"
    )
    seed: Optional[int] = Field(default=None, description="Seed for generation")
    response_format: str = Field(default="b64_json", description="Response format: b64_json or url")


class ImageData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageResponse(BaseModel):
    created: int
    data: List[ImageData]


class ErrorResponse(BaseModel):
    error: dict


# ============ Gemini Client Management ============

async def get_gemini_client() -> GeminiClient:
    global gemini_client

    if gemini_client is None or not gemini_client._running:
        cookie_1psid = CONFIG.get("cookie_1PSID", "")
        cookie_1psidts = CONFIG.get("cookie_1PSIDTS", "")

        if not cookie_1psid or cookie_1psid == "YOUR_SECURE_1PSID_HERE":
            raise HTTPException(
                status_code=500,
                detail="cookie_1PSID not configured. Please update config.json"
            )

        gemini_client = GeminiClient(
            secure_1psid=cookie_1psid,
            secure_1psidts=cookie_1psidts if cookie_1psidts and cookie_1psidts != "YOUR_SECURE_1PSIDTS_HERE" else None
        )

        await gemini_client.init(
            timeout=CONFIG.get("timeout", 120),
            auto_close=False,
            auto_refresh=True
        )
        print("[API] Gemini client initialized successfully")

    return gemini_client


def verify_auth(authorization: Optional[str]) -> bool:
    """Verify Bearer token authorization."""
    if not authorization:
        return False

    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False

    token = parts[1]
    expected_token = CONFIG.get("api_key", "sk-demo")

    return token == expected_token


async def download_image_as_base64(image_obj, cookies: dict) -> tuple[str, str]:
    """Download image and convert to base64."""
    fd, temp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)

    try:
        # Save image to temp file
        await image_obj.save(
            path=os.path.dirname(temp_path),
            filename=os.path.basename(temp_path),
            cookies=cookies
        )

        # Read and convert to base64
        with open(temp_path, "rb") as f:
            image_bytes = f.read()

        b64_string = base64.b64encode(image_bytes).decode("utf-8")
        return b64_string, image_obj.url

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def get_size_prompt(size: Optional[str]) -> str:
    """Convert size parameter to prompt modifier for aspect ratio control."""
    if not size:
        return ""

    size_lower = size.lower()

    # Landscape 16:9
    if "landscape" in size_lower or "1792x1024" in size_lower or "16:9" in size_lower:
        return ", wide cinematic landscape aspect ratio 16:9"

    # Portrait 9:16
    elif "portrait" in size_lower or "1024x1792" in size_lower or "9:16" in size_lower:
        return ", tall vertical portrait aspect ratio 9:16"

    # Square 1:1
    elif "square" in size_lower or "1024x1024" in size_lower or "1:1" in size_lower:
        return ", square aspect ratio 1:1"

    return ""


# ============ Lifespan ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"[API] Starting Gemini Image API Server on {CONFIG.get('host', '0.0.0.0')}:{CONFIG.get('port', 8000)}")
    yield
    # Shutdown
    global gemini_client
    if gemini_client and gemini_client._running:
        await gemini_client.close()
        print("[API] Gemini client closed")


# ============ FastAPI App ============

app = FastAPI(
    title="Gemini Image Generation API",
    description="OpenAI-compatible API for Google Gemini image generation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Endpoints ============

@app.get("/")
async def root():
    return {
        "message": "Gemini Image Generation API",
        "endpoints": {
            "models": "GET /models",
            "generate": "POST /v1/images/generations",
            "edit": "POST /v1/images/edits"
        }
    }


# Models data
MODELS_DATA = [
    {
        "id": "gemini-2.5-flash",
        "object": "model",
        "created": 1700000000,
        "owned_by": "google",
        "description": "Fast model optimized for speed",
        "capabilities": ["text-to-image", "image-to-image", "chat"]
    },
    {
        "id": "gemini-2.5-pro",
        "object": "model",
        "created": 1700000000,
        "owned_by": "google",
        "description": "Pro model with enhanced quality",
        "capabilities": ["text-to-image", "image-to-image", "chat"]
    },
    {
        "id": "gemini-3.0-pro",
        "object": "model",
        "created": 1700000000,
        "owned_by": "google",
        "description": "Latest Pro model with best quality",
        "capabilities": ["text-to-image", "image-to-image", "chat"]
    },
]


@app.get("/models")
async def get_models():
    """List all available AI models (no auth required)."""
    return {
        "object": "list",
        "data": MODELS_DATA
    }


@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get details of a specific model."""
    for model in MODELS_DATA:
        if model["id"] == model_id:
            return model
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


@app.get("/v1/models")
async def list_models_v1(authorization: Optional[str] = Header(None)):
    """List available models (OpenAI compatible, requires auth)."""
    if not verify_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid authorization token")

    return {
        "object": "list",
        "data": MODELS_DATA
    }


@app.post("/v1/images/generations", response_model=ImageResponse)
async def generate_image(
    request: ImageGenerateRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Generate images from text prompt.

    Compatible with OpenAI's /v1/images/generations endpoint.
    """
    # Verify authorization
    if not verify_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid authorization token")

    try:
        client = await get_gemini_client()

        # Build prompt with size modifier
        full_prompt = request.prompt + get_size_prompt(request.size)

        # Add seed to prompt if provided
        if request.seed:
            full_prompt += f" (seed: {request.seed})"

        print(f"[API] Generating image with prompt: {full_prompt[:100]}...")

        # Call Gemini API
        response = await client.generate_content(
            prompt=full_prompt,
            model=request.model,
            image_mode=True,
            timeout=CONFIG.get("timeout", 120)
        )

        if not response.images:
            raise HTTPException(
                status_code=500,
                detail=f"No images generated. Response: {response.text[:200] if response.text else 'No text'}"
            )

        # Get revised prompt from response
        revised_prompt = response.text or request.prompt

        # Download and convert images to base64
        image_data_list = []
        images_to_process = response.images[:request.n]  # Limit to requested number

        for img in images_to_process:
            try:
                b64_string, url = await download_image_as_base64(img, client.cookies)
                image_data_list.append(ImageData(
                    b64_json=b64_string if request.response_format == "b64_json" else None,
                    url=url if request.response_format == "url" else None,
                    revised_prompt=revised_prompt
                ))
            except Exception as e:
                print(f"[API] Error downloading image: {e}")
                continue

        if not image_data_list:
            raise HTTPException(status_code=500, detail="Failed to download generated images")

        print(f"[API] Successfully generated {len(image_data_list)} image(s)")

        return ImageResponse(
            created=int(time.time()),
            data=image_data_list
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/images/edits", response_model=ImageResponse)
async def edit_image(
    image: UploadFile = File(..., description="Image to edit"),
    prompt: str = Form(..., description="Edit instruction"),
    model: str = Form(default="gemini-2.5-flash"),
    n: int = Form(default=1, ge=1, le=4),
    size: Optional[str] = Form(default=None),
    authorization: Optional[str] = Header(None)
):
    """
    Edit an image with a text prompt.

    Compatible with OpenAI's /v1/images/edits endpoint.
    """
    # Verify authorization
    if not verify_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid authorization token")

    temp_path = None

    try:
        client = await get_gemini_client()

        # Save uploaded image to temp file
        fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)

        content = await image.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Build prompt with size modifier
        full_prompt = prompt + get_size_prompt(size)

        print(f"[API] Editing image with prompt: {full_prompt[:100]}...")

        # Call Gemini API with image
        response = await client.generate_content(
            prompt=full_prompt,
            files=[temp_path],
            model=model,
            image_mode=True,
            timeout=CONFIG.get("timeout", 120)
        )

        if not response.images:
            raise HTTPException(
                status_code=500,
                detail=f"No images generated. Response: {response.text[:200] if response.text else 'No text'}"
            )

        # Get revised prompt from response
        revised_prompt = response.text or prompt

        # Download and convert images to base64
        image_data_list = []
        images_to_process = response.images[:n]

        for img in images_to_process:
            try:
                b64_string, url = await download_image_as_base64(img, client.cookies)
                image_data_list.append(ImageData(
                    b64_json=b64_string,
                    revised_prompt=revised_prompt
                ))
            except Exception as e:
                print(f"[API] Error downloading image: {e}")
                continue

        if not image_data_list:
            raise HTTPException(status_code=500, detail="Failed to download generated images")

        print(f"[API] Successfully edited image, got {len(image_data_list)} result(s)")

        return ImageResponse(
            created=int(time.time()),
            data=image_data_list
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global gemini_client
    return {
        "status": "healthy",
        "gemini_client_active": gemini_client is not None and gemini_client._running
    }


# ============ Main ============

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=CONFIG.get("host", "0.0.0.0"),
        port=CONFIG.get("port", 8000),
        reload=False
    )
