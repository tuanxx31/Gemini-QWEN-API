"""
OpenAI-Compatible Qwen Image Generation API Server (GWEN-API)

Features:
- Multi-account token rotation (auto-switch when rate limited)
- POST /v1/images/generations  (text-to-image)
- POST /v1/images/edits        (image-to-image)

Usage:
    python GWEN-API.py

API Format matches OpenAI's image generation API.
"""

import asyncio
import base64
import json
import os
import tempfile
import time
from pathlib import Path
from loguru import logger
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Header, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import Qwen client
from qwen_client import QwenClient


# Load config
CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()


# ============ Account Manager with Token Rotation ============


@dataclass
class Account:
    """Represents a Qwen account."""

    name: str
    token: str
    cnaui: str
    aui: str
    bx_ua: str = ""
    bx_umidtoken: str = ""
    extra_cookies: Dict[str, str] = None
    client: Optional[QwenClient] = None
    is_rate_limited: bool = False
    rate_limit_until: float = 0

    def __post_init__(self):
        if self.extra_cookies is None:
            self.extra_cookies = {}


class AccountManager:
    """Manages multiple Qwen accounts with automatic rotation."""

    def __init__(self, config: dict):
        self.accounts: List[Account] = []
        self.current_index = 0
        self.verbose = config.get("verbose", True)
        self.timeout = config.get("timeout", 120)

        # Load accounts from config
        self._load_accounts(config)

        logger.info(f"AccountManager initialized with {len(self.accounts)} account(s)")

    def _load_accounts(self, config: dict):
        """Load accounts from config - supports both old and new format."""

        # New format: accounts array
        if "accounts" in config:
            for acc_data in config["accounts"]:
                account = Account(
                    name=acc_data.get("name", f"Account {len(self.accounts) + 1}"),
                    token=acc_data.get("cookie_token", ""),
                    cnaui=acc_data.get("cookie_cnaui", ""),
                    aui=acc_data.get("cookie_aui", ""),
                    bx_ua=acc_data.get("cookie_bx_ua", ""),
                    bx_umidtoken=acc_data.get("cookie_bx_umidtoken", ""),
                )
                if account.token:
                    self.accounts.append(account)
                    logger.debug(f"Loaded account: {account.name}")

        # Old format: single account with cookie_* keys
        else:
            cookies = {
                k.replace("cookie_", ""): v
                for k, v in config.items()
                if k.startswith("cookie_")
            }

            token = cookies.pop("token", "")
            if token:
                account = Account(
                    name="Default Account",
                    token=token,
                    cnaui=cookies.pop("cnaui", ""),
                    aui=cookies.pop("aui", ""),
                    bx_ua=cookies.pop("bx_ua", ""),
                    bx_umidtoken=cookies.pop("bx_umidtoken", ""),
                    extra_cookies=cookies,
                )
                self.accounts.append(account)
                logger.debug(f"Loaded legacy account format")

        if not self.accounts:
            raise ValueError("No valid accounts configured. Please update config.json")

    def _get_available_account(self) -> Optional[Account]:
        """Get the next available account (not rate limited)."""
        current_time = time.time()

        # Try all accounts starting from current index
        for i in range(len(self.accounts)):
            idx = (self.current_index + i) % len(self.accounts)
            account = self.accounts[idx]

            # Check if rate limit has expired
            if account.is_rate_limited and current_time >= account.rate_limit_until:
                account.is_rate_limited = False
                logger.info(
                    f"Account '{account.name}' rate limit expired, now available"
                )

            if not account.is_rate_limited:
                self.current_index = idx
                return account

        return None

    async def get_client(self) -> tuple[QwenClient, Account]:
        """Get an available client, rotating if necessary."""
        account = self._get_available_account()

        if account is None:
            # All accounts are rate limited
            earliest_available = min(acc.rate_limit_until for acc in self.accounts)
            wait_time = earliest_available - time.time()
            raise HTTPException(
                status_code=429,
                detail=f"All accounts rate limited. Try again in {int(wait_time)}s",
            )

        # Initialize client if needed
        if account.client is None or not account.client._running:
            logger.info(f"Initializing client for account: {account.name}")

            account.client = QwenClient(
                token=account.token,
                cnaui=account.cnaui,
                aui=account.aui,
                bx_ua=account.bx_ua,
                bx_umidtoken=account.bx_umidtoken,
                extra_cookies=account.extra_cookies,
                verbose=self.verbose,
                silent=not self.verbose,  # Silent when not verbose
            )

            await account.client.init(timeout=self.timeout)
            logger.success(f"Client initialized for account: {account.name}")

        # Round-robin: rotate to next account immediately for next request
        self.current_index = (self.current_index + 1) % len(self.accounts)

        return account.client, account

    def mark_rate_limited(
        self, account: Account, duration: int = 54000
    ):  # Default 15 hours
        """Mark an account as rate limited."""
        account.is_rate_limited = True
        account.rate_limit_until = time.time() + duration

        # Format duration for logging
        if duration >= 3600:
            duration_str = f"{duration // 3600}h {(duration % 3600) // 60}m"
        elif duration >= 60:
            duration_str = f"{duration // 60}m"
        else:
            duration_str = f"{duration}s"

        logger.warning(
            f"Account '{account.name}' marked as rate limited for {duration_str}"
        )

        # Rotate to next account
        self.current_index = (self.current_index + 1) % len(self.accounts)

        next_account = self._get_available_account()
        if next_account:
            logger.info(f"Switched to account: {next_account.name}")

    async def close_all(self):
        """Close all client sessions."""
        for account in self.accounts:
            if account.client and account.client._running:
                await account.client.close()
                logger.debug(f"Closed client for account: {account.name}")

    def rotate_next(self):
        """Rotate to next account for round-robin load balancing."""
        old_index = self.current_index
        self.current_index = (self.current_index + 1) % len(self.accounts)
        if len(self.accounts) > 1:
            logger.debug(
                f"Rotated from account index {old_index} to {self.current_index}"
            )

    def get_status(self) -> List[dict]:
        """Get status of all accounts."""
        current_time = time.time()
        return [
            {
                "name": acc.name,
                "active": acc.client is not None and acc.client._running,
                "rate_limited": acc.is_rate_limited,
                "rate_limit_remaining": (
                    max(0, int(acc.rate_limit_until - current_time))
                    if acc.is_rate_limited
                    else 0
                ),
            }
            for acc in self.accounts
        ]


# Global account manager
account_manager: Optional[AccountManager] = None


# ============ Request/Response Models ============


class ImageGenerateRequest(BaseModel):
    model: str = Field(default="qwen3-max", description="Model to use")
    prompt: str = Field(..., description="Text prompt for image generation")
    n: int = Field(default=1, ge=1, le=4, description="Number of images (1-4)")
    size: Optional[str] = Field(
        default=None,
        description="Aspect ratio: 'landscape' or '16:9' | 'portrait' or '9:16' | 'square' or '1:1'",
    )
    seed: Optional[int] = Field(default=None, description="Seed for generation")
    response_format: str = Field(
        default="b64_json", description="Response format: b64_json or url"
    )


class ImageData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageResponse(BaseModel):
    created: int
    data: List[ImageData]


class ErrorResponse(BaseModel):
    error: dict


# ============ Helper Functions ============


def get_valid_api_keys() -> set:
    """Get all valid API keys from config."""
    valid_keys = set()

    # Add master key (api_key)
    master_key = CONFIG.get("api_key", "")
    if master_key:
        valid_keys.add(master_key)

    # Add additional keys from api_keys list
    additional_keys = CONFIG.get("api_keys", [])
    if isinstance(additional_keys, list):
        for key in additional_keys:
            if isinstance(key, str) and key:
                valid_keys.add(key)
            elif isinstance(key, dict) and key.get("key"):
                valid_keys.add(key["key"])

    return valid_keys


def verify_auth(authorization: Optional[str]) -> bool:
    """Verify Bearer token authorization."""
    valid_keys = get_valid_api_keys()

    # If no keys configured, allow all requests
    if not valid_keys:
        return True

    # If sk-demo is in valid keys, allow all requests (demo mode)
    if "sk-demo" in valid_keys:
        return True

    if not authorization:
        return False

    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False

    token = parts[1]
    return token in valid_keys


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


def normalize_model(model: str) -> str:
    """Normalize model name to Qwen format."""
    model_lower = model.lower()

    # qwen-max-latest (newest model)
    if "qwen-max-latest" in model_lower or "max-latest" in model_lower:
        return "qwen-max-latest"
    # qwen3-max
    elif "qwen3-max" in model_lower or "qwen-3" in model_lower:
        return "qwen3-max-2025-09-23"
    # qwen2.5
    elif "qwen2.5" in model_lower or "qwen-2.5" in model_lower:
        return "qwen2.5-vl-72b-instruct"

    # Default to qwen-max-latest
    return "qwen-max-latest"


def is_rate_limit_error(error: Exception) -> bool:
    """Check if the error is a rate limit error or token expired."""
    error_str = str(error).lower()

    # Common rate limit keywords
    rate_limit_keywords = [
        "rate limit",
        "too many requests",
        "429",
        "quota",
        "limit exceeded",
        "request limit",
        "frequency",
        "throttl",
        "busy",
        "overload",
        "try again later",
        "slow down",
        "exceeded",
        "maximum",
        "too fast",
        "wait",
        "cooldown",
        "blocked",
        "banned",
        "daily usage limit",
        "usage limit",
        "wait 10 hours",  # Qwen specific
        "limited",
        "daily limit",
        "reached the",
        "限制",
        "频繁",
        "请稍后",  # Chinese keywords
        # Token expired/invalid keywords
        "token expired",
        "token invalid",
        "invalid token",
        "expired",
        "unauthorized",
        "401",
        "authentication",
        "login",
        "session",
        "cookie",
        "invalid credentials",
        "not logged in",
        "登录",
        "过期",
        "失效",
        "认证",
    ]

    is_limited = any(keyword in error_str for keyword in rate_limit_keywords)

    if is_limited:
        logger.warning(f"Rate limit detected in error: {error_str[:100]}...")

    return is_limited


# ============ Lifespan ============


@asynccontextmanager
async def lifespan(app: FastAPI):
    global account_manager

    # Startup
    logger.info(
        f"Starting GWEN Image API Server on {CONFIG.get('host', '0.0.0.0')}:{CONFIG.get('port', 8001)}"
    )
    account_manager = AccountManager(CONFIG)

    yield

    # Shutdown
    if account_manager:
        await account_manager.close_all()
        logger.info("All clients closed")


# ============ FastAPI App ============

app = FastAPI(
    title="GWEN-API - Qwen Image Generation API",
    description="OpenAI-compatible API for Qwen image generation with multi-account support",
    version="2.0.0",
    lifespan=lifespan,
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
        "message": "GWEN-API - Qwen Image Generation API",
        "version": "2.0.0",
        "features": ["multi-account", "token-rotation"],
        "endpoints": {
            "models": "GET /models",
            "generate": "POST /v1/images/generations",
            "edit": "POST /v1/images/edits",
            "accounts": "GET /accounts",
        },
    }


# Models data
MODELS_DATA = [
    {
        "id": "qwen-max-latest",
        "object": "model",
        "created": 1700000000,
        "owned_by": "alibaba",
        "description": "Qwen Max Latest - newest model for image generation and editing",
        "capabilities": ["text-to-image", "image-to-image", "chat"],
    },
    {
        "id": "qwen3-max",
        "object": "model",
        "created": 1700000000,
        "owned_by": "alibaba",
        "description": "Qwen Max Latest - newest model for image generation and editing",
        "capabilities": ["text-to-image", "image-to-image", "chat"],
    },
]


@app.get("/models")
async def get_models():
    """List all available AI models (no auth required)."""
    return {"object": "list", "data": MODELS_DATA}


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

    return {"object": "list", "data": MODELS_DATA}


@app.get("/accounts")
async def get_accounts_status():
    """Get status of all configured accounts."""
    if account_manager is None:
        raise HTTPException(status_code=500, detail="Account manager not initialized")

    return {
        "total_accounts": len(account_manager.accounts),
        "accounts": account_manager.get_status(),
    }


@app.post("/v1/images/generations", response_model=ImageResponse)
async def generate_image(
    request: ImageGenerateRequest, authorization: Optional[str] = Header(None)
):
    """
    Generate images from text prompt.
    Compatible with OpenAI's /v1/images/generations endpoint.
    Supports automatic account rotation on rate limit.
    """
    # Verify authorization
    if not verify_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid authorization token")

    max_retries = len(account_manager.accounts) if account_manager else 1
    last_error = None

    for attempt in range(max_retries):
        try:
            client, account = await account_manager.get_client()

            # Build prompt with size modifier
            full_prompt = request.prompt + get_size_prompt(request.size)

            # Add seed to prompt if provided
            if request.seed:
                full_prompt += f" (seed: {request.seed})"

            # Normalize model name
            model = normalize_model(request.model)

            logger.info(
                f"[{account.name}] Generating image with prompt: {full_prompt[:100]}..."
            )

            # Call Qwen API
            response = await client.generate_image(
                prompt=full_prompt, model=model, timeout=CONFIG.get("timeout", 120)
            )

            if not response.images:
                # Check if response indicates rate limit or token issue
                response_text = response.text or ""
                logger.warning(
                    f"[{account.name}] No images returned. Response text: {response_text[:300]}..."
                )

                if is_rate_limit_error(Exception(response_text)):
                    logger.warning(
                        f"[{account.name}] Rate limit/token issue detected in response, trying next account..."
                    )
                    # Rate limit: 15 hours
                    account_manager.mark_rate_limited(account, duration=54000)
                else:
                    logger.warning(
                        f"[{account.name}] No images generated, trying next account..."
                    )
                    # Other issues: 60 seconds
                    account_manager.mark_rate_limited(account, duration=60)

                last_error = Exception(
                    f"No images generated. Response: {response_text[:200] if response_text else 'No text'}"
                )
                continue  # Try next account

            # Get revised prompt from response
            revised_prompt = response.text or request.prompt

            # Download and convert images to base64
            image_data_list = []
            images_to_process = response.images[: request.n]

            for img in images_to_process:
                try:
                    if request.response_format == "b64_json":
                        b64_string = await img.to_base64(client._session)
                        image_data_list.append(
                            ImageData(
                                b64_json=b64_string, revised_prompt=revised_prompt
                            )
                        )
                    else:
                        image_data_list.append(
                            ImageData(url=img.url, revised_prompt=revised_prompt)
                        )
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    continue

            if not image_data_list:
                raise HTTPException(
                    status_code=500, detail="Failed to process generated images"
                )

            logger.success(
                f"[{account.name}] Successfully generated {len(image_data_list)} image(s)"
            )

            return ImageResponse(created=int(time.time()), data=image_data_list)

        except HTTPException as e:
            if e.status_code == 429:
                # Rate limit from our check - all accounts exhausted
                raise
            last_error = e
            logger.warning(
                f"[{account.name}] HTTPException: {e.detail}, trying next account..."
            )
            account_manager.mark_rate_limited(
                account, duration=60
            )  # Short duration for HTTP errors
            continue  # Try next account

        except Exception as e:
            last_error = e
            error_message = str(e)

            # Log chi tiết lỗi
            logger.error(f"[{account.name}] Exception type: {type(e).__name__}")
            logger.error(f"[{account.name}] Error message: {error_message}")

            # Mark account as problematic and try next
            if is_rate_limit_error(e):
                logger.warning(f"[{account.name}] Rate limited, trying next account...")
                account_manager.mark_rate_limited(account, duration=54000)  # 15 hours
            else:
                logger.warning(
                    f"[{account.name}] Error occurred, trying next account..."
                )
                account_manager.mark_rate_limited(account, duration=60)  # 60 seconds

            continue  # Always try next account

    # All retries failed
    raise HTTPException(status_code=500, detail=f"All accounts failed: {last_error}")


@app.post("/v1/images/edits", response_model=ImageResponse)
async def edit_image(
    image: UploadFile = File(..., description="Image to edit"),
    prompt: str = Form(..., description="Edit instruction"),
    model: str = Form(default="qwen3-max"),
    n: int = Form(default=1, ge=1, le=4),
    size: Optional[str] = Form(default=None),
    authorization: Optional[str] = Header(None),
):
    """
    Edit an image with a text prompt.
    Compatible with OpenAI's /v1/images/edits endpoint.
    Supports automatic account rotation on rate limit.
    """
    # Verify authorization
    if not verify_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid authorization token")

    temp_path = None
    max_retries = len(account_manager.accounts) if account_manager else 1
    last_error = None

    try:
        # Save uploaded image to temp file
        fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)

        content = await image.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        for attempt in range(max_retries):
            try:
                client, account = await account_manager.get_client()

                # Build prompt with size modifier
                full_prompt = prompt + get_size_prompt(size)

                # Normalize model name
                qwen_model = normalize_model(model)

                logger.info(
                    f"[{account.name}] Editing image with prompt: {full_prompt[:100]}..."
                )

                # Call Qwen API with image
                response = await client.generate_image(
                    prompt=full_prompt,
                    model=qwen_model,
                    image_path=temp_path,
                    timeout=CONFIG.get("timeout", 120),
                )

                if not response.images:
                    # Check if response indicates rate limit
                    response_text = response.text or ""
                    logger.warning(
                        f"[{account.name}] No images returned. Response text: {response_text[:300]}..."
                    )

                    if is_rate_limit_error(Exception(response_text)):
                        logger.warning(
                            f"[{account.name}] Rate limit detected in response, trying next account..."
                        )
                        # Rate limit: 15 hours
                        account_manager.mark_rate_limited(account, duration=54000)
                    else:
                        logger.warning(
                            f"[{account.name}] No images generated, trying next account..."
                        )
                        # Other issues: 60 seconds
                        account_manager.mark_rate_limited(account, duration=60)

                    last_error = Exception(
                        f"No images generated. Response: {response_text[:200] if response_text else 'No text'}"
                    )
                    continue  # Try next account

                # Get revised prompt from response
                revised_prompt = response.text or prompt

                # Download and convert images to base64
                image_data_list = []
                images_to_process = response.images[:n]

                for img in images_to_process:
                    try:
                        b64_string = await img.to_base64(client._session)
                        image_data_list.append(
                            ImageData(
                                b64_json=b64_string, revised_prompt=revised_prompt
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error downloading image: {e}")
                        continue

                if not image_data_list:
                    raise HTTPException(
                        status_code=500, detail="Failed to process generated images"
                    )

                logger.success(
                    f"[{account.name}] Successfully edited image, got {len(image_data_list)} result(s)"
                )

                return ImageResponse(created=int(time.time()), data=image_data_list)

            except HTTPException as e:
                if e.status_code == 429:
                    # Rate limit from our check - all accounts exhausted
                    raise
                last_error = e
                logger.warning(
                    f"[{account.name}] HTTPException: {e.detail}, trying next account..."
                )
                account_manager.mark_rate_limited(
                    account, duration=60
                )  # Short duration for HTTP errors
                continue  # Try next account

            except Exception as e:
                last_error = e
                error_message = str(e)

                # Log chi tiết lỗi
                logger.error(f"[{account.name}] Exception type: {type(e).__name__}")
                logger.error(f"[{account.name}] Error message: {error_message}")

                # Mark account as problematic and try next
                if is_rate_limit_error(e):
                    logger.warning(
                        f"[{account.name}] Rate limited, trying next account..."
                    )
                    account_manager.mark_rate_limited(
                        account, duration=54000
                    )  # 15 hours
                else:
                    logger.warning(
                        f"[{account.name}] Error occurred, trying next account..."
                    )
                    account_manager.mark_rate_limited(
                        account, duration=60
                    )  # 60 seconds

                continue  # Always try next account

        # All retries failed
        raise HTTPException(
            status_code=500, detail=f"All accounts failed: {last_error}"
        )

    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "accounts": account_manager.get_status() if account_manager else [],
    }


# ============ Main ============

if __name__ == "__main__":
    uvicorn.run(
        "GWEN-API:app",
        host=CONFIG.get("host", "0.0.0.0"),
        port=CONFIG.get("port", 8001),
        reload=False,
    )
