
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
import aiohttp

# Import Seedream client
from seedream_client import SeedreamClient


# Load config
CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config():
    if not CONFIG_PATH.exists():
        # Create default config if not exists
        default_config = {
            "host": "0.0.0.0",
            "port": 8002,
            "api_key": "sk-demo",
            "verbose": True,
            "timeout": 120,
            "accounts": []
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        return default_config
        
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


CONFIG = load_config()


# ============ Account Manager with Rotation ============


@dataclass
class Account:
    """Represents a Seedream account."""
    name: str
    cookie: str
    csrf_token: str
    web_id: str
    endpoint_id: str = "seedream-4-5-251128"
    model: str = "seedream-4-5"
    client: Optional[SeedreamClient] = None
    is_rate_limited: bool = False
    rate_limit_until: float = 0


class AccountManager:
    """Manages multiple Seedream accounts with automatic rotation."""

    def __init__(self, config: dict):
        self.accounts: List[Account] = []
        self.current_index = 0
        self.verbose = config.get("verbose", True)
        self.timeout = config.get("timeout", 120)

        # Load accounts from config
        self._load_accounts(config)

        print("\n" + "="*50)
        logger.info(f" SEEDREAM Account Manager v1.0.0")
        logger.info(f" Loaded {len(self.accounts)} account(s)")
        print("="*50 + "\n")

    def _load_accounts(self, config: dict):
        """Load accounts from config."""
        if "accounts" in config:
            for acc_data in config["accounts"]:
                account = Account(
                    name=acc_data.get("name", f"Account {len(self.accounts) + 1}"),
                    cookie=acc_data.get("cookie", ""),
                    csrf_token=acc_data.get("csrf_token", ""),
                    web_id=acc_data.get("web_id", ""),
                    endpoint_id=acc_data.get("endpoint_id", "seedream-4-5-251128"),
                    model=acc_data.get("model", "seedream-4-5")
                )
                if account.cookie:
                    self.accounts.append(account)
                    logger.debug(f"Loaded account: {account.name}")

        if not self.accounts:
            logger.warning("No accounts configured. Please add tokens to config.json")

    def _get_available_account(self) -> Optional[Account]:
        """Get the next available account (not rate limited)."""
        current_time = time.time()

        for i in range(len(self.accounts)):
            idx = (self.current_index + i) % len(self.accounts)
            account = self.accounts[idx]

            if account.is_rate_limited and current_time >= account.rate_limit_until:
                account.is_rate_limited = False
                logger.info(f"Account '{account.name}' rate limit expired")

            if not account.is_rate_limited:
                self.current_index = idx
                return account

        return None

    async def get_client(self) -> tuple[SeedreamClient, Account]:
        """Get an available client, rotating if necessary."""
        account = self._get_available_account()

        if account is None:
            if not self.accounts:
                raise HTTPException(status_code=500, detail="No accounts configured")
            
            earliest_available = min(acc.rate_limit_until for acc in self.accounts)
            wait_time = earliest_available - time.time()
            raise HTTPException(
                status_code=429,
                detail=f"All accounts rate limited. Try again in {int(wait_time)}s",
            )

        if account.client is None or not account.client._running:
            logger.info(f"Initializing client for account: {account.name}")
            account.client = SeedreamClient(
                cookie=account.cookie,
                csrf_token=account.csrf_token,
                web_id=account.web_id,
                verbose=self.verbose
            )
            await account.client.init()
            logger.success(f"Client initialized for account: {account.name}")

        self.current_index = (self.current_index + 1) % len(self.accounts)
        return account.client, account

    def mark_rate_limited(self, account: Account, duration: int = 3600):
        """Mark an account as rate limited."""
        account.is_rate_limited = True
        account.rate_limit_until = time.time() + duration
        
        if duration >= 3600:
            duration_str = f"{duration // 3600}h {(duration % 3600) // 60}m"
        elif duration >= 60:
            duration_str = f"{duration // 60}m"
        else:
            duration_str = f"{duration}s"
            
        logger.warning(f"Account '{account.name}' rate limited for {duration_str}")
        self.current_index = (self.current_index + 1) % len(self.accounts)

    async def close_all(self):
        """Close all client sessions."""
        for account in self.accounts:
            if account.client:
                await account.client.close()

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
    model: str = Field(default="seedream-4-5", description="Model to use")
    prompt: str = Field(..., description="Text prompt for image generation")
    n: int = Field(default=1, ge=1, le=4)
    size: Optional[str] = Field(default="2048x2048")
    response_format: str = Field(default="url")


class ImageData(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None


class ImageResponse(BaseModel):
    created: int
    data: List[ImageData]


# ============ Helper Functions ============

def get_valid_api_keys() -> set:
    """Get all valid API keys from config."""
    valid_keys = set()
    master_key = CONFIG.get("api_key", "")
    if master_key: valid_keys.add(master_key)
    
    additional_keys = CONFIG.get("api_keys", [])
    if isinstance(additional_keys, list):
        for key in additional_keys:
            if isinstance(key, str) and key: valid_keys.add(key)
            elif isinstance(key, dict) and key.get("key"): valid_keys.add(key["key"])
    return valid_keys

def verify_auth(authorization: Optional[str]) -> bool:
    """Verify Bearer token authorization."""
    valid_keys = get_valid_api_keys()
    if not valid_keys or "sk-demo" in valid_keys: return True
    if not authorization: return False
    
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer": return False
    return parts[1] in valid_keys

def is_rate_limit_error(error: Exception) -> bool:
    """Check if the error is a rate limit error or token expired."""
    err_str = str(error).lower()
    keywords = ["rate limit", "429", "quota", "frequency", "too many requests", "invalid token", "expired", "401", "unauthorized"]
    return any(k in err_str for k in keywords)

async def download_image_b64(url: str, session: aiohttp.ClientSession) -> str:
    async with session.get(url) as resp:
        if resp.status == 200:
            data = await resp.read()
            return base64.b64encode(data).decode("utf-8")
    return ""

def parse_size(size_str: str) -> tuple[int, int]:
    """
    Parse size string and ensure it meets Seedream 4.5 minimum requirements.
    Seedream 4.5 requires >= 3,686,400 pixels.
    """
    # Standard Presets
    presets = {
        "2k": (2048, 2048),
        "4k": (4096, 4096), # Default 1:1 4K
    }
    
    if size_str.lower() in presets:
        return presets[size_str.lower()]

    try:
        if "x" in size_str:
            w, h = map(int, size_str.split("x"))
            
            # If total pixels < 3.68M, upscale maintaining EXACT ratio
            min_pixels = 3686400
            current_pixels = w * h
            if current_pixels < min_pixels:
                ratio = w / h
                # Solve for h: (h * ratio) * h = min_pixels => h^2 = min_pixels / ratio
                new_h = (min_pixels / ratio) ** 0.5
                new_w = new_h * ratio
                
                # Round to multiples of 8 (common requirements) while keeping ratio close
                w = int(round(new_w / 8) * 8)
                h = int(round(new_h / 8) * 8)
                
                # Double check we didn't fall below min_pixels after rounding
                if w * h < min_pixels:
                    if ratio >= 1: w += 8
                    else: h += 8
                
                logger.warning(f"Size too small for Seedream 4.5, upscaling to {w}x{h} (Original {size_str})")
            
            return w, h
    except:
        pass
    
    # Default to 2K (Safe for Seedream 4.5)
    return 2048, 2048


# ============ Lifespan ============


@asynccontextmanager
async def lifespan(app: FastAPI):
    global account_manager

    # Startup
    logger.info(f"Starting SEEDREAM Image API Server on {CONFIG.get('host', '0.0.0.0')}:{CONFIG.get('port', 8002)}")
    account_manager = AccountManager(CONFIG)

    yield

    # Shutdown
    if account_manager:
        await account_manager.close_all()
        logger.info("All clients closed")


# ============ FastAPI App ============

app = FastAPI(
    title="SEEDREAM-API",
    description="OpenAI-compatible API for BytePlus Seedream 4.5",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "SEEDREAM-API is running",
        "version": "1.0.0",
        "endpoints": {
            "models": "GET /models",
            "accounts": "GET /accounts",
            "generate": "POST /v1/images/generations",
            "edit": "POST /v1/images/edits"
        }
    }

@app.get("/models")
async def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "seedream-4-5",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "byteplus",
                "capabilities": ["text-to-image", "image-to-image"]
            }
        ]
    }

@app.get("/accounts")
async def get_accounts_status():
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
    Supports automatic account rotation on rate limit.
    """
    if not verify_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid API Key")

    max_retries = len(account_manager.accounts) if account_manager else 1
    last_error = None

    for attempt in range(max_retries):
        try:
            client, account = await account_manager.get_client()
            w, h = parse_size(request.size)
            
            logger.info(f"[{account.name}] Generating image with prompt: {request.prompt[:100]}...")
            response = await client.generate_image(
                prompt=request.prompt,
                width=w,
                height=h,
                model=account.model,
                endpoint_id=account.endpoint_id,
                timeout=CONFIG.get("timeout", 120)
            )

            if not response.images:
                err_msg = response.text or "No images returned"
                if is_rate_limit_error(Exception(err_msg)):
                    account_manager.mark_rate_limited(account, duration=3600)
                else:
                    account_manager.mark_rate_limited(account, duration=60)
                last_error = Exception(err_msg)
                continue

            image_data_list = []
            for img in response.images:
                if request.response_format == "b64_json":
                    b64 = await download_image_b64(img.url, client._session)
                    image_data_list.append(ImageData(b64_json=b64))
                else:
                    image_data_list.append(ImageData(url=img.url))

            logger.success(f"[{account.name}] Successfully generated {len(image_data_list)} image(s)")
            return ImageResponse(created=int(time.time()), data=image_data_list)

        except Exception as e:
            last_error = e
            if is_rate_limit_error(e):
                account_manager.mark_rate_limited(account, duration=3600)
            else:
                account_manager.mark_rate_limited(account, duration=60)
            continue

    raise HTTPException(status_code=500, detail=f"All accounts failed: {last_error}")


@app.post("/v1/images/edits", response_model=ImageResponse)
async def edit_image(
    image: UploadFile = File(..., description="Image to edit"),
    prompt: str = Form(..., description="Edit instruction"),
    model: str = Form(default="seedream-4-5"),
    n: int = Form(default=1, ge=1, le=4),
    size: Optional[str] = Form(default="2048x2048"),
    response_format: str = Form(default="url"),
    authorization: Optional[str] = Header(None),
):
    """
    Edit an image using Seedream 4.5.
    Supports automatic account rotation on rate limit.
    """
    if not verify_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid API Key")

    temp_path = None
    max_retries = len(account_manager.accounts) if account_manager else 1
    last_error = None

    try:
        # Save uploaded image to temp file
        suffix = Path(image.filename).suffix or ".png"
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        content = await image.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Determine correct width/height from input image ratio
        from PIL import Image
        with Image.open(temp_path) as img:
            input_w, input_h = img.size
        ratio = input_w / input_h
        
        # Always target 4K (14.7M pixels) for Seedream 4.5 as requested
        min_pixels = 14745600 
        target_h = (min_pixels / ratio) ** 0.5
        target_w = target_h * ratio
        w = int(round(target_w / 8) * 8)
        h = int(round(target_h / 8) * 8)
        if w * h < min_pixels:
            if ratio >= 1: w += 8
            else: h += 8

        for attempt in range(max_retries):
            try:
                client, account = await account_manager.get_client()
                logger.info(f"[{account.name}] Editing image with prompt: {prompt[:100]}... (Auto 4K)")
                
                response = await client.generate_image(
                    prompt=prompt,
                    width=w,
                    height=h,
                    model=model,
                    image_path=temp_path,
                    timeout=CONFIG.get("timeout", 120)
                )

                if not response.images:
                    err_msg = response.text or "No images returned"
                    if is_rate_limit_error(Exception(err_msg)):
                        account_manager.mark_rate_limited(account, duration=3600)
                    else:
                        account_manager.mark_rate_limited(account, duration=60)
                    last_error = Exception(err_msg)
                    continue

                image_data_list = []
                for img in response.images:
                    if response_format == "b64_json":
                        b64 = await download_image_b64(img.url, client._session)
                        image_data_list.append(ImageData(b64_json=b64))
                    else:
                        image_data_list.append(ImageData(url=img.url))

                logger.success(f"[{account.name}] Successfully edited image, got {len(image_data_list)} result(s)")
                return ImageResponse(created=int(time.time()), data=image_data_list)

            except Exception as e:
                last_error = e
                if is_rate_limit_error(e):
                    account_manager.mark_rate_limited(account, duration=3600)
                else:
                    account_manager.mark_rate_limited(account, duration=60)
                continue

        raise HTTPException(status_code=500, detail=f"All accounts failed: {last_error}")

    finally:
        if temp_path and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass


if __name__ == "__main__":
    uvicorn.run(
        "SEEDREAM-API:app", 
        host=CONFIG.get("host", "0.0.0.0"), 
        port=CONFIG.get("port", 8002),
        reload=False
    )
