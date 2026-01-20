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
import random
import secrets
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Header, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# Paths
# Use /app/data/accounts.json in container (when /app/data is mounted), 
# otherwise use accounts.json in same directory as api_server.py
_data_path = Path("/app/data")
if _data_path.exists() and _data_path.is_dir():
    ACCOUNTS_PATH = _data_path / "accounts.json"
else:
    ACCOUNTS_PATH = Path(__file__).parent / "accounts.json"

# ============ Admin Authentication ============

# Admin accounts mặc định
ADMIN_ACCOUNTS = {
    "admin@tuna311": "admin31zx@@",
    "kinlaster@admin": "admin@kinlaster123@@zz",
}

# Active admin sessions: token -> {username: str, created_at: float}
# Session hết hạn sau 12 giờ (43200 giây)
admin_sessions: Dict[str, dict] = {}
SESSION_DURATION = 12 * 60 * 60  # 12 giờ


def verify_admin_credentials(username: str, password: str) -> bool:
    """Xác thực thông tin đăng nhập admin."""
    return username in ADMIN_ACCOUNTS and ADMIN_ACCOUNTS[username] == password


def create_admin_session(username: str) -> str:
    """Tạo session token cho admin với thời hạn 12 giờ."""
    token = secrets.token_urlsafe(32)
    admin_sessions[token] = {
        "username": username,
        "created_at": time.time()
    }
    return token


def verify_admin_session(token: str) -> Optional[str]:
    """Xác thực admin session token, trả về username nếu hợp lệ và chưa hết hạn."""
    if token not in admin_sessions:
        return None
    
    session = admin_sessions[token]
    created_at = session.get("created_at", 0)
    elapsed = time.time() - created_at
    
    # Kiểm tra hết hạn (12 giờ)
    if elapsed > SESSION_DURATION:
        # Xóa session hết hạn
        del admin_sessions[token]
        return None
    
    return session.get("username")


def cleanup_expired_sessions():
    """Dọn dẹp các session đã hết hạn."""
    current_time = time.time()
    expired_tokens = [
        token for token, session in admin_sessions.items()
        if current_time - session.get("created_at", 0) > SESSION_DURATION
    ]
    for token in expired_tokens:
        del admin_sessions[token]
    if expired_tokens:
        print(f"[API] Cleaned up {len(expired_tokens)} expired admin session(s)")

# ============ Account Management ============

@dataclass
class Account:
    """Represents a Gemini account."""
    name: str
    cookie_1PSID: str
    cookie_1PSIDTS: str = ""
    client: Optional[GeminiClient] = None
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __post_init__(self):
        if not self.cookie_1PSID or self.cookie_1PSID == "YOUR_SECURE_1PSID_HERE":
            self.is_active = False


class AccountManager:
    """Manages multiple Gemini accounts with round-robin selection."""
    
    def __init__(self, accounts_path: Path, timeout: int = 120):
        self.accounts_path = accounts_path
        self.timeout = timeout
        self.accounts: List[Account] = []
        self.current_index = 0
        self._load_accounts()
        print(f"[API] AccountManager initialized with {len(self.accounts)} account(s)")
    
    def _load_accounts(self):
        """Load accounts from accounts.json file. Auto-create file if not exists."""
        self.accounts = []
        
        # Auto-create accounts.json if not exists
        if not self.accounts_path.exists():
            print(f"[INFO] accounts.json not found at {self.accounts_path}, creating new file...")
            try:
                # Create parent directory if needed
                self.accounts_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create default empty accounts file
                default_data = {"accounts": []}
                with open(self.accounts_path, "w", encoding="utf-8") as f:
                    json.dump(default_data, f, indent=2, ensure_ascii=False)
                print(f"[INFO] Created new accounts.json at {self.accounts_path}")
            except Exception as e:
                print(f"[ERROR] Failed to create accounts.json: {e}")
                return
        
        try:
            with open(self.accounts_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            accounts_data = data.get("accounts", [])
            
            # Get current timestamp for fallback
            current_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            
            for acc_data in accounts_data:
                # Load timestamps, fallback to current time for old accounts
                created_at = acc_data.get("created_at", current_timestamp)
                updated_at = acc_data.get("updated_at", current_timestamp)
                
                account = Account(
                    name=acc_data.get("name", f"Account {len(self.accounts) + 1}"),
                    cookie_1PSID=acc_data.get("cookie_1PSID", ""),
                    cookie_1PSIDTS=acc_data.get("cookie_1PSIDTS", ""),
                    created_at=created_at,
                    updated_at=updated_at
                )
                if account.cookie_1PSID:
                    self.accounts.append(account)
                    print(f"[API] Loaded account: {account.name}")
            
            if not self.accounts:
                print("[INFO] No accounts found in accounts.json. You can add accounts via /admin interface")
        
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON in accounts.json: {e}")
            print("[INFO] File will be recreated on next save")
        except Exception as e:
            print(f"[ERROR] Failed to load accounts.json: {e}")
    
    async def get_client(self) -> tuple[GeminiClient, Account]:
        """Get an available client using round-robin selection."""
        if not self.accounts:
            raise HTTPException(
                status_code=500,
                detail="No accounts configured. Please add accounts via /admin interface"
            )
        
        # Round-robin: get next account
        account = self.accounts[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.accounts)
        
        # Initialize client if needed
        if account.client is None or not account.client._running:
            print(f"[API] Initializing client for account: {account.name}")
            
            account.client = GeminiClient(
                secure_1psid=account.cookie_1PSID,
                secure_1psidts=account.cookie_1PSIDTS if account.cookie_1PSIDTS and account.cookie_1PSIDTS != "YOUR_SECURE_1PSIDTS_HERE" else None
            )
            
            await account.client.init(
                timeout=self.timeout,
                auto_close=False,
                auto_refresh=True
            )
            print(f"[API] Client initialized for account: {account.name}")
        
        return account.client, account
    
    async def reload(self):
        """Reload accounts from file and close old clients."""
        print("[API] Reloading accounts...")
        
        # Close all existing clients
        await self.close_all()
        
        # Reload accounts
        self._load_accounts()
        self.current_index = 0
        
        print(f"[API] Reloaded {len(self.accounts)} account(s)")
    
    async def close_all(self):
        """Close all client sessions."""
        for account in self.accounts:
            if account.client and account.client._running:
                try:
                    await account.client.close()
                    print(f"[API] Closed client for account: {account.name}")
                except Exception as e:
                    print(f"[API] Error closing client for {account.name}: {e}")
                finally:
                    account.client = None
    
    def get_status(self) -> List[dict]:
        """Get status of all accounts."""
        return [
            {
                "name": acc.name,
                "active": acc.client is not None and acc.client._running if acc.client else False,
                "is_active": acc.is_active
            }
            for acc in self.accounts
        ]
    
    def get_accounts_data(self) -> List[dict]:
        """Get accounts data for API responses (full data, requires auth)."""
        return [
            {
                "id": idx,
                "name": acc.name,
                "cookie_1PSID": acc.cookie_1PSID,
                "cookie_1PSIDTS": acc.cookie_1PSIDTS,
                "is_active": acc.is_active,
                "active": acc.client is not None and acc.client._running if acc.client else False,
                "created_at": acc.created_at,
                "updated_at": acc.updated_at
            }
            for idx, acc in enumerate(self.accounts)
        ]
    
    def save_accounts(self):
        """Save accounts to file."""
        accounts_data = [
            {
                "name": acc.name,
                "cookie_1PSID": acc.cookie_1PSID,
                "cookie_1PSIDTS": acc.cookie_1PSIDTS,
                "created_at": acc.created_at or time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "updated_at": acc.updated_at or time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            }
            for acc in self.accounts
        ]
        
        data = {"accounts": accounts_data}
        
        try:
            with open(self.accounts_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[API] Saved {len(self.accounts)} account(s) to {self.accounts_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save accounts.json: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save accounts: {e}")


# Global account manager
account_manager: Optional[AccountManager] = None


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

async def get_gemini_client() -> tuple[GeminiClient, Account]:
    """Get Gemini client using AccountManager with round-robin selection."""
    global account_manager
    
    if account_manager is None:
        raise HTTPException(
            status_code=500,
            detail="AccountManager not initialized"
        )
    
    return await account_manager.get_client()


def verify_auth(authorization: Optional[str]) -> bool:
    """Verify Bearer token authorization (cho API endpoints)."""
    if not authorization:
        return False

    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False

    token = parts[1]
    expected_token = CONFIG.get("api_key", "sk-demo")

    return token == expected_token


def verify_admin_auth(authorization: Optional[str]) -> bool:
    """Verify admin authorization (cho admin endpoints). Hỗ trợ cả Bearer token và admin session token."""
    if not authorization:
        return False

    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False

    token = parts[1]
    
    # Kiểm tra admin session token trước
    if verify_admin_session(token):
        return True
    
    # Kiểm tra Bearer token (API key) như cũ
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
    global account_manager
    print(f"[API] Starting Gemini Image API Server on {CONFIG.get('host', '0.0.0.0')}:{CONFIG.get('port', 8000)}")
    account_manager = AccountManager(ACCOUNTS_PATH, timeout=CONFIG.get("timeout", 120))
    yield
    # Shutdown
    if account_manager:
        await account_manager.close_all()
        print("[API] All Gemini clients closed")


# ============ FastAPI App ============

app = FastAPI(
    title="Gemini Image Generation API",
    description="OpenAI-compatible API for Google Gemini image generation",
    version="1.0.0",
    lifespan=lifespan
)

# Static files serving
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

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
        "message": "Gemini Image Generation APIIII",
        "endpoints": {
            "models": "GET /models",
            "generate": "POST /v1/images/generations",
            "edit": "POST /v1/images/edits",
            "admin": "GET /admin (Web interface for account management)"
        }
    }


@app.get("/admin")
async def admin_interface():
    """Serve admin web interface."""
    static_file = Path(__file__).parent / "static" / "index.html"
    if static_file.exists():
        return FileResponse(str(static_file))
    raise HTTPException(status_code=404, detail="Admin interface not found")


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
        client, account = await get_gemini_client()

        # Build prompt with size modifier
        full_prompt = request.prompt + get_size_prompt(request.size)

        # Add seed to prompt if provided
        if request.seed:
            full_prompt += f" (seed: {request.seed})"

        print(f"[API] [{account.name}] Generating image with prompt: {full_prompt[:100]}...")

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

        print(f"[API] [{account.name}] Successfully generated {len(image_data_list)} image(s)")

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
        client, account = await get_gemini_client()

        # Save uploaded image to temp file
        fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)

        content = await image.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Build prompt with size modifier
        full_prompt = prompt + get_size_prompt(size)

        print(f"[API] [{account.name}] Editing image with prompt: {full_prompt[:100]}...")

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

        print(f"[API] [{account.name}] Successfully edited image, got {len(image_data_list)} result(s)")

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
    global account_manager
    return {
        "status": "healthy",
        "total_accounts": len(account_manager.accounts) if account_manager else 0,
        "accounts": account_manager.get_status() if account_manager else []
    }


# ============ Admin Endpoints ============

class AdminLoginRequest(BaseModel):
    username: str = Field(..., description="Admin username")
    password: str = Field(..., description="Admin password")


class AdminLoginResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    message: str


class AccountCreateRequest(BaseModel):
    name: str = Field(..., description="Account name")
    cookie_1PSID: str = Field(..., description="Cookie 1PSID")
    cookie_1PSIDTS: str = Field(default="", description="Cookie 1PSIDTS (optional)")


@app.post("/admin/login", response_model=AdminLoginResponse)
async def admin_login(request: AdminLoginRequest):
    """Đăng nhập admin với username/password."""
    if verify_admin_credentials(request.username, request.password):
        token = create_admin_session(request.username)
        return AdminLoginResponse(
            success=True,
            token=token,
            message="Đăng nhập thành công"
        )
    else:
        raise HTTPException(status_code=401, detail="Tên đăng nhập hoặc mật khẩu không đúng")


class AccountUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, description="Account name")
    cookie_1PSID: Optional[str] = Field(None, description="Cookie 1PSID")
    cookie_1PSIDTS: Optional[str] = Field(None, description="Cookie 1PSIDTS")


@app.get("/admin/accounts")
async def get_accounts(authorization: Optional[str] = Header(None)):
    """Get all accounts (requires auth)."""
    if not verify_admin_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    global account_manager
    if account_manager is None:
        raise HTTPException(status_code=500, detail="AccountManager not initialized")
    
    return {
        "total_accounts": len(account_manager.accounts),
        "accounts": account_manager.get_accounts_data()
    }


@app.get("/admin/accounts/{account_id}")
async def get_account(account_id: int, authorization: Optional[str] = Header(None)):
    """Get detailed account information (requires auth)."""
    if not verify_admin_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    global account_manager
    if account_manager is None:
        raise HTTPException(status_code=500, detail="AccountManager not initialized")
    
    if account_id < 0 or account_id >= len(account_manager.accounts):
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    
    account = account_manager.accounts[account_id]
    return {
        "id": account_id,
        "name": account.name,
        "cookie_1PSID": account.cookie_1PSID,
        "cookie_1PSIDTS": account.cookie_1PSIDTS,
        "is_active": account.is_active,
        "client_active": account.client is not None and account.client._running if account.client else False,
        "created_at": account.created_at,
        "updated_at": account.updated_at
    }


@app.post("/admin/accounts")
async def create_account(
    request: AccountCreateRequest,
    authorization: Optional[str] = Header(None)
):
    """Add a new account (requires auth)."""
    if not verify_admin_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    global account_manager
    if account_manager is None:
        raise HTTPException(status_code=500, detail="AccountManager not initialized")
    
    # Create new account with timestamps
    current_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    new_account = Account(
        name=request.name,
        cookie_1PSID=request.cookie_1PSID,
        cookie_1PSIDTS=request.cookie_1PSIDTS,
        created_at=current_timestamp,
        updated_at=current_timestamp
    )
    
    account_manager.accounts.append(new_account)
    account_manager.save_accounts()
    
    return {
        "message": "Account added successfully",
        "account": {
            "id": len(account_manager.accounts) - 1,
            "name": new_account.name,
            "is_active": new_account.is_active
        }
    }


@app.put("/admin/accounts/{account_id}")
async def update_account(
    account_id: int,
    request: AccountUpdateRequest,
    authorization: Optional[str] = Header(None)
):
    """Update an account (requires auth)."""
    if not verify_admin_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    global account_manager
    if account_manager is None:
        raise HTTPException(status_code=500, detail="AccountManager not initialized")
    
    if account_id < 0 or account_id >= len(account_manager.accounts):
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    
    account = account_manager.accounts[account_id]
    
    # Close existing client if updating credentials
    if request.cookie_1PSID and request.cookie_1PSID != account.cookie_1PSID:
        if account.client and account.client._running:
            await account.client.close()
            account.client = None
    
    # Update account
    if request.name is not None:
        account.name = request.name
    if request.cookie_1PSID is not None:
        account.cookie_1PSID = request.cookie_1PSID
        account.is_active = bool(request.cookie_1PSID and request.cookie_1PSID != "YOUR_SECURE_1PSID_HERE")
    if request.cookie_1PSIDTS is not None:
        account.cookie_1PSIDTS = request.cookie_1PSIDTS
    
    # Update updated_at timestamp (keep created_at unchanged)
    account.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    # Ensure created_at exists (for old accounts)
    if not account.created_at:
        account.created_at = account.updated_at
    
    account_manager.save_accounts()
    
    return {
        "message": "Account updated successfully",
        "account": {
            "id": account_id,
            "name": account.name,
            "is_active": account.is_active
        }
    }


@app.delete("/admin/accounts/{account_id}")
async def delete_account(
    account_id: int,
    authorization: Optional[str] = Header(None)
):
    """Delete an account (requires auth)."""
    if not verify_admin_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    global account_manager
    if account_manager is None:
        raise HTTPException(status_code=500, detail="AccountManager not initialized")
    
    if account_id < 0 or account_id >= len(account_manager.accounts):
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    
    account = account_manager.accounts[account_id]
    
    # Close client if exists
    if account.client and account.client._running:
        await account.client.close()
    
    # Remove account
    account_manager.accounts.pop(account_id)
    account_manager.save_accounts()
    
    # Reset current_index if needed
    if account_manager.current_index >= len(account_manager.accounts):
        account_manager.current_index = 0
    
    return {
        "message": "Account deleted successfully",
        "deleted_account_id": account_id
    }


@app.post("/admin/accounts/reload")
async def reload_accounts(authorization: Optional[str] = Header(None)):
    """Reload accounts from file (requires auth)."""
    if not verify_admin_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    global account_manager
    if account_manager is None:
        raise HTTPException(status_code=500, detail="AccountManager not initialized")
    
    await account_manager.reload()
    
    return {
        "message": "Accounts reloaded successfully",
        "total_accounts": len(account_manager.accounts)
    }


# ============ Main ============

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=CONFIG.get("host", "0.0.0.0"),
        port=CONFIG.get("port", 8989),
        reload=False
    )
