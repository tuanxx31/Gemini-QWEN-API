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
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Header, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Import Gemini client and utilities
from gemini_webapi import GeminiClient
from gemini_webapi.utils import rotate_1psidts
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
# Silence verbose libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("gemini_webapi").setLevel(logging.INFO)

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
_app_data = Path("/app/data")
_local_data = Path(__file__).parent / "data"

if _app_data.exists() and _app_data.is_dir():
    ACCOUNTS_PATH = _app_data / "accounts.json"
elif _local_data.exists() and _local_data.is_dir():
    ACCOUNTS_PATH = _local_data / "accounts.json"
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
    is_dead: bool = False
    status_message: str = "Chưa kiểm tra"
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
        print(f"🔋 Manager: {len(self.accounts)} account(s)")
    
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
                    print(f"📦 Loaded: {account.name}")
            
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
        # Initialize client if needed
        if account.client is None or not account.client._running:
            account.client = GeminiClient(
                secure_1psid=account.cookie_1PSID,
                secure_1psidts=account.cookie_1PSIDTS if account.cookie_1PSIDTS and account.cookie_1PSIDTS != "YOUR_SECURE_1PSIDTS_HERE" else None
            )
            await account.client.init(
                timeout=self.timeout,
                auto_close=False,
                auto_refresh=True
            )
            account.is_dead = False
            account.status_message = "Đang hoạt động"
            print(f"✅ Linked: {account.name}")
        
        return account.client, account

    async def initialize_account(self, account: Account):
        """Khởi tạo một tài khoản cụ thể với cơ chế tự phục hồi cookie TS."""
        if account.client is None or not account.client._running:
            try:
                # Thử lần 1: Dùng cả 2 cookie hiện có
                account.client = GeminiClient(
                    secure_1psid=account.cookie_1PSID,
                    secure_1psidts=account.cookie_1PSIDTS if account.cookie_1PSIDTS and account.cookie_1PSIDTS != "YOUR_SECURE_1PSIDTS_HERE" else None
                )
                await account.client.init(timeout=self.timeout, auto_close=False, auto_refresh=True)
                account.is_dead = False
                account.status_message = "Đang hoạt động"
                print(f"✅ [{account.name}] Auto-init success")
            except Exception as e:
                # Thử lần 2: Nếu lỗi, dùng rotate_1psidts để xin mã mới trực tiếp từ SID
                print(f"🔄 [{account.name}] Cookie TS may be expired ({e}), attempting manual rotate...")
                try:
                    # Tạo dictionary cookie cho hàm rotate
                    manual_cookies = {"__Secure-1PSID": account.cookie_1PSID}
                    if account.cookie_1PSIDTS:
                        manual_cookies["__Secure-1PSIDTS"] = account.cookie_1PSIDTS

                    new_psidts = await rotate_1psidts(manual_cookies)
                    
                    if new_psidts:
                        print(f"✨ [{account.name}] Manual rotate success! New TS obtained.")
                        account.cookie_1PSIDTS = new_psidts
                        account.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
                        
                        # Thử khởi tạo lại với mã mới
                        account.client = GeminiClient(
                            secure_1psid=account.cookie_1PSID,
                            secure_1psidts=new_psidts
                        )
                        await account.client.init(timeout=self.timeout, auto_close=False, auto_refresh=True)
                        
                        account.is_dead = False
                        account.status_message = "Đã tự phục hồi (Auto-Rotate)"
                        self.save_accounts() # Lưu ngay để lần sau không bị lỗi nữa
                        print(f"✅ [{account.name}] Auto-recovery success")
                        return
                    else:
                        raise Exception("Google didn't return a new cookie during manual rotate")
                        
                except Exception as e2:
                    # Thử lần 3: Nếu rotate thất bại, dùng Browser Auto-Login để lấy cookie mới
                    print(f"🔄 [{account.name}] Standard rotate failed, attempting Browser Automation recovery...")
                    browser_success = await self.run_browser_auto_login(account.name)
                    if browser_success:
                        # Sau khi browser login thành công, thử khởi tạo lại 1 lần cuối
                        # Ta reload lại dnah sách account nội bộ
                        for acc_data in self.accounts:
                            if acc_data.name == account.name:
                                try:
                                    account.cookie_1PSID = acc_data.cookie_1PSID
                                    account.cookie_1PSIDTS = acc_data.cookie_1PSIDTS
                                    account.client = GeminiClient(secure_1psid=account.cookie_1PSID, secure_1psidts=account.cookie_1PSIDTS)
                                    await account.client.init(timeout=self.timeout, auto_close=False, auto_refresh=True)
                                    account.is_dead = False
                                    account.status_message = "Đã phục hồi qua Browser Automation"
                                    print(f"✅ [{account.name}] Browser recovery successful")
                                    return
                                except Exception as e3:
                                    print(f"❌ [{account.name}] Final init failed after browser login: {e3}")
                    
                    account.is_dead = True
                    account.status_message = f"Khởi tạo lỗi (Hết hạn): {str(e2)}"
                    print(f"❌ [{account.name}] Auto-recovery failed: {e2}")

    async def initialize_all(self):
        """Khởi tạo song song tất cả các tài khoản đang active."""
        if not self.accounts:
            return
        
        print(f"[API] Starting auto-initialization for {len(self.accounts)} account(s)...")
        tasks = [self.initialize_account(acc) for acc in self.accounts if acc.is_active]
        if tasks:
            await asyncio.gather(*tasks)
        print("[API] All accounts initialization finished")

    async def sync_with_clients(self):
        """Duyệt qua các client đang chạy, cập nhật cookie mới và lưu vào file nếu có thay đổi."""
        changed = False
        for account in self.accounts:
            if account.client and account.client._running:
                # 1. Kiểm tra cookie rotate (PSIDTS)
                new_psidts = account.client.cookies.get("__Secure-1PSIDTS")
                if new_psidts and new_psidts != account.cookie_1PSIDTS:
                    print(f"🔄 [Sync] {account.name} - New Cookie! Saving...")
                    account.cookie_1PSIDTS = new_psidts
                    account.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
                    changed = True
                
                # 2. Kiểm tra status (thông qua task rotate_tasks trong gemini_webapi.utils)
                from gemini_webapi.utils import rotate_tasks
                psid = account.cookie_1PSID
                if psid in rotate_tasks:
                    task = rotate_tasks[psid]
                    if task.done() and task.exception():
                        # Nếu task refresh bị lỗi (thường là AuthError), đánh dấu die
                        if not account.is_dead:
                            print(f"[Sync] Tài khoản {account.name} đã bị DIE (Lỗi refresh)!")
                            account.is_dead = True
                            account.status_message = f"Lỗi xác thực (Die): {task.exception()}"
                            changed = True
                    elif not task.done():
                        account.is_dead = False
                        account.status_message = "Đang chạy (Live)"
                
                # 3. Nếu account active nhưng chưa có client (ví dụ bị đóng do idle)
                # chúng ta không cần force khởi tạo ở đây vì get_client sẽ lo
            else:
                if account.is_active and not account.is_dead:
                    account.status_message = "Chờ khởi tạo"

        if changed:
            self.save_accounts()
            print("[Sync] Đã tự động lưu các thay đổi vào accounts.json")
            
    async def run_auto_heartbeat(self):
        """Gửi tin nhắn 'hi' định kỳ để giữ cho session luôn 'nóng' (tự động 100%)."""
        print("[API] Đang chạy Auto Heartbeat cho tất cả tài khoản...")
        for account in self.accounts:
            if account.is_active and not account.is_dead:
                try:
                    if account.client is None or not account.client._running:
                        await self.initialize_account(account)
                    
                    if account.client and account.client._running:
                        print(f"[API] [{account.name}] Gửi Heartbeat ping...")
                        # Gửi prompt ngắn nhất có thể
                        await account.client.generate_content("hi", timeout=60)
                        print(f"[API] [{account.name}] Heartbeat thành công ✅")
                except Exception as e:
                    print(f"[API] [{account.name}] Heartbeat thất bại: {e}")
            
    async def run_browser_auto_login(self, email: str = None):
        """Chạy script automation trình duyệt để lấy cookie mới."""
        print(f"[Browser] Đang tự động đăng nhập để lấy cookie cho {email or 'tất cả'}...")
        try:
            # Sử dụng sys.executable để lấy đúng đường dẫn python đang chạy
            import sys
            import subprocess
            cmd = [sys.executable, "browser_auto_login.py"]
            if email:
                cmd.extend(["--email", email])
            
            # Chạy script và đợi tối đa 120 giây
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=120)
            
            if process.returncode == 0:
                print(f"[Browser] Lấy cookie thành công cho {email or 'tất cả'}!")
                # Load lại accounts từ file (vì script đã cập nhật accounts.json)
                self._load_accounts()
                return True
            else:
                print(f"[Browser] Lấy cookie thất bại: {stderr}")
                return False
        except Exception as e:
            print(f"[Browser] Lỗi khi chạy script automation: {e}")
            return False
            
    async def force_rotate_account(self, account_id: int) -> tuple[bool, str]:
        """Bắt buộc làm mới cookie cho một tài khoản cụ thể."""
        if account_id < 0 or account_id >= len(self.accounts):
            return False, "Không tìm thấy tài khoản"
            
        account = self.accounts[account_id]
        from gemini_webapi.utils.rotate_1psidts import rotate_1psidts
        
        cookies = {
            "__Secure-1PSID": account.cookie_1PSID,
            "__Secure-1PSIDTS": account.cookie_1PSIDTS or ""
        }
        
        try:
            new_psidts = await rotate_1psidts(cookies)
            if new_psidts:
                account.cookie_1PSIDTS = new_psidts
                account.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
                account.is_dead = False
                account.status_message = "Đã làm mới (Manual Rotate)"
                
                # Cập nhật client đang chạy nếu có
                if account.client and account.client._running:
                    account.client.cookies["__Secure-1PSIDTS"] = new_psidts
                    account.client.client.cookies.set("__Secure-1PSIDTS", new_psidts)
                
                self.save_accounts()
                return True, "Làm mới thành công!"
            else:
                # Nếu không có cookie mới, thử chạy Browser Auto-Login để cứu
                print(f"[API] [{account.name}] No new cookie, attempting Browser Auto-Login...")
                browser_success = await self.run_browser_auto_login(account.name)
                if browser_success:
                    # Load lại account đã được cập nhật
                    for acc in self.accounts:
                        if acc.name == account.name:
                            await self.initialize_account(acc)
                            if not acc.is_dead:
                                return True, "Đã khôi phục thành công bằng Browser Automation!"
                
                return False, "Google không trả về cookie mới và tự động đăng nhập thất bại"
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                error_msg = "Cookie 1PSID đã chết, không thể làm mới TS"
            return False, f"Lỗi làm mới: {error_msg}"
    
    async def reload(self):
        """Reload accounts from file and close old clients."""
        print("[API] Reloading accounts...")
        
        # Close all existing clients
        await self.close_all()
        
        # Reload accounts
        self._load_accounts()
        self.current_index = 0
        
        # Khởi tạo lại tất cả tài khoản ngay lập tức
        asyncio.create_task(self.initialize_all())
        
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
                "is_dead": acc.is_dead,
                "status_message": acc.status_message,
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


# ============ Chat API Models ============

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gemini-2.5-flash")
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


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


def verify_auth(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key")
) -> bool:
    """Verify authorization. Supports Authorization (Bearer) and x-api-key headers."""
    # 1. Get token from any available header
    token = None
    if authorization:
        parts = authorization.split(" ")
        token = parts[1] if len(parts) == 2 else parts[0]
    elif x_api_key:
        token = x_api_key
        
    if not token:
        return False

    # 2. Check if it's a valid admin session
    if verify_admin_session(token):
        return True
        
    # 3. Check if it matches the API key
    expected_token = CONFIG.get("api_key", "sk-demo")
    return token == expected_token


def verify_admin_auth(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key")
) -> bool:
    """Verify admin authorization. Hỗ trợ cả Authorization và x-api-key."""
    token = None
    if authorization:
        parts = authorization.split(" ")
        token = parts[1] if len(parts) == 2 else parts[0]
    elif x_api_key:
        token = x_api_key
        
    if not token:
        return False
    
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

async def background_sync_task():
    """Chạy ngầm để đồng bộ cookie và kiểm tra sức khỏe tài khoản mỗi 30 giây."""
    print("[API] Background sync task started")
    heartbeat_counter = 0
    # Chạy lần đầu sau 10 phút khởi động, sau đó cứ mỗi 1 tiếng (Dành cho VPS)
    heartbeat_interval = 3600 
    
    while True:
        try:
            await asyncio.sleep(30)
            heartbeat_counter += 30
            
            if account_manager:
                # 1. Đồng bộ cookie PSIDTS (Xoay vòng tự động)
                await account_manager.sync_with_clients()
                
                # 2. Heartbeat định kỳ (Giữ session sống)
                if heartbeat_counter >= heartbeat_interval:
                    await account_manager.run_auto_heartbeat()
                    heartbeat_counter = 0
                    
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[Sync Error] {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global account_manager
    print(f"[API] Starting Gemini Image API Server on {CONFIG.get('host', '0.0.0.0')}:{CONFIG.get('port', 8000)}")
    account_manager = AccountManager(ACCOUNTS_PATH, timeout=CONFIG.get("timeout", 120))
    
    # Khởi tạo toàn bộ account ngay khi bật server
    asyncio.create_task(account_manager.initialize_all())
    
    # Khởi tạo background task
    sync_task = asyncio.create_task(background_sync_task())
    
    yield
    # Shutdown
    sync_task.cancel()
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

        print(f"🚀 [Gen] [{account.name}] Prompt: {full_prompt[:80]}...")

        # Call Gemini API
        response = await client.generate_content(
            prompt=full_prompt,
            model=request.model,
            image_mode=True,
            timeout=CONFIG.get("timeout", 120),
            debug_mode=True
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

        print(f"✅ [Gen] [{account.name}] Done: {len(image_data_list)} image(s)")

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
    response_format: str = Form(default="b64_json", description="Response format: b64_json or url"),
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
        start_time = time.time()
        response = await client.generate_content(
            prompt=full_prompt,
            files=[temp_path],
            model=model,
            image_mode=True,
            timeout=CONFIG.get("timeout", 120),
            debug_mode=False
        )
        print(f"⚡ [Edit] [{account.name}] API Responded: {time.time() - start_time:.2f}s")

        if not response.images:
            raise HTTPException(
                status_code=500,
                detail=f"No images generated. Response: {response.text[:200] if response.text else 'No text'}"
            )

        # Get revised prompt from response
        revised_prompt = response.text or prompt

        # Download and convert images to base64
        image_data_list = []
        # Always choose the first image as requested by user (best quality)
        images_to_process = response.images[:1]

        for idx, img in enumerate(images_to_process):
            try:
                dl_start = time.time()
                b64_string, url = await download_image_as_base64(img, client.cookies)
                print(f"📥 [DL] [{account.name}] Image {idx+1}/{len(images_to_process)}: {time.time() - dl_start:.2f}s")
                image_data_list.append(ImageData(
                    b64_json=b64_string if response_format == "b64_json" else None,
                    url=url if response_format == "url" else None,
                    revised_prompt=revised_prompt
                ))
            except Exception as e:
                print(f"[API] Error downloading image {idx+1}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not image_data_list:
            # Check if any image was found at all
            if not response.images:
                raise HTTPException(status_code=500, detail=f"Gemini did not return any images. Text: {response.text[:100]}")
            else:
                raise HTTPException(status_code=500, detail=f"Failed to download/process {len(response.images)} images returned by Gemini.")

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


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):
    """
    OpenAI-compatible chat completions endpoint.
    """
    if not verify_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid authorization token")

    try:
        client, account = await get_gemini_client()
        
        # Convert messages to a single prompt or handle history
        # For simple mapping, we use the last message as prompt
        # TODO: Implement full history support if GeminiClient supports it easily
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages list cannot be empty")
            
        full_prompt = ""
        for msg in request.messages:
            full_prompt += f"{msg.role}: {msg.content}\n"
        
        # Add a final instruction for Gemini to respond as assistant
        full_prompt += "assistant: "

        print(f"[API] [{account.name}] Chat request with {len(request.messages)} messages")

        response = await client.generate_content(
            prompt=full_prompt,
            model=request.model,
            timeout=CONFIG.get("timeout", 120),
            debug_mode=True
        )

        choices = [
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response.text or ""),
                finish_reason="stop"
            )
        ]

        return ChatCompletionResponse(
            id=f"chatcmpl-{secrets.token_hex(12)}",
            created=int(time.time()),
            model=request.model,
            choices=choices
        )

    except Exception as e:
        print(f"[API] Chat Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/admin/accounts/{account_id}/refresh")
async def force_refresh_account(
    account_id: int,
    authorization: Optional[str] = Header(None)
):
    """Bắt buộc làm mới cookie cho một tài khoản cụ thể (requires auth)."""
    if not verify_admin_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    global account_manager
    if account_manager is None:
        raise HTTPException(status_code=500, detail="AccountManager not initialized")
    
    success, message = await account_manager.force_rotate_account(account_id)
    if success:
        return {"success": True, "message": message}
    else:
        raise HTTPException(status_code=400, detail=message)


class AccountTestRequest(BaseModel):
    cookie_1PSID: str = Field(..., description="Cookie 1PSID to test")
    cookie_1PSIDTS: Optional[str] = Field(None, description="Cookie 1PSIDTS to test")


@app.post("/admin/accounts/test")
async def test_account_cookies(
    request: AccountTestRequest,
    authorization: Optional[str] = Header(None)
):
    """Test if cookies are valid without saving (requires auth)."""
    if not verify_admin_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    print(f"[API] Testing cookies for manual request...")
    
    try:
        # Initialize a temporary client
        test_client = GeminiClient(
            secure_1psid=request.cookie_1PSID,
            secure_1psidts=request.cookie_1PSIDTS if request.cookie_1PSIDTS else None
        )
        
        # Try to initialize (this checks if cookies are valid)
        await test_client.init(timeout=30, auto_close=True, verbose=False)
        
        is_active = test_client._running
        await test_client.close()
        
        if is_active:
            return {"success": True, "message": "Cookies are valid! ✅"}
        else:
            return {"success": False, "message": "Cookies are invalid or expired. ❌"}
            
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return {"success": False, "message": "Cookies expired (401 Unauthorized). ❌"}
        return {"success": False, "message": f"Error: {error_msg}"}


@app.post("/admin/accounts/rotate_test")
async def manual_rotate_test(
    request: AccountTestRequest,
    authorization: Optional[str] = Header(None)
):
    """Test cookie rotation manually (requires auth)."""
    if not verify_admin_auth(authorization):
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    from gemini_webapi.utils.rotate_1psidts import rotate_1psidts
    
    print(f"[API] Manual rotation test requested...")
    
    if not request.cookie_1PSID:
        return {"success": False, "message": "Missing __Secure-1PSID cookie."}

    cookies = {
        "__Secure-1PSID": request.cookie_1PSID,
        "__Secure-1PSIDTS": request.cookie_1PSIDTS or ""
    }
    
    try:
        new_1psidts = await rotate_1psidts(cookies)
        if new_1psidts:
            return {
                "success": True, 
                "message": "Rotate successful! ✅",
                "new_1psidts": new_1psidts
            }
        else:
            return {
                "success": False, 
                "message": "Rotate failed: No new cookie returned (Internal check failed). ❌"
            }
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return {"success": False, "message": "Rotate failed: Unauthorized/Expired (401). ❌"}
        return {"success": False, "message": f"Rotate Error: {error_msg}"}


# ============ Main ============

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=CONFIG.get("host", "0.0.0.0"),
        port=CONFIG.get("port", 8989),
        reload=False
    )
