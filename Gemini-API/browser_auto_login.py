"""
Browser Automation Script for Google Auto-Login
Automatically logs into Google and extracts Gemini cookies.

Requirements:
    pip install playwright pyotp
    playwright install chromium

Usage:
    python browser_auto_login.py [--headless]
"""

import asyncio
import json
import argparse
from pathlib import Path
from datetime import datetime

try:
    import pyotp
except ImportError:
    print("❌ Thiếu thư viện pyotp. Chạy: pip install pyotp")
    exit(1)

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
except ImportError:
    print("❌ Thiếu thư viện playwright. Chạy: pip install playwright && playwright install chromium")
    exit(1)

# Paths
SCRIPT_DIR = Path(__file__).parent

# Auto-detect data directory (for Server/Docker)
DATA_DIR = SCRIPT_DIR
if Path("/app/data").exists():
    DATA_DIR = Path("/app/data")
elif (SCRIPT_DIR / "data").exists():
    DATA_DIR = SCRIPT_DIR / "data"

CREDENTIALS_PATH = DATA_DIR / "credentials.json"
ACCOUNTS_PATH = DATA_DIR / "accounts.json"
BROWSER_DATA_PATH = DATA_DIR / "browser_data"

# Fallback for credentials if not in DATA_DIR
if not CREDENTIALS_PATH.exists() and (SCRIPT_DIR / "credentials.json").exists():
    CREDENTIALS_PATH = SCRIPT_DIR / "credentials.json"

# URLs
GEMINI_URL = "https://gemini.google.com/"
GOOGLE_LOGIN_URL = "https://accounts.google.com/signin"


def load_credentials():
    """Load credentials from credentials.json"""
    if not CREDENTIALS_PATH.exists():
        print(f"❌ Không tìm thấy file {CREDENTIALS_PATH}")
        print("Vui lòng tạo file credentials.json với nội dung:")
        print('''
{
  "accounts": [
    {
      "email": "your_email@gmail.com",
      "password": "your_password",
      "totp_secret": "your_totp_secret_without_spaces"
    }
  ]
}
        ''')
        return None
    
    with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_totp(secret: str) -> str:
    """Generate TOTP code from secret"""
    # Remove spaces and convert to uppercase
    secret = secret.replace(" ", "").upper()
    totp = pyotp.TOTP(secret)
    return totp.now()


async def login_google(page, email: str, password: str, totp_secret: str = None):
    """Login to Google account"""
    print(f"[LOGIN] Dang dang nhap vao {email}...")
    
    # Go to Google login
    await page.goto(GOOGLE_LOGIN_URL)
    await asyncio.sleep(2)
    
    # Enter email
    print("[EMAIL] Nhap email...")
    await page.fill('input[type="email"]', email)
    await page.click('button:has-text("Tiếp theo"), button:has-text("Next")')
    await asyncio.sleep(3)
    
    # Enter password
    print("[PASS] Nhap mat khau...")
    try:
        await page.wait_for_selector('input[type="password"]', timeout=10000)
        await page.fill('input[type="password"]', password)
        await page.click('button:has-text("Tiếp theo"), button:has-text("Next")')
        await asyncio.sleep(3)
    except PlaywrightTimeout:
        print("[WARN] Khong tim thay o mat khau, co the da dang nhap san")
    
    # Check for 2FA
    try:
        # Check if TOTP input is present
        totp_input = await page.query_selector('input[type="tel"]')
        if totp_input and totp_secret:
            print("[OTP] Nhap ma Google Authenticator...")
            totp_code = generate_totp(totp_secret)
            print(f"   Mã OTP: {totp_code}")
            await totp_input.fill(totp_code)
            await page.click('button:has-text("Tiếp theo"), button:has-text("Next")')
            await asyncio.sleep(3)
    except Exception as e:
        print(f"[WARN] Khong can xac thuc 2FA hoac loi: {e}")
    
    # Check for "confirm this is you" prompt
    try:
        confirm_button = await page.query_selector('button:has-text("Xác nhận"), button:has-text("Confirm")')
        if confirm_button:
            print("[WAIT] Google yeu cau xac nhan. Vui long xac nhan tren dien thoai...")
            # Wait up to 60 seconds for user to confirm
            await asyncio.sleep(60)
    except:
        pass
    
    print("[OK] Dang nhap hoan tat!")
    return True


async def get_gemini_cookies(page):
    """Navigate to Gemini and extract cookies"""
    print("[GEMINI] Dang vao trang Gemini...")
    try:
        await page.goto(GEMINI_URL, timeout=60000)  # 60 second timeout
    except Exception as e:
        print(f"[WARN] Trang load cham: {e}")
    await asyncio.sleep(5)
    
    # Get cookies
    cookies = await page.context.cookies()
    
    result = {}
    for cookie in cookies:
        if cookie["name"] == "__Secure-1PSID":
            result["1PSID"] = cookie["value"]
        elif cookie["name"] == "__Secure-1PSIDTS":
            result["1PSIDTS"] = cookie["value"]
    
    if "1PSID" in result:
        print(f"[OK] Da lay duoc cookie 1PSID: {result['1PSID'][:30]}...")
    if "1PSIDTS" in result:
        print(f"[OK] Da lay duoc cookie 1PSIDTS: {result['1PSIDTS'][:30]}...")
    
    return result


def update_accounts_json(email: str, cookies: dict):
    """Update accounts.json with new cookies"""
    accounts_data = {"accounts": []}
    
    if ACCOUNTS_PATH.exists():
        with open(ACCOUNTS_PATH, "r", encoding="utf-8") as f:
            accounts_data = json.load(f)
    
    # Find and update account or create new
    found = False
    for acc in accounts_data.get("accounts", []):
        if acc.get("name") == email or acc.get("email") == email:
            acc["cookie_1PSID"] = cookies.get("1PSID", "")
            acc["cookie_1PSIDTS"] = cookies.get("1PSIDTS", "")
            acc["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
            acc["is_active"] = True
            found = True
            break
    
    if not found:
        # Create new account entry
        new_acc = {
            "id": len(accounts_data.get("accounts", [])),
            "name": email,
            "cookie_1PSID": cookies.get("1PSID", ""),
            "cookie_1PSIDTS": cookies.get("1PSIDTS", ""),
            "is_active": True,
            "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
            "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        }
        accounts_data["accounts"].append(new_acc)
    
    with open(ACCOUNTS_PATH, "w", encoding="utf-8") as f:
        json.dump(accounts_data, f, indent=2, ensure_ascii=False)
    
    print(f"[SAVE] Da luu cookie vao {ACCOUNTS_PATH}")


async def auto_login(email: str, password: str, totp_secret: str = None, headless: bool = False, fresh: bool = False):
    """Main function to auto-login and get cookies"""
    
    # Create browser data directory
    BROWSER_DATA_PATH.mkdir(exist_ok=True)
    user_data_dir = BROWSER_DATA_PATH / email.replace("@", "_at_").replace(".", "_")
    
    # If fresh mode, delete old profile
    if fresh and user_data_dir.exists():
        import shutil
        shutil.rmtree(user_data_dir)
        print(f"[FRESH] Da xoa profile cu de tao moi")
    
    async with async_playwright() as p:
        # Launch browser with persistent context (saves session)
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox"
            ]
        )
        
        page = await browser.new_page()
        
        try:
            # Check if already logged in
            try:
                await page.goto(GEMINI_URL, timeout=60000)
            except:
                print("[WARN] Trang Gemini load cham, dang thu lai...")
                await asyncio.sleep(5)
            await asyncio.sleep(3)
            
            # Try to get cookies first
            cookies = await get_gemini_cookies(page)
            
            # If we got valid cookies, we're done
            if cookies.get("1PSID"):
                print("[OK] Da co session hop le!")
                update_accounts_json(email, cookies)
                return cookies
            
            # No valid cookies, need to login
            print("[RETRY] Session khong hop le, dang nhap lai...")
            await login_google(page, email, password, totp_secret)
            
            # Get cookies after login
            cookies = await get_gemini_cookies(page)
            
            if cookies.get("1PSID"):
                update_accounts_json(email, cookies)
                return cookies
            else:
                print("[FAIL] Khong lay duoc cookie!")
                return None
                
        finally:
            await browser.close()


async def main():
    parser = argparse.ArgumentParser(description="Auto-login Google and get Gemini cookies")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--fresh", action="store_true", help="Delete old profile and create new one (for testing)")
    parser.add_argument("--email", type=str, help="Specific email to login")
    args = parser.parse_args()
    
    credentials = load_credentials()
    if not credentials:
        return
    
    for account in credentials.get("accounts", []):
        email = account.get("email")
        password = account.get("password")
        totp_secret = account.get("totp_secret")
        
        if args.email and email != args.email:
            continue
        
        if not password or password == "NHAP_MAT_KHAU_VAO_DAY":
            print(f"[WARN] Ban chua nhap mat khau cho {email} trong credentials.json")
            continue
        
        print(f"\n{'='*50}")
        print(f"[START] Xu ly tai khoan: {email}")
        print(f"{'='*50}")
        
        result = await auto_login(email, password, totp_secret, headless=args.headless, fresh=args.fresh)
        
        if result:
            print(f"[OK] Hoan tat cho {email}!")
        else:
            print(f"[FAIL] That bai cho {email}!")


if __name__ == "__main__":
    asyncio.run(main())
    print("\n[DONE] Xong! Khoi dong lai server de su dung cookie moi.")
