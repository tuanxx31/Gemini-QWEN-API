import asyncio
import json
from pathlib import Path
from gemini_webapi import GeminiClient

async def check_cookies():
    accounts_path = Path("accounts.json")
    if not accounts_path.exists():
        print("❌ Error: accounts.json not found")
        return

    try:
        with open(accounts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading accounts.json: {e}")
        return
    
    accounts = data.get("accounts", [])
    if not accounts:
        print("⚠️ No accounts found in accounts.json")
        return

    print(f"🔍 Testing {len(accounts)} account(s)...\n")
    
    for acc in accounts:
        name = acc.get("name", "Unknown")
        psid = acc.get("cookie_1PSID", "")
        psidts = acc.get("cookie_1PSIDTS", "")
        
        print(f"Testing Account: {name}")
        
        if not psid:
            print(f"  ❌ Status: [MISSING] Cookie __Secure-1PSID is empty\n")
            continue

        client = GeminiClient(secure_1psid=psid, secure_1psidts=psidts if psidts else None)
        
        try:
            # init will attempt to get SNlM0e token, which verifies cookies
            await client.init(verbose=False)
            if client._running:
                print(f"  ✅ Status: [ACTIVE] Cookies are working perfectly.")
                # Optional: Test a simple heartbeat or message here if needed
            else:
                print(f"  ❌ Status: [INACTIVE] Client failed to start.")
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                print(f"  ❌ Status: [EXPIRED] Cookies are invalid or expired.")
            else:
                print(f"  ❌ Status: [ERROR] {error_msg}")
        finally:
            if client._running:
                await client.close()
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(check_cookies())
