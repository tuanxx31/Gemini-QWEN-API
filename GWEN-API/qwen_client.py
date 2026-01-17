"""
Qwen Chat API Client

Unofficial client for chat.qwen.ai API.
Supports text-to-image and image editing via chat interface.
"""

import aiohttp
import asyncio
import json
import uuid
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import base64
import os


@dataclass
class QwenImage:
    """Represents a generated image from Qwen."""

    url: str
    file_id: str

    async def save(
        self, path: str, filename: str, session: aiohttp.ClientSession
    ) -> str:
        """Download and save image to file."""
        filepath = os.path.join(path, filename)
        async with session.get(self.url) as resp:
            if resp.status == 200:
                with open(filepath, "wb") as f:
                    f.write(await resp.read())
        return filepath

    async def to_base64(self, session: aiohttp.ClientSession) -> str:
        """Download image and convert to base64."""
        async with session.get(self.url) as resp:
            if resp.status == 200:
                data = await resp.read()
                return base64.b64encode(data).decode("utf-8")
        return ""


@dataclass
class QwenResponse:
    """Response from Qwen chat completion."""

    text: str
    images: List[QwenImage]
    message_id: str
    chat_id: str


class QwenClient:
    """
    Client for Qwen Chat API.

    Usage:
        client = QwenClient(
            token="eyJhbGciOiJIUzI1NiI...",
            cookies={...}
        )
        await client.init()
        response = await client.generate_image("a cute cat")
        print(response.images)
    """

    BASE_URL = "https://chat.qwen.ai"
    API_URL = f"{BASE_URL}/api/v2/chat/completions"
    UPLOAD_URL = f"{BASE_URL}/api/v1/files/upload"

    def __init__(
        self,
        token: str,
        cnaui: str,
        aui: str,
        bx_ua: str = "",
        bx_umidtoken: str = "",
        chat_id: Optional[str] = None,
        extra_cookies: Optional[Dict[str, str]] = None,
        verbose: bool = False,  # Set to True to show debug logs
        silent: bool = True,  # Set to False to show info logs
    ):
        self.token = token
        self.cnaui = cnaui
        self.aui = aui
        self.bx_ua = bx_ua
        self.bx_umidtoken = bx_umidtoken
        self.chat_id = chat_id  # Existing chat_id to reuse
        self.extra_cookies = extra_cookies or {}
        self.verbose = verbose  # Control debug output
        self.silent = silent  # Control all output

        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False

    def _log(self, message: str, level: str = "info"):
        """Internal logging with level control."""
        if self.silent:
            return  # Skip all logs when silent
        if level == "debug" and not self.verbose:
            return  # Skip debug messages when not verbose
        print(message)

    @property
    def cookies(self) -> Dict[str, str]:
        """Build cookies dict."""
        cookies = {
            "token": self.token,
            "cnaui": self.cnaui,
            "aui": self.aui,
        }
        cookies.update(self.extra_cookies)
        return cookies

    @property
    def headers(self) -> Dict[str, str]:
        """Build headers dict."""
        return {
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/json",
            "Origin": self.BASE_URL,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
            "X-Accel-Buffering": "no",
            "bx-ua": self.bx_ua,
            "bx-umidtoken": self.bx_umidtoken,
            "bx-v": "2.5.36",
            "source": "web",
        }

    async def init(self, timeout: int = 120):
        """Initialize the client session."""
        if self._session is None or self._session.closed:
            from yarl import URL

            timeout_config = aiohttp.ClientTimeout(total=timeout)

            # Create cookie jar
            jar = aiohttp.CookieJar()

            # Set cookies before creating session
            base_url = URL(self.BASE_URL)
            for name, value in self.cookies.items():
                jar.update_cookies({name: value}, response_url=base_url)

            self._session = aiohttp.ClientSession(
                timeout=timeout_config, headers=self.headers, cookie_jar=jar
            )

            self._running = True
            self._log("[QwenClient] Initialized")

    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._running = False
            self._log("[QwenClient] Closed")

    def _generate_uuid(self) -> str:
        """Generate a UUID string."""
        return str(uuid.uuid4())

    async def create_chat(self, model: str = "qwen3-max-2025-09-23") -> str:
        """
        Get or create a dedicated chat for API usage.

        Qwen API requires a valid existing chat_id - cannot use random UUIDs.
        This looks for a chat named "API-GWEN" or uses the first available chat.

        Returns the chat_id for the chat.
        """
        if not self._session:
            raise RuntimeError("Client not initialized. Call init() first.")

        # Get existing chats - Qwen requires a real chat_id
        list_url = f"{self.BASE_URL}/api/v1/chats"

        try:
            async with self._session.get(list_url) as resp:
                if resp.status == 200:
                    chats = await resp.json()
                    if chats and len(chats) > 0:
                        # First, look for dedicated API chat
                        for chat in chats:
                            title = chat.get("title", "").lower()
                            if "api" in title or "gwen" in title:
                                chat_id = chat.get("id")
                                if chat_id:
                                    self._log(f"[QwenClient] Using API chat: {chat_id}")
                                    return chat_id

                        # If no API chat found, use the last chat (oldest/less used)
                        chat_id = chats[-1].get("id")
                        if chat_id:
                            self._log(f"[QwenClient] Using existing chat: {chat_id}")
                            return chat_id
        except Exception as e:
            self._log(f"[DEBUG] List chats error: {e}", "debug")

        # If no chats available, cannot proceed
        raise RuntimeError(
            "No existing chats found. Please create a chat at https://chat.qwen.ai first."
        )

    async def get_chat_parent_id(self, chat_id: str) -> Optional[str]:
        """
        Get the last message ID from a chat to use as parent_id.

        Returns the last message fid or None if chat is empty.
        """
        # For image generation, parent_id is optional - return None to start fresh
        return None

    async def upload_image(self, image_path: str) -> Dict[str, Any]:
        """
        Upload an image to Qwen storage using STS token and OSS SDK.

        Flow:
        1. Get STS token from /api/v2/files/getstsToken
        2. Upload to Alibaba OSS using oss2 SDK

        Returns dict with file info including id and url.
        """
        import oss2

        if not self._session:
            raise RuntimeError("Client not initialized. Call init() first.")

        filename = os.path.basename(image_path)

        with open(image_path, "rb") as f:
            file_data = f.read()

        file_size = len(file_data)

        # Step 1: Get STS token
        sts_url = f"{self.BASE_URL}/api/v2/files/getstsToken"
        sts_payload = {"filename": filename, "filesize": file_size, "filetype": "image"}

        self._log(
            f"[DEBUG] Getting STS token for {filename} ({file_size} bytes)", "debug"
        )

        async with self._session.post(sts_url, json=sts_payload) as resp:
            if resp.status != 200:
                raise Exception(
                    f"Get STS token failed: {resp.status} - {await resp.text()}"
                )

            sts_result = await resp.json()

            if not sts_result.get("success"):
                raise Exception(f"Get STS token failed: {sts_result}")

            sts_data = sts_result.get("data", {})

        # Step 2: Upload to OSS using oss2 SDK
        access_key_id = sts_data.get("access_key_id", "")
        access_key_secret = sts_data.get("access_key_secret", "")
        security_token = sts_data.get("security_token", "")
        bucket_name = sts_data.get("bucketname", "qwen-webui-prod")
        endpoint = sts_data.get("endpoint", "oss-accelerate.aliyuncs.com")
        file_path = sts_data.get("file_path", "")
        file_id = sts_data.get("file_id", self._generate_uuid())

        self._log(f"[DEBUG] Uploading to OSS bucket: {bucket_name}", "debug")

        # Create STS auth
        auth = oss2.StsAuth(access_key_id, access_key_secret, security_token)

        # Create bucket object
        bucket = oss2.Bucket(auth, f"https://{endpoint}", bucket_name)

        # Upload file (run in executor to not block async loop)
        import asyncio

        loop = asyncio.get_event_loop()

        def do_upload():
            result = bucket.put_object(file_path, file_data)
            return result

        result = await loop.run_in_executor(None, do_upload)

        self._log(f"[DEBUG] OSS upload status: {result.status}", "debug")

        if result.status not in [200, 201]:
            raise Exception(f"OSS upload failed: {result.status}")

        self._log(f"[QwenClient] Upload successful! file_id: {file_id}")

        # Return file info for use in message
        return {
            "id": file_id,
            "url": f"https://{bucket_name}.{endpoint}/{file_path}",
            "name": filename,
            "file_path": file_path,
        }

    async def generate_image(
        self,
        prompt: str,
        model: str = "qwen3-max-2025-09-23",
        image_path: Optional[str] = None,
        timeout: int = 120,
    ) -> QwenResponse:
        """
        Generate or edit an image.

        Args:
            prompt: Text prompt for generation/editing
            model: Model to use (default: qwen3-max-2025-09-23)
            image_path: Optional path to image for editing
            timeout: Request timeout in seconds

        Returns:
            QwenResponse with generated images
        """
        if not self._session:
            raise RuntimeError("Client not initialized. Call init() first.")

        # Use existing chat_id or get one from available chats
        if self.chat_id:
            chat_id = self.chat_id
            self._log(f"[QwenClient] Using provided chat_id: {chat_id}")
        else:
            chat_id = await self.create_chat(model)
            self._log(f"[QwenClient] Auto-selected chat_id: {chat_id}")

        # Get parent_id from chat history (last message ID)
        parent_id = await self.get_chat_parent_id(chat_id)
        if not parent_id:
            self._log(
                f"[DEBUG] No parent_id found, chat may be empty or error occurred",
                "debug",
            )
            parent_id = None  # Will be omitted from message

        message_id = self._generate_uuid()

        # Build files array if image provided
        files = []
        if image_path:
            # Upload image first
            upload_result = await self.upload_image(image_path)
            file_id = upload_result.get("id", self._generate_uuid())

            files = [
                {
                    "type": "image",
                    "file": upload_result,
                    "id": file_id,
                    "url": upload_result.get("url", ""),
                    "name": os.path.basename(image_path),
                    "file_type": "image/png",
                    "showType": "image",
                    "file_class": "vision",
                }
            ]

        # Build message
        message = {
            "fid": message_id,
            "parentId": parent_id,
            "childrenIds": [],
            "role": "user",
            "content": prompt,
            "user_action": "chat",
            "files": files,
            "timestamp": int(time.time()),
            "models": [model],
            "chat_type": "image_edit" if image_path else "t2i",
            "feature_config": {
                "thinking_enabled": False,
                "output_schema": "phase",
                "research_mode": "normal",
            },
            "extra": {"meta": {"subChatType": "image_edit" if image_path else "t2i"}},
            "sub_chat_type": "image_edit" if image_path else "t2i",
            "parent_id": parent_id,
        }

        # Build request payload - must include chat_id
        payload = {
            "stream": True,
            "version": "2.1",
            "incremental_output": True,
            "chat_id": chat_id,
            "chat_mode": "normal",
            "model": model,
            "parent_id": parent_id,
            "messages": [message],
            "timestamp": int(time.time()),
        }

        # Include chat_id in URL as query param
        url = f"{self.API_URL}?chat_id={chat_id}"

        headers = {
            **self.headers,
            "Referer": f"{self.BASE_URL}/c/{chat_id}",
            "X-Request-Id": self._generate_uuid(),
            "Timezone": time.strftime("%a %b %d %Y %H:%M:%S GMT%z"),
        }

        self._log(f"[QwenClient] Sending request to generate image...")

        images = []
        full_text = ""

        async with self._session.post(url, json=payload, headers=headers) as resp:
            self._log(f"[DEBUG] Response status: {resp.status}", "debug")
            self._log(f"[DEBUG] Response headers: {dict(resp.headers)}", "debug")

            if resp.status != 200:
                error_text = await resp.text()
                self._log(f"[DEBUG] Error response: {error_text[:500]}", "debug")
                raise Exception(f"API request failed: {resp.status} - {error_text}")

            # Read raw response for debugging
            raw_content = await resp.text()
            self._log(f"[DEBUG] Raw response length: {len(raw_content)}", "debug")
            if len(raw_content) > 0:
                self._log(f"[DEBUG] First 500 chars: {raw_content[:500]}", "debug")

            # Re-process as lines
            for line in raw_content.split("\n"):
                line = line.strip()

                if not line or not line.startswith("data:"):
                    continue

                try:
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        continue

                    data = json.loads(data_str)

                    # Debug: print all keys we receive
                    self._log(f"[DEBUG] Response keys: {list(data.keys())}", "debug")

                    # Check for content
                    if "choices" in data:
                        for choice in data["choices"]:
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_text += content

                                # Check if content contains image URL (cdn.qwenlm.ai)
                                if "cdn.qwenlm.ai" in content:
                                    # Extract the URL - content might be the URL directly
                                    img_url = content.strip()
                                    if img_url.startswith("http") and img_url not in [
                                        i.url for i in images
                                    ]:
                                        self._log(
                                            f"[DEBUG] Found image URL in content: {img_url[:100]}...",
                                            "debug",
                                        )
                                        images.append(
                                            QwenImage(
                                                url=img_url,
                                                file_id=self._generate_uuid(),
                                            )
                                        )

                    if "extra_info" in data:
                        extra = data["extra_info"]
                        self._log(
                            f"[DEBUG] extra_info keys: {list(extra.keys())}", "debug"
                        )

                        if "image_urls" in extra:
                            for img_url in extra["image_urls"]:
                                if img_url not in [i.url for i in images]:
                                    images.append(
                                        QwenImage(
                                            url=img_url, file_id=self._generate_uuid()
                                        )
                                    )

                        # Check wanx_info for generated images
                        if "wanx_info" in extra:
                            wanx = extra["wanx_info"]
                            self._log(
                                f"[DEBUG] wanx_info keys: {list(wanx.keys())}", "debug"
                            )
                            if "image_list" in wanx:
                                for img_data in wanx["image_list"]:
                                    url = img_data.get("url", "")
                                    if url and url not in [i.url for i in images]:
                                        images.append(
                                            QwenImage(
                                                url=url,
                                                file_id=img_data.get(
                                                    "file_id", self._generate_uuid()
                                                ),
                                            )
                                        )

                    # Check for images directly in response
                    if "images" in data:
                        for img in data["images"]:
                            if isinstance(img, str):
                                images.append(
                                    QwenImage(url=img, file_id=self._generate_uuid())
                                )
                            elif isinstance(img, dict):
                                url = img.get("url", img.get("image_url", ""))
                                if url:
                                    images.append(
                                        QwenImage(
                                            url=url,
                                            file_id=img.get(
                                                "id", self._generate_uuid()
                                            ),
                                        )
                                    )

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    self._log(f"[QwenClient] Error parsing response: {e}")
                    continue

        self._log(f"[QwenClient] Generated {len(images)} image(s)")

        return QwenResponse(
            text=full_text, images=images, message_id=message_id, chat_id=chat_id
        )

    async def __aenter__(self):
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Helper function to parse cookies from curl command
def parse_curl_cookies(curl_command: str) -> Dict[str, str]:
    """
    Parse cookies from a curl command string.

    Args:
        curl_command: Full curl command with -b flag

    Returns:
        Dict of cookie name -> value
    """
    import re

    # Find -b or --cookie flag content
    cookie_match = re.search(r'-b\s+"([^"]+)"', curl_command)
    if not cookie_match:
        cookie_match = re.search(r'-b\s+\^"([^"]+)\^"', curl_command)  # Windows escaped

    if not cookie_match:
        return {}

    cookie_string = cookie_match.group(1)
    cookies = {}

    for part in cookie_string.split("; "):
        if "=" in part:
            key, value = part.split("=", 1)
            cookies[key.strip()] = value.strip()

    return cookies


def parse_curl_headers(curl_command: str) -> Dict[str, str]:
    """
    Parse headers from a curl command string.

    Args:
        curl_command: Full curl command with -H flags

    Returns:
        Dict of header name -> value
    """
    import re

    headers = {}

    # Find all -H flags
    header_matches = re.findall(r'-H\s+\^?"([^"]+)\^?"', curl_command)

    for header in header_matches:
        if ":" in header:
            key, value = header.split(":", 1)
            headers[key.strip()] = value.strip()

    return headers
