
import aiohttp
import asyncio
import json
import uuid
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import os
from loguru import logger

@dataclass
class SeedreamImage:
    """Represents a generated image from Seedream."""
    url: str
    id: str

@dataclass
class SeedreamResponse:
    """Response from Seedream generation."""
    images: List[SeedreamImage]
    message_id: str
    text: str = ""

class SeedreamClient:
    """
    Client for BytePlus Seedream 4.5 API (Backend for Frontend).
    """
    BASE_URL = "https://modelark-api.console.byteplus.com"
    GENERATE_URL = f"{BASE_URL}/ark/bff/api/ap-southeast-1/2024/CreateImageGeneration"
    
    def __init__(
        self, 
        cookie: str,
        csrf_token: str,
        web_id: str,
        verbose: bool = False
    ):
        self.cookie = cookie
        self.csrf_token = csrf_token
        self.web_id = web_id
        self.verbose = verbose
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False

    def _log(self, message: str, level: str = "info"):
        if level == "debug" and not self.verbose:
            return
        if level == "error":
            logger.error(message)
        elif level == "success":
            logger.success(message)
        else:
            logger.info(message)

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Cookie": self.cookie,
            "Origin": "https://www.byteplus.com",
            "Referer": "https://www.byteplus.com/",
            "Sec-Ch-Ua": '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
            "X-Csrf-Token": self.csrf_token,
            "X-Web-Id": self.web_id,
        }

    async def init(self):
        """Initialize the client session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self.headers)
            self._running = True

    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._running = False

    @property
    def upload_headers(self) -> Dict[str, str]:
        """Headers specifically for the UploadArkTosFile endpoint to match browser."""
        return {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Cookie": self.cookie,
            "Origin": "https://www.byteplus.com",
            "Referer": "https://www.byteplus.com/en/ai-playground/media?mode=vision&modelId=seedream-4-5-251128&tab=GenImage",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0",
            "X-Csrf-Token": self.csrf_token,
            "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Microsoft Edge";v="144"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
        }

    async def _upload_image(self, image_path: str) -> Dict[str, Any]:
        """Uploads a local image to BytePlus storage and returns the full metadata."""
        if not self._session:
            await self.init()

        upload_url = "https://arkbff-ap-southeast1.console.byteplus.com/api/2024-10-01/UploadArkTosFile?"
        
        
        data = aiohttp.FormData()
        with open(image_path, "rb") as f:
            img_data = f.read()
        
        data.add_field('biz', 'experience')
        data.add_field('webId', self.web_id)
        data.add_field('file', img_data, filename=os.path.basename(image_path), content_type='image/jpeg')

        async with self._session.post(upload_url, headers=self.upload_headers, data=data) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise Exception(f"Upload failed: {resp.status} - {text}")
            
            try:
                res_json = json.loads(text)
                result = res_json.get("Result", {})
                if not result or (not result.get("Url") and not result.get("Binary")):
                    raise Exception(f"Upload response missing result data: {text}")
                
                return result
            except Exception as e:
                raise Exception(f"Failed to parse upload response: {e}. Raw: {text}")

    async def generate_image(
        self, 
        prompt: str, 
        width: int = 1024, 
        height: int = 1024,
        model: str = "seedream-4-5",
        endpoint_id: str = "seedream-4-5-251128",
        image_path: Optional[str] = None,
        timeout: int = 120
    ) -> SeedreamResponse:
        """
        Generate or edit images using Seedream. Handles streaming responses.
        """
        if not self._session:
            await self.init()

        images_payload = []
        if image_path:
            # Step 1: Upload and get metadata
            upload_res = await self._upload_image(image_path)
            
            # Replicate playground payload structure
            images_payload = [{
                "BucketName": upload_res.get("Bucket"),
                "ObjectKey": upload_res.get("Prefix"),
                "Url": upload_res.get("Url") or upload_res.get("Binary")
            }]

        payload = {
            "From": "pc",
            "Model": model,
            "EndpointId": endpoint_id,
            "Type": "group", # Playground uses "group" for both gen and edit
            "Images": images_payload,
            "Prompt": prompt,
            "Ratio": "1:1",
            "Size": f"{width}x{height}",
            "Watermark": False
        }

        if image_path:
            payload["ImageCount"] = 1
        else:
            payload["ImageSequence"] = 1

        gcd = self._gcd(width, height)
        payload["Ratio"] = f"{width//gcd}:{height//gcd}"
        
        async with self._session.post(self.GENERATE_URL, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Generation failed: {resp.status} - {error_text}")

            final_data = None
            error_data = None
            all_images = []
            
            async for line in resp.content:
                if not line: continue
                line_str = line.decode("utf-8").strip()
                if not line_str: continue
                
                if line_str.startswith("data:"):
                    data_str = line_str[5:].strip()
                    if data_str == "[DONE]": break
                    
                    try:
                        chunk = json.loads(data_str)
                        if "Error" in chunk or "err_msg" in chunk or "ErrMsg" in chunk or "msg" in chunk:
                            # Catch msg if it's an error chunk (like what we saw in logs)
                            if "msg" in chunk and ("code" in chunk or "InvalidParameter" in str(chunk) or "error" in line_str):
                                error_data = chunk
                                break

                        # Look for Images in Result, items, or directly in chunk
                        res_part = chunk.get("Result", {})
                        imgs = res_part.get("Images") or chunk.get("Images") or chunk.get("items") or res_part.get("items")
                        if imgs:
                            all_images.extend(imgs)
                            final_data = chunk # Track the last chunk with images
                    except:
                        pass

            if error_data:
                err = error_data.get("Error") or error_data.get("ErrMsg") or error_data.get("err_msg") or error_data.get("msg")
                raise Exception(f"Stream error: {err}")

            if not all_images:
                raise Exception("No images found in stream. Check logs for raw line data.")

            # Construct the final response from collected images
            images = []
            for img in all_images:
                url = img.get("Url") or img.get("Binary")
                if url:
                    images.append(SeedreamImage(url=url, id=str(uuid.uuid4())))
            
            return SeedreamResponse(images=images, message_id=str(uuid.uuid4()))

    def _parse_result(self, data: Dict[str, Any]) -> SeedreamResponse:
        """Helper to parse the Result dict into SeedreamResponse."""
        res_data = data.get("Result", {})
        img_list = res_data.get("Images", [])
        
        images = []
        for img in img_list:
            url = img.get("Url")
            if url:
                images.append(SeedreamImage(url=url, id=str(uuid.uuid4())))

        if not images:
            raise Exception(f"No images returned in Result. Response: {data}")

        return SeedreamResponse(images=images, message_id=str(uuid.uuid4()))

    async def _poll_task_status(self, task_id: str, timeout: int) -> SeedreamResponse:
        """Poll for task result if the API is asynchronous."""
        # This is a placeholder as the exact polling endpoint for BFF is often /GetTaskResult or similar
        poll_url = f"{self.BASE_URL}/ark/bff/api/ap-southeast-1/2024/GetImageGenerationResult"
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            async with self._session.post(poll_url, json={"TaskId": task_id}) as resp:
                if resp.status != 200:
                    await asyncio.sleep(3)
                    continue
                
                data = await resp.json()
                res_data = data.get("Result", {})
                status = res_data.get("Status", "").lower()
                
                if status == "success":
                    images = []
                    for img in res_data.get("Images", []):
                        url = img.get("Url")
                        if url:
                            images.append(SeedreamImage(url=url, id=str(uuid.uuid4())))
                    return SeedreamResponse(images=images, message_id=task_id)
                elif status == "failed":
                    raise Exception(f"Seedream generation failed: {res_data.get('ErrMsg')}")
                
                self._log(f"[SeedreamClient] Status: {status}...", "debug")
                
            await asyncio.sleep(3)
            
        raise Exception(f"Task {task_id} timed out after {timeout}s")

    def _gcd(self, a, b):
        import math
        return math.gcd(a, b)

    async def __aenter__(self):
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
