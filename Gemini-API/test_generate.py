"""
Test script to generate image and save to PNG
"""

import requests
import base64
import json

API_URL = "http://localhost:8000/v1/images/generations"
API_KEY = "sk-demo"

def generate_and_save(prompt: str, output_file: str = "output.png"):
    """Generate image and save to file."""

    print(f"Generating image: {prompt}")

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "prompt": prompt,
            "n": 1,
            "model": "gemini-2.5-flash"
        }
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    data = response.json()

    if not data.get("data"):
        print("No images in response")
        return

    # Get first image
    b64_json = data["data"][0].get("b64_json")
    revised_prompt = data["data"][0].get("revised_prompt", "")

    if not b64_json:
        print("No base64 data in response")
        return

    # Decode and save
    image_bytes = base64.b64decode(b64_json)

    with open(output_file, "wb") as f:
        f.write(image_bytes)

    print(f"Image saved to: {output_file}")
    print(f"Revised prompt: {revised_prompt[:100]}...")
    print(f"File size: {len(image_bytes)} bytes")


if __name__ == "__main__":
    generate_and_save(
        prompt="A cute cat playing with yarn ball, photorealistic",
        output_file="output.png"
    )
