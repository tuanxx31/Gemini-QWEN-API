
import requests
import json
import os

def test_api_edit():
    url = "http://localhost:8002/v1/images/edits"
    headers = {
        "Authorization": "Bearer sk-seedream-secret"
    }
    
    # Path to the test image we created earlier
    image_path = "test_large.jpg"
    if not os.path.exists(image_path):
        from PIL import Image
        # 2048 * 2048 = 4,194,304 (meets > 3,686,400 requirement)
        img = Image.new('RGB', (2048, 2048), color = 'red')
        img.save(image_path, 'JPEG')
        print(f"Created {image_path}")

    files = {
        'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')
    }
    
    data = {
        'prompt': 'A beautiful landscape with mountains',
        'size': '2048x2048',
        'response_format': 'url'
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {json.dumps(response.json(), indent=4)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api_edit()
