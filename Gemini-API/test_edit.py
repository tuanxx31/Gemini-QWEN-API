import requests
import json
import os

API_URL = "http://192.168.88.180:8000/v1/images/edits"
API_KEY = "sk-demo"

def test_edit_image():
    # Giả định có một tệp ảnh tên là output.png từ bài kiểm tra trước
    image_path = "output.png"
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy {image_path}. Vui lòng chạy test_generate.py trước.")
        return

    print(f"Đang gửi yêu cầu chỉnh sửa ảnh: {image_path}")
    
    files = {
        'image': ('image.png', open(image_path, 'rb'), 'image/png')
    }
    
    data = {
        'prompt': 'make the cat wear a small red hat',
        'n': 1,
        'size': '1024x1024',
        'response_format': 'url'
    }
    
    headers = {
        'Authorization': f'Bearer {API_KEY}'
    }

    try:
        response = requests.post(API_URL, headers=headers, files=files, data=data)
        
        if response.status_code == 200:
            print("Thành công!")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Lỗi {response.status_code}:")
            print(response.text)
    except Exception as e:
        print(f"Yêu cầu thất bại: {e}")

if __name__ == "__main__":
    test_edit_image()
