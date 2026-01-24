import requests
import json

def test_chat():
    url = "http://127.0.0.1:8989/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-demo",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gemini-2.0-flash",
        "messages": [
            {"role": "user", "content": "Chào bạn, hãy giới thiệu về bản thân bạn đi?"}
        ]
    }
    
    print(f"Sending chat request to {url}...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        print(f"HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print("\nAssistant's Response:")
            print("-" * 50)
            print(content)
            print("-" * 50)
            print(f"\nTest PASSED")
        else:
            print(f"Test FAILED: {response.text}")
            
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    test_chat()
