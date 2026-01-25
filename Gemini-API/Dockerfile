# Sử dụng Python image chính thức làm base
FROM python:3.11-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho Playwright và Chromium
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt các dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cài đặt Chromium cho Playwright
RUN playwright install chromium

# Copy toàn bộ code vào container
COPY . .

# Tạo thư mục data nếu chưa có
RUN mkdir -p /app/data

# Khai báo Volume để lưu trữ dữ liệu bền vững (accounts, cookies, browser_data, v.v.)
VOLUME ["/app/data"]

# Mở port cho API
EXPOSE 8000

# Thiết lập biến môi trường để Python không cache file .pyc
ENV PYTHONUNBUFFERED=1

# Lệnh khởi chạy server (chạy trực tiếp api_server.py)
CMD ["python", "api_server.py"]
