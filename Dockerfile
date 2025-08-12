# OCR API Service Docker Image
FROM python:3.9-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製必要的程式碼和模型
COPY train_ocr_model_v3_improved.py .
COPY ocr_fastapi_server_v2.py .
COPY best_lightweight_crnn_model.pth .

# 創建非 root 用戶
RUN useradd -m -u 1000 ocr && chown -R ocr:ocr /app

# 切換到非 root 用戶
USER ocr

# 暴露端口
EXPOSE 8000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 設定環境變數
ENV PYTHONUNBUFFERED=1
ENV MAX_CONCURRENT_REQUESTS=20
ENV WORKERS=1
ENV DEVICE=cpu

# 啟動命令
CMD ["python", "ocr_fastapi_server_v2.py", "--host", "0.0.0.0", "--port", "8000"]