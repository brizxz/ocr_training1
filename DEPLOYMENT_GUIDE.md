# OCR API 部署指南

## 🎉 恭喜！模型訓練成功
- **驗證準確率**: 99.38%
- **訓練準確率**: 97.92%
- **推理速度**: ~7ms (CPU) / ~2ms (GPU)

## 🚀 快速啟動

### 1. 啟動 API 服務（v2 高並發版本）

```bash
# 基本啟動（允許外網訪問）
python ocr_fastapi_server_v2.py --host 0.0.0.0 --port 8000

# 生產環境（多進程）
python ocr_fastapi_server_v2.py --host 0.0.0.0 --port 8000 --workers 4
```

### 2. 測試服務

```bash
# 本地測試
curl -X POST -F 'file=@captcha.png' http://localhost:8000/predict

# 外網測試（替換為你的公網 IP）
curl -X POST -F 'file=@captcha.png' http://YOUR_PUBLIC_IP:8000/predict
```

## 📡 外網訪問配置

### 方法 1: 直接暴露端口（適合測試）

1. **開放防火牆端口**
```bash
# Ubuntu/Debian
sudo ufw allow 8000

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

2. **確認公網 IP**
```bash
curl ifconfig.me
```

3. **訪問服務**
```
http://YOUR_PUBLIC_IP:8000
```

### 方法 2: 使用 Nginx 反向代理（推薦生產環境）

1. **安裝 Nginx**
```bash
sudo apt install nginx  # Ubuntu/Debian
sudo yum install nginx  # CentOS/RHEL
```

2. **配置 Nginx**
```nginx
# /etc/nginx/sites-available/ocr-api
server {
    listen 80;
    server_name your-domain.com;  # 或使用 IP
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # 超時設定
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

3. **啟用配置**
```bash
sudo ln -s /etc/nginx/sites-available/ocr-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 方法 3: 使用 Ngrok（臨時測試）

```bash
# 安裝 ngrok
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
unzip ngrok-stable-linux-amd64.zip

# 啟動隧道
./ngrok http 8000
```

## 🔧 API 使用說明

### 並發請求處理

新版 API 專門優化了並發處理，支援多個客戶端同時請求：

```python
import asyncio
import aiohttp

async def predict_concurrent():
    """同時發送多個預測請求"""
    url = "http://your-server:8000/predict"
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(10):  # 10 個並發請求
            with open(f'captcha_{i}.png', 'rb') as f:
                task = session.post(url, data={'file': f})
                tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        for resp in responses:
            result = await resp.json()
            print(result)

# 執行
asyncio.run(predict_concurrent())
```

### 測試並發性能

```bash
# 使用測試腳本
python test_concurrent_client.py --url http://localhost:8000/predict --concurrent 10

# 執行基準測試
python test_concurrent_client.py --url http://localhost:8000/predict --benchmark
```

## 📊 性能優化

### 1. 系統層面
```bash
# 增加文件描述符限制
ulimit -n 65535

# 優化 TCP 設定
echo "net.core.somaxconn = 1024" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 2048" >> /etc/sysctl.conf
sysctl -p
```

### 2. 服務配置
```python
# 調整最大並發數（在 ocr_fastapi_server_v2.py 中）
max_concurrent_requests = 20  # 根據服務器能力調整

# 使用多進程模式
python ocr_fastapi_server_v2.py --workers 4  # CPU 核心數
```

## 🐳 Docker 部署

1. **創建 Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製模型和代碼
COPY best_lightweight_crnn_model.pth .
COPY ocr_fastapi_server_v2.py .
COPY train_ocr_model_v3_improved.py .

# 暴露端口
EXPOSE 8000

# 啟動服務
CMD ["python", "ocr_fastapi_server_v2.py", "--host", "0.0.0.0", "--port", "8000"]
```

2. **構建和運行**
```bash
docker build -t ocr-api .
docker run -d -p 8000:8000 --name ocr-service ocr-api
```

## 🔒 安全建議

1. **使用 HTTPS**
   - 配置 SSL 證書（Let's Encrypt）
   - 通過 Nginx 處理 SSL

2. **添加認證**
```python
# 在 API 中添加簡單的 API Key 認證
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    if credentials.credentials != "your-secret-key":
        raise HTTPException(status_code=401, detail="Invalid API Key")
    # ... 處理預測
```

3. **限流**
```python
# 使用 slowapi 限流
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("100/minute")  # 每分鐘 100 次
async def predict(request: Request, file: UploadFile = File(...)):
    # ... 處理預測
```

## 📈 監控

### 查看服務狀態
```bash
# 健康檢查
curl http://localhost:8000/health

# 查看統計
curl http://localhost:8000/stats

# 即時狀態
curl http://localhost:8000/status
```

### 日誌監控
```bash
# 查看服務日誌
tail -f ocr_api.log

# 使用 systemd
journalctl -u ocr-api -f
```

## 🚦 Systemd 服務配置

創建 `/etc/systemd/system/ocr-api.service`:

```ini
[Unit]
Description=OCR API Service
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/ocr_training1
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
ExecStart=/usr/bin/python3 /path/to/ocr_training1/ocr_fastapi_server_v2.py --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

啟動服務：
```bash
sudo systemctl daemon-reload
sudo systemctl enable ocr-api
sudo systemctl start ocr-api
sudo systemctl status ocr-api
```

## 📝 API 端點總覽

| 端點 | 方法 | 說明 |
|------|------|------|
| `/` | GET | 服務信息 |
| `/health` | GET | 健康檢查 |
| `/stats` | GET | 統計信息 |
| `/status` | GET | 即時狀態 |
| `/predict` | POST | 預測單張圖片（支援並發） |

## 🎯 使用範例

### Python 客戶端
```python
import requests

# 單個請求
with open('captcha.png', 'rb') as f:
    response = requests.post(
        'http://your-server:8000/predict',
        files={'file': f}
    )
    print(response.json())
```

### JavaScript 客戶端
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://your-server:8000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

### cURL
```bash
curl -X POST -F 'file=@captcha.png' http://your-server:8000/predict
```

## ⚡ 效能指標

根據你的訓練結果和服務配置：
- **準確率**: 99.38%
- **單次推理**: 2-10ms
- **並發處理**: 10-20 req/s
- **最大並發**: 20 個請求

## 🆘 疑難排解

### 連接被拒絕
- 檢查防火牆設定
- 確認服務正在運行
- 檢查綁定地址是否為 0.0.0.0

### 503 服務忙碌
- 增加 max_concurrent_requests
- 使用多進程模式
- 升級服務器資源

### 推理速度慢
- 使用 GPU
- 減少模型大小
- 啟用批次處理