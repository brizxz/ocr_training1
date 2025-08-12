# OCR 訓練與部署系統

高準確率驗證碼識別系統，基於 CRNN + CTC 架構，達到 **99.38%** 驗證準確率。

## 🚀 系統特點

- **高準確率**: 99.38% 驗證準確率
- **快速推理**: 7ms (CPU) / 2ms (GPU)
- **高並發**: 支援 10-20 個並發請求
- **外網訪問**: 支援公網 IP 直接訪問
- **Docker 部署**: 容器化部署，防止意外中斷

## 📂 專案結構

```
ocr_training1/
├── 核心程式
│   ├── train_ocr_model_v3_improved.py  # 訓練腳本（解決過擬合）
│   ├── ocr_fastapi_server_v2.py        # API 服務（高並發版）
│   └── test_concurrent_client.py       # 並發測試工具
├── 快速啟動
│   ├── start_server.sh                 # API 服務啟動器
│   ├── quick_train_v3.sh              # 快速訓練腳本
│   ├── quick_test.sh                   # API 測試腳本
│   └── run_training.sh                # 訓練選單腳本
├── 部署檔案
│   ├── Dockerfile                      # Docker 映像定義
│   ├── docker-compose.yml             # Docker Compose 配置
│   └── requirements.txt               # Python 依賴
├── 文檔
│   ├── DEPLOYMENT_GUIDE.md            # 部署指南
│   ├── TRAINING_PRINCIPLE.md          # 訓練原理說明
│   └── TRAINING_RECOMMENDATIONS.md    # 訓練建議
├── 資料
│   └── captcha_auto_label/            # 標註資料目錄
│       └── merged_20250811_155009/    # 合併的訓練資料
└── 模型
    └── best_lightweight_crnn_model.pth # 訓練好的模型

```

## 🎯 快速開始

### 1. 環境設置

```bash
# 安裝依賴
pip install -r requirements.txt
```

### 2. 訓練模型（可選，已有預訓練模型）

```bash
# 使用互動式腳本
./quick_train_v3.sh

# 或直接執行
python train_ocr_model_v3_improved.py \
    --data_dir captcha_auto_label/merged_20250811_155009 \
    --labels captcha_auto_label/merged_20250811_155009/training_data.txt \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0005
```

### 3. 啟動 API 服務

#### 方法 A: 直接啟動
```bash
# 使用啟動腳本
./start_server.sh
# 選擇 2 (外網訪問) 或 3 (生產環境)

# 或直接執行
python ocr_fastapi_server_v2.py --host 0.0.0.0 --port 8000
```

#### 方法 B: Docker 部署（推薦）
```bash
# 構建並啟動
docker-compose up -d

# 查看日誌
docker-compose logs -f

# 停止服務
docker-compose down
```

### 4. 測試服務

```bash
# 本地測試
./quick_test.sh

# 外網測試（替換 YOUR_IP）
curl -X POST -F 'file=@captcha.png' http://YOUR_IP:8000/predict

# 並發測試
python test_concurrent_client.py --benchmark
```

## 📡 API 端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/` | GET | 服務信息 |
| `/health` | GET | 健康檢查 |
| `/stats` | GET | 統計信息 |
| `/status` | GET | 即時狀態 |
| `/predict` | POST | 預測圖片（支援並發） |

## 💻 使用範例

### Python
```python
import requests

# 單個請求
with open('captcha.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    result = response.json()
    print(f"預測: {result['result']}")
    print(f"信心: {result['confidence']}")
```

### 並發請求
```python
import asyncio
import aiohttp

async def predict_async(session, url, image_path):
    with open(image_path, 'rb') as f:
        async with session.post(url, data={'file': f}) as resp:
            return await resp.json()

# 同時發送多個請求
async def main():
    url = "http://localhost:8000/predict"
    async with aiohttp.ClientSession() as session:
        tasks = [
            predict_async(session, url, f'captcha_{i}.png')
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        for r in results:
            print(r)

asyncio.run(main())
```

## 🐳 Docker 部署

### 使用 Docker Compose（推薦）
```bash
# 啟動服務（背景執行）
docker-compose up -d

# 擴展到多個實例
docker-compose up -d --scale ocr-api=3

# 查看狀態
docker-compose ps

# 查看日誌
docker-compose logs -f ocr-api

# 重啟服務
docker-compose restart

# 停止並移除
docker-compose down
```

### 手動 Docker 操作
```bash
# 構建映像
docker build -t ocr-api:latest .

# 執行容器
docker run -d \
    --name ocr-api \
    -p 8000:8000 \
    --restart unless-stopped \
    ocr-api:latest
```

## 🌐 外網訪問

1. **開放防火牆端口**
```bash
# Ubuntu
sudo ufw allow 8000

# CentOS
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

2. **獲取公網 IP**
```bash
curl ifconfig.me
```

3. **訪問服務**
```
http://YOUR_PUBLIC_IP:8000
```

## 📊 性能指標

- **模型準確率**: 99.38%
- **單次推理**: 2-10ms
- **並發處理**: 10-20 req/s
- **最大並發**: 20 個請求
- **記憶體使用**: ~500MB
- **CPU 使用**: ~10-30%

## 🔧 配置說明

### 環境變數
```bash
# 設定最大並發數
export MAX_CONCURRENT_REQUESTS=20

# 設定工作進程數
export WORKERS=4

# 設定設備
export DEVICE=cuda  # 或 cpu
```

### 訓練參數
- `--epochs`: 訓練回合數（預設 50）
- `--batch_size`: 批次大小（預設 32）
- `--lr`: 學習率（預設 0.0005）
- `--device`: 訓練設備（cuda/cpu）

### API 參數
- `--host`: 監聽地址（0.0.0.0 允許外網）
- `--port`: 服務端口（預設 8000）
- `--workers`: 工作進程數（預設 1）

## 🛠️ 疑難排解

### 問題：無法連接服務
- 檢查防火牆設定
- 確認服務正在運行：`docker-compose ps`
- 檢查端口占用：`netstat -tulpn | grep 8000`

### 問題：推理速度慢
- 使用 GPU：`--device cuda`
- 增加工作進程：`--workers 4`
- 使用 Docker 部署避免 Python GIL 限制

### 問題：記憶體不足
- 減少批次大小
- 減少最大並發數
- 使用 Docker 限制記憶體：`docker run -m 2g`

## 📈 監控

### 查看服務狀態
```bash
# API 端點
curl http://localhost:8000/health
curl http://localhost:8000/stats
curl http://localhost:8000/status

# Docker 日誌
docker-compose logs -f --tail=100

# 系統資源
docker stats ocr-api
```

### Prometheus 監控（可選）
API 服務提供 `/metrics` 端點供 Prometheus 採集。

## 🔄 更新維護

### 更新模型
```bash
# 訓練新模型
./quick_train_v3.sh

# 替換模型檔案
cp best_lightweight_crnn_model.pth docker/

# 重建並重啟
docker-compose build
docker-compose up -d
```

### 備份
```bash
# 備份模型和資料
tar -czf backup_$(date +%Y%m%d).tar.gz \
    best_lightweight_crnn_model.pth \
    captcha_auto_label/

# 備份 Docker 卷
docker run --rm \
    -v ocr_data:/data \
    -v $(pwd):/backup \
    alpine tar czf /backup/docker_backup.tar.gz /data
```

## 📝 授權

MIT License

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📧 聯絡

如有問題請開 Issue 討論。