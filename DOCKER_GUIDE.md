# Docker 部署指南

## 🚀 快速開始（5分鐘部署）

```bash
# 1. 快速啟動
./docker_quick_start.sh

# 2. 測試服務
curl -X POST -F 'file=@captcha.png' http://localhost:8000/predict
```

就這麼簡單！服務已在背景執行，不會因為終端關閉而中斷。

## 📦 Docker 優勢

### 為什麼使用 Docker？
1. **防止意外中斷** - 容器在背景持續運行
2. **自動重啟** - 系統重啟後自動恢復服務
3. **資源隔離** - 不影響主機環境
4. **易於擴展** - 輕鬆擴展到多個實例
5. **一鍵部署** - 無需手動安裝依賴

## 🛠️ 詳細操作

### 1. 構建映像
```bash
docker-compose build
```

### 2. 啟動服務

#### 單實例（開發環境）
```bash
docker-compose up -d
```

#### 多實例（生產環境）
```bash
# 啟動 4 個實例，自動負載均衡
docker-compose up -d --scale ocr-api=4
```

### 3. 管理服務

```bash
# 查看狀態
docker-compose ps

# 查看日誌
docker-compose logs -f

# 停止服務
docker-compose stop

# 重啟服務
docker-compose restart

# 完全移除
docker-compose down
```

### 4. 使用管理腳本

```bash
# 互動式管理選單
./docker_deploy.sh

# 命令列模式
./docker_deploy.sh start      # 啟動
./docker_deploy.sh stop       # 停止
./docker_deploy.sh status     # 狀態
./docker_deploy.sh logs       # 日誌
./docker_deploy.sh test       # 測試
```

## 🔧 配置說明

### 環境變數
編輯 `docker-compose.yml` 中的環境變數：

```yaml
environment:
  - MAX_CONCURRENT_REQUESTS=20  # 最大並發數
  - WORKERS=4                   # 工作進程數
  - DEVICE=cpu                  # 使用設備 (cpu/cuda)
```

### 資源限制
```yaml
deploy:
  resources:
    limits:
      cpus: '2'        # 最大 CPU 使用
      memory: 2G       # 最大記憶體
```

### 端口映射
```yaml
ports:
  - "8000:8000"  # 主機端口:容器端口
```

## 📊 監控與維護

### 即時監控
```bash
# 資源使用
docker stats ocr-api

# 健康檢查
watch -n 2 'curl -s http://localhost:8000/health | jq'

# 服務統計
curl http://localhost:8000/stats | jq
```

### 日誌管理
```bash
# 查看最近 100 行
docker-compose logs --tail=100

# 持續監控
docker-compose logs -f

# 導出日誌
docker-compose logs > ocr_logs_$(date +%Y%m%d).txt
```

### 備份
```bash
# 使用管理腳本備份
./docker_deploy.sh backup

# 手動備份
tar -czf backup_$(date +%Y%m%d).tar.gz \
    best_lightweight_crnn_model.pth \
    docker-compose.yml \
    Dockerfile
```

## 🌐 生產環境部署

### 1. 使用 Nginx 負載均衡

```bash
# 啟動包含 Nginx 的完整堆疊
docker-compose --profile production up -d
```

### 2. HTTPS 配置

1. 準備 SSL 證書：
```bash
mkdir ssl
cp your-cert.pem ssl/cert.pem
cp your-key.pem ssl/key.pem
```

2. 修改 `nginx.conf` 中的域名

3. 重啟服務：
```bash
docker-compose restart nginx
```

### 3. 自動擴展

根據負載自動調整實例數：
```bash
# 增加實例
docker-compose up -d --scale ocr-api=6

# 減少實例
docker-compose up -d --scale ocr-api=2
```

## 🐛 疑難排解

### 問題：容器無法啟動
```bash
# 查看詳細錯誤
docker-compose logs ocr-api

# 檢查配置
docker-compose config

# 重新構建
docker-compose build --no-cache
```

### 問題：無法連接服務
```bash
# 檢查容器是否運行
docker ps | grep ocr-api

# 檢查端口
netstat -tulpn | grep 8000

# 測試容器內部
docker exec ocr-api curl http://localhost:8000/health
```

### 問題：性能不佳
```bash
# 增加實例數
docker-compose up -d --scale ocr-api=4

# 分配更多資源（修改 docker-compose.yml）
# 重啟服務
docker-compose up -d
```

## 📈 性能優化

### 1. 使用 BuildKit 加速構建
```bash
DOCKER_BUILDKIT=1 docker-compose build
```

### 2. 多階段構建（減小映像大小）
已在 Dockerfile 中優化

### 3. 健康檢查優化
```yaml
healthcheck:
  interval: 30s   # 檢查間隔
  timeout: 10s    # 超時時間
  retries: 3      # 重試次數
```

## 🔄 CI/CD 整合

### GitHub Actions 範例
```yaml
name: Deploy OCR API

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build and Push
        run: |
          docker build -t ocr-api:latest .
          docker save ocr-api:latest | ssh user@server docker load
          
      - name: Deploy
        run: |
          ssh user@server "cd /app && docker-compose up -d"
```

## 🎯 最佳實踐

1. **定期備份模型和日誌**
2. **監控資源使用**
3. **設定自動重啟策略**
4. **使用健康檢查**
5. **限制資源使用**
6. **定期更新基礎映像**

## 📝 常用命令速查

```bash
# 一鍵啟動
./docker_quick_start.sh

# 查看所有容器
docker ps -a

# 進入容器
docker exec -it ocr-api bash

# 複製檔案到容器
docker cp file.txt ocr-api:/app/

# 從容器複製檔案
docker cp ocr-api:/app/logs.txt ./

# 清理未使用資源
docker system prune -a

# 查看映像大小
docker images | grep ocr-api

# 導出/導入映像
docker save ocr-api:latest > ocr-api.tar
docker load < ocr-api.tar
```

## 🆘 需要幫助？

遇到問題時：
1. 查看日誌：`docker-compose logs`
2. 檢查狀態：`docker-compose ps`
3. 運行測試：`./docker_deploy.sh test`
4. 查看本指南的疑難排解部分