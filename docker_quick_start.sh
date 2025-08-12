#!/bin/bash

# 快速啟動 Docker 服務

echo "=========================================="
echo "OCR API Docker 快速啟動"
echo "模型準確率: 99.38%"
echo "=========================================="

# 檢查模型檔案
if [ ! -f "best_lightweight_crnn_model.pth" ]; then
    echo "錯誤: 找不到模型檔案"
    echo "請先訓練模型: ./quick_train_v3.sh"
    exit 1
fi

# 檢查 Docker
if ! command -v docker &> /dev/null; then
    echo "錯誤: Docker 未安裝"
    exit 1
fi

# 構建並啟動
echo "構建 Docker 映像..."
docker-compose build

echo ""
echo "啟動服務（4 個工作進程）..."
docker-compose up -d

# 等待服務啟動
echo "等待服務啟動..."
sleep 5

# 檢查服務
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    PUBLIC_IP=$(curl -s ifconfig.me)
    echo ""
    echo "✅ 服務啟動成功！"
    echo ""
    echo "訪問地址:"
    echo "  本地: http://localhost:8000"
    echo "  外網: http://${PUBLIC_IP}:8000"
    echo ""
    echo "測試命令:"
    echo "  curl -X POST -F 'file=@captcha.png' http://localhost:8000/predict"
    echo ""
    echo "查看日誌:"
    echo "  docker-compose logs -f"
    echo ""
    echo "停止服務:"
    echo "  docker-compose down"
else
    echo "❌ 服務啟動失敗"
    echo "查看日誌: docker-compose logs"
fi