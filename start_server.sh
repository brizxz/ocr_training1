#!/bin/bash

echo "==========================================="
echo "OCR API 服務啟動器"
echo "模型準確率: 99.38%"
echo "==========================================="
echo ""

# 檢查模型檔案
if [ ! -f "best_lightweight_crnn_model.pth" ]; then
    echo "錯誤: 找不到模型檔案 best_lightweight_crnn_model.pth"
    echo "請先執行訓練腳本: ./quick_train_v3.sh"
    exit 1
fi

echo "請選擇啟動模式:"
echo "  1. 本地開發 (localhost only)"
echo "  2. 外網訪問 (0.0.0.0) - 單進程"
echo "  3. 生產環境 (0.0.0.0) - 多進程"
echo "  4. 自訂配置"
echo ""
read -p "請選擇 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "啟動本地開發服務..."
        echo "訪問地址: http://localhost:8000"
        echo ""
        python ocr_fastapi_server_v2.py --host 127.0.0.1 --port 8000
        ;;
    2)
        echo ""
        echo "啟動外網訪問服務（單進程）..."
        echo "獲取公網 IP..."
        PUBLIC_IP=$(curl -s ifconfig.me)
        echo "本地訪問: http://localhost:8000"
        echo "外網訪問: http://${PUBLIC_IP}:8000"
        echo ""
        echo "注意: 請確保防火牆已開放 8000 端口"
        echo "Ubuntu: sudo ufw allow 8000"
        echo "CentOS: sudo firewall-cmd --add-port=8000/tcp"
        echo ""
        read -p "按 Enter 繼續..."
        python ocr_fastapi_server_v2.py --host 0.0.0.0 --port 8000
        ;;
    3)
        echo ""
        echo "啟動生產環境服務（多進程）..."
        read -p "工作進程數 (建議 2-4，預設 4): " workers
        workers=${workers:-4}
        
        PUBLIC_IP=$(curl -s ifconfig.me)
        echo ""
        echo "本地訪問: http://localhost:8000"
        echo "外網訪問: http://${PUBLIC_IP}:8000"
        echo "工作進程: ${workers}"
        echo ""
        echo "注意: 請確保防火牆已開放 8000 端口"
        echo ""
        read -p "按 Enter 繼續..."
        python ocr_fastapi_server_v2.py --host 0.0.0.0 --port 8000 --workers $workers
        ;;
    4)
        echo ""
        echo "自訂配置"
        read -p "Host (預設 0.0.0.0): " host
        host=${host:-0.0.0.0}
        read -p "Port (預設 8000): " port
        port=${port:-8000}
        read -p "Workers (預設 1): " workers
        workers=${workers:-1}
        
        echo ""
        echo "啟動配置:"
        echo "  Host: $host"
        echo "  Port: $port"
        echo "  Workers: $workers"
        echo ""
        python ocr_fastapi_server_v2.py --host $host --port $port --workers $workers
        ;;
    *)
        echo "無效的選擇"
        exit 1
        ;;
esac