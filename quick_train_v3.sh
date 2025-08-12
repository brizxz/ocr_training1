#!/bin/bash

echo "==========================================="
echo "OCR 模型 v3 快速訓練腳本"
echo "解決過擬合問題的改進版本"
echo "==========================================="
echo ""

# 設定預設參數
DATA_DIR="captcha_auto_label/merged_20250811_155009"
LABELS_FILE="captcha_auto_label/merged_20250811_155009/training_data.txt"

# 檢查資料目錄
if [ ! -d "$DATA_DIR" ]; then
    echo "錯誤: 找不到資料目錄 $DATA_DIR"
    exit 1
fi

if [ ! -f "$LABELS_FILE" ]; then
    echo "錯誤: 找不到標註檔案 $LABELS_FILE"
    exit 1
fi

# 顯示資料統計
echo "資料統計:"
TOTAL_LINES=$(wc -l < "$LABELS_FILE")
LABELED_LINES=$(grep -E ",[A-Za-z0-9]{4}$" "$LABELS_FILE" | wc -l)
echo "  總圖片數: $TOTAL_LINES"
echo "  已標註數: $LABELED_LINES"
echo "  標註率: $(echo "scale=1; $LABELED_LINES * 100 / $TOTAL_LINES" | bc)%"
echo ""

# 選擇訓練模式
echo "請選擇訓練模式:"
echo "  1. 快速測試 (10 epochs, 測試環境)"
echo "  2. 標準訓練 (50 epochs, 推薦)"
echo "  3. 深度訓練 (100 epochs, 最佳效果)"
echo "  4. 自訂參數"
echo ""
read -p "請選擇 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "執行快速測試訓練..."
        echo "參數: 10 epochs, batch_size=32, lr=0.001"
        python train_ocr_model_v3_improved.py \
            --data_dir "$DATA_DIR" \
            --labels "$LABELS_FILE" \
            --epochs 10 \
            --batch_size 32 \
            --lr 0.001 \
            --device cuda \
            --test_speed
        ;;
    2)
        echo ""
        echo "執行標準訓練..."
        echo "參數: 50 epochs, batch_size=32, lr=0.0005"
        python train_ocr_model_v3_improved.py \
            --data_dir "$DATA_DIR" \
            --labels "$LABELS_FILE" \
            --epochs 50 \
            --batch_size 32 \
            --lr 0.0005 \
            --device cuda
        ;;
    3)
        echo ""
        echo "執行深度訓練..."
        echo "參數: 100 epochs, batch_size=16, lr=0.0003"
        python train_ocr_model_v3_improved.py \
            --data_dir "$DATA_DIR" \
            --labels "$LABELS_FILE" \
            --epochs 100 \
            --batch_size 16 \
            --lr 0.0003 \
            --device cuda
        ;;
    4)
        echo ""
        echo "自訂參數訓練"
        read -p "Epochs (預設 50): " epochs
        epochs=${epochs:-50}
        read -p "Batch size (預設 32): " batch_size
        batch_size=${batch_size:-32}
        read -p "Learning rate (預設 0.0005): " lr
        lr=${lr:-0.0005}
        read -p "Device (cuda/cpu, 預設 cuda): " device
        device=${device:-cuda}
        
        echo ""
        echo "執行自訂訓練..."
        echo "參數: epochs=$epochs, batch_size=$batch_size, lr=$lr, device=$device"
        python train_ocr_model_v3_improved.py \
            --data_dir "$DATA_DIR" \
            --labels "$LABELS_FILE" \
            --epochs $epochs \
            --batch_size $batch_size \
            --lr $lr \
            --device $device
        ;;
    *)
        echo "無效的選擇"
        exit 1
        ;;
esac

# 檢查訓練結果
if [ -f "best_lightweight_crnn_model.pth" ]; then
    echo ""
    echo "==========================================="
    echo "訓練完成！"
    echo "模型已儲存至: best_lightweight_crnn_model.pth"
    echo "訓練曲線已儲存至: training_history_v3.png"
    echo ""
    echo "下一步:"
    echo "1. 啟動 API 服務:"
    echo "   python ocr_fastapi_server.py"
    echo ""
    echo "2. 測試模型:"
    echo "   curl -X POST -F 'file=@captcha.png' http://localhost:8000/predict"
    echo "==========================================="
fi