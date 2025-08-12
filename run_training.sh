#!/bin/bash

echo "==========================================="
echo "OCR 模型訓練腳本"
echo "使用合併的驗證碼標註資料"
echo "==========================================="
echo ""
echo "可用的選項:"
echo "  1. 快速測試 (10 epochs, 用於驗證環境)"
echo "  2. 標準訓練 (50 epochs)"
echo "  3. 完整訓練 (100 epochs)"
echo "  4. 自訂參數"
echo ""
read -p "請選擇選項 (1-4): " choice

case $choice in
    1)
        echo "執行快速測試訓練..."
        python train_with_merged_data.py --epochs 10 --batch_size 32
        ;;
    2)
        echo "執行標準訓練..."
        python train_with_merged_data.py --epochs 50 --batch_size 64
        ;;
    3)
        echo "執行完整訓練..."
        python train_with_merged_data.py --epochs 100 --batch_size 64
        ;;
    4)
        echo "自訂參數訓練"
        echo "範例: python train_with_merged_data.py --epochs 50 --batch_size 32 --lr 0.001 --device cuda"
        echo ""
        read -p "請輸入完整的命令參數: " params
        python train_with_merged_data.py $params
        ;;
    *)
        echo "無效的選項"
        exit 1
        ;;
esac