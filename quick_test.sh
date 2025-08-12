#!/bin/bash

echo "OCR API 快速測試"
echo "================="

# 設定 API URL
API_URL="${1:-http://localhost:8000}"

# 檢查服務是否運行
echo "檢查服務狀態..."
if curl -s "${API_URL}/health" > /dev/null 2>&1; then
    echo "✓ 服務正在運行"
    curl -s "${API_URL}/health" | python -m json.tool
else
    echo "✗ 無法連接到服務"
    echo "請先啟動服務: ./start_server.sh"
    exit 1
fi

echo ""
echo "測試單張圖片預測..."

# 找一張測試圖片
TEST_IMAGE=$(find captcha_auto_label/merged_20250811_155009 -name "*.png" | head -1)

if [ -z "$TEST_IMAGE" ]; then
    echo "找不到測試圖片"
    exit 1
fi

echo "使用圖片: $TEST_IMAGE"
echo ""

# 發送預測請求
RESULT=$(curl -s -X POST -F "file=@${TEST_IMAGE}" "${API_URL}/predict")

if [ $? -eq 0 ]; then
    echo "預測結果:"
    echo "$RESULT" | python -m json.tool
    
    # 提取結果
    PREDICTED=$(echo "$RESULT" | python -c "import sys, json; print(json.load(sys.stdin)['result'])")
    CONFIDENCE=$(echo "$RESULT" | python -c "import sys, json; print(json.load(sys.stdin)['confidence'])")
    TIME=$(echo "$RESULT" | python -c "import sys, json; print(json.load(sys.stdin)['inference_time_ms'])")
    
    echo ""
    echo "摘要:"
    echo "  預測結果: $PREDICTED"
    echo "  信心分數: $CONFIDENCE"
    echo "  推理時間: ${TIME}ms"
else
    echo "預測失敗"
fi

echo ""
echo "查看服務統計..."
curl -s "${API_URL}/stats" | python -m json.tool