# OCR 模型訓練建議與解決方案

## 問題診斷

您的模型出現了嚴重的**過擬合問題**：
- 訓練準確率: 93.69%
- 驗證準確率: 6.03%

### 主要原因：
1. **標籤格式不一致**: 訓練資料標籤是小寫，但模型預期大寫
2. **資料增強不足**: 原始模型缺乏足夠的資料增強
3. **模型複雜度過高**: MobileNetV3 對於簡單的驗證碼可能過於複雜
4. **正則化不足**: Dropout 和 L2 正則化設定不當

## 解決方案

### 1. 使用改進的訓練腳本 (推薦)

```bash
# 使用 v3 改進版腳本
python train_ocr_model_v3_improved.py \
    --data_dir captcha_auto_label/merged_20250811_155009 \
    --labels captcha_auto_label/merged_20250811_155009/training_data.txt \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0005 \
    --device cuda
```

### 2. 關鍵改進

#### a. 資料處理
- ✅ 自動將標籤轉換為大寫
- ✅ 增加訓練集比例 (85% vs 80%)
- ✅ 驗證集不使用資料增強

#### b. 強化的資料增強
- 隨機旋轉 (±5度)
- 隨機縮放 (0.9-1.1x)
- 高斯模糊
- 隨機噪點
- 亮度/對比度調整
- 模擬干擾線

#### c. 輕量化模型架構
- 使用簡單的 CNN + BiLSTM
- 減少參數量，提升推理速度
- 目標: 0.1-0.2秒推理時間

#### d. 正則化策略
- Dropout: 0.5 (FC層), 0.3 (LSTM)
- L2 權重衰減: 1e-4
- 梯度裁剪: max_norm=5.0

## 訓練參數建議

### 初次測試（快速驗證）
```bash
python train_ocr_model_v3_improved.py \
    --data_dir captcha_auto_label/merged_20250811_155009 \
    --labels captcha_auto_label/merged_20250811_155009/training_data.txt \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.001 \
    --device cuda \
    --test_speed
```

### 標準訓練（推薦）
```bash
python train_ocr_model_v3_improved.py \
    --data_dir captcha_auto_label/merged_20250811_155009 \
    --labels captcha_auto_label/merged_20250811_155009/training_data.txt \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0005 \
    --device cuda
```

### 深度訓練（最佳效果）
```bash
python train_ocr_model_v3_improved.py \
    --data_dir captcha_auto_label/merged_20250811_155009 \
    --labels captcha_auto_label/merged_20250811_155009/training_data.txt \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0003 \
    --device cuda
```

## 預期結果

使用改進的方法，您應該能達到：
- **驗證準確率**: 85-95%
- **推理速度**: 50-100ms (CPU), 10-20ms (GPU)
- **並發處理**: 5-10 個請求/秒

## 部署建議

### 1. 啟動 FastAPI 服務

```bash
# 單工作進程（開發）
python ocr_fastapi_server.py --host 0.0.0.0 --port 8000

# 多工作進程（生產）
python ocr_fastapi_server.py --host 0.0.0.0 --port 8000 --workers 4
```

### 2. API 端點

- `POST /predict` - 單張圖片預測
- `POST /predict_batch` - 批次預測（最多20張）
- `POST /predict_async` - 非同步預測
- `GET /health` - 健康檢查
- `GET /stats` - 服務統計

### 3. 測試 API

```python
import requests

# 單張預測
with open('captcha.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    print(response.json())
    
# 批次預測
files = [
    ('files', open('captcha1.png', 'rb')),
    ('files', open('captcha2.png', 'rb')),
]
response = requests.post(
    'http://localhost:8000/predict_batch',
    files=files
)
print(response.json())
```

## 監控訓練過程

觀察以下指標：
1. **Loss 曲線**: 訓練和驗證 loss 應該同步下降
2. **準確率差距**: 訓練和驗證準確率差距不應超過 10%
3. **早停**: 如果驗證準確率 20 個 epochs 沒改善，自動停止

## 疑難排解

### 問題: 驗證準確率仍然很低
- 檢查標籤格式是否正確
- 增加資料量或資料增強
- 降低學習率

### 問題: 推理速度太慢
- 使用 CPU 時確保沒有使用太大的 batch size
- 考慮使用 ONNX 或 TorchScript 優化
- 使用批次預測 API

### 問題: 記憶體不足
- 減少 batch size
- 使用梯度累積
- 減少模型大小

## 下一步

1. **收集更多資料**: 持續收集和標註新的驗證碼
2. **主動學習**: 優先標註模型不確定的樣本
3. **模型蒸餾**: 訓練更小的學生模型
4. **部署優化**: 使用 ONNX Runtime 或 TensorRT