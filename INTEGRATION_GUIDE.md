# OCR 模型整合指南

本文件說明如何將訓練好的 OCR 模型整合到 `chrome_tou_fixed_v4_network_optimized.py` 中。

## 目錄
1. [系統架構](#系統架構)
2. [訓練流程](#訓練流程)
3. [部署 API 服務](#部署-api-服務)
4. [整合到 chrome_tou](#整合到-chrome_tou)
5. [效能優化建議](#效能優化建議)

## 系統架構

```
┌─────────────────────────────────────────────────────┐
│                  資料收集與標註                        │
├─────────────────────────────────────────────────────┤
│  collect_captcha.py  →  auto_label_captcha.py       │
│         ↓                      ↓                     │
│   驗證碼圖片            training_data.txt            │
└─────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│                    模型訓練                          │
├─────────────────────────────────────────────────────┤
│         train_ocr_model_v2.py                       │
│              ↓              ↓                        │
│    .pth 模型檔案    .onnx 模型檔案                    │
└─────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│                   API 服務部署                        │
├─────────────────────────────────────────────────────┤
│         ocr_api_server.py                           │
│         FastAPI (Port 8000)                         │
└─────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│                  整合到搶票程式                        │
├─────────────────────────────────────────────────────┤
│    chrome_tou_fixed_v4_network_optimized.py         │
└─────────────────────────────────────────────────────┘
```

## 訓練流程

### 1. 收集驗證碼資料

```bash
# 收集驗證碼圖片
cd ocr_training
python collect_captcha.py

# 自動標註（使用現有 ddddocr + 實際測試）
python auto_label_captcha.py
```

### 2. 手動標註失敗的驗證碼

編輯 `training_data.txt`，填入空白的答案：
```
captcha_00000.png,AB3D
captcha_00001.png,X9K2
captcha_00002.png,     # 需要手動填入
```

### 3. 訓練模型

```bash
# 使用 GPU 訓練（RTX 30/40 系列）
python train_ocr_model_v2.py \
    --data_dir captcha_auto_label/20250808_173711 \
    --labels captcha_auto_label/20250808_173711/training_data.txt \
    --epochs 100 \
    --batch_size 64 \
    --device cuda \
    --export_onnx

# 使用 CPU 訓練（較慢）
python train_ocr_model_v2.py \
    --data_dir captcha_auto_label/20250808_173711 \
    --labels captcha_auto_label/20250808_173711/training_data.txt \
    --epochs 50 \
    --batch_size 16 \
    --device cpu
```

訓練完成後會產生：
- `best_mobilenet_crnn_model.pth` - PyTorch 模型
- `mobilenet_crnn_model.onnx` - ONNX 模型（用於部署）
- `training_history.png` - 訓練曲線圖

## 部署 API 服務

### 1. 安裝依賴

```bash
pip install fastapi uvicorn onnxruntime pillow opencv-python
```

### 2. 啟動服務

```bash
# 啟動 API 服務（預設 port 8000）
python ocr_api_server.py --host 127.0.0.1 --port 8000

# 背景執行（Windows）
start /B python ocr_api_server.py

# 背景執行（Linux/Mac）
nohup python ocr_api_server.py &
```

### 3. 測試 API

```python
import requests
import base64

# 讀取驗證碼圖片
with open("test_captcha.png", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

# 發送請求
response = requests.post(
    "http://localhost:8000/predict",
    json={"image": img_base64, "format": "base64"}
)

result = response.json()
print(f"識別結果: {result['result']}")
print(f"信心分數: {result['confidence']}")
```

## 整合到 chrome_tou

### 方法一：使用 API 服務（推薦）

修改 `chrome_tou_fixed_v4_network_optimized.py` 中的 `tixcraft_get_ocr_answer` 函數：

```python
def tixcraft_get_ocr_answer(driver, ocr, ocr_captcha_image_source, Captcha_Browser, domain_name):
    """取得OCR答案 - 使用 API 服務版本"""
    import requests
    import base64
    
    ocr_answer = None
    
    try:
        # 取得驗證碼圖片
        image_element = WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#TicketForm_verifyCode-image'))
        )
        
        # 使用 screenshot 方法
        img_data = image_element.screenshot_as_png
        
        if img_data:
            # 轉為 base64
            img_base64 = base64.b64encode(img_data).decode()
            
            # 呼叫 OCR API
            try:
                response = requests.post(
                    "http://localhost:8000/predict",
                    json={"image": img_base64, "format": "base64"},
                    timeout=1  # 1秒超時
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result['success']:
                        ocr_answer = result['result']
                        confidence = result.get('confidence', 0)
                        
                        # 信心分數太低則忽略
                        if confidence < 0.7:
                            print(f"[{INSTANCE_ID}] OCR信心分數過低: {confidence}")
                            ocr_answer = None
                        else:
                            print(f"[{INSTANCE_ID}] OCR API: {ocr_answer} (信心: {confidence:.2f})")
            except requests.exceptions.Timeout:
                print(f"[{INSTANCE_ID}] OCR API 超時")
            except Exception as e:
                print(f"[{INSTANCE_ID}] OCR API 錯誤: {e}")
                
            # 如果 API 失敗，退回使用 ddddocr
            if not ocr_answer and ocr:
                try:
                    ocr_answer = ocr.classification(img_data)
                    if ocr_answer:
                        ocr_answer = ocr_answer.strip()
                        import re
                        ocr_answer = re.sub(r'[^A-Za-z0-9]', '', ocr_answer)
                except Exception as e:
                    print(f"[{INSTANCE_ID}] ddddocr 錯誤: {e}")
                    
    except Exception as e:
        print(f"[{INSTANCE_ID}] 取得驗證碼圖片錯誤: {e}")
    
    return ocr_answer
```

### 方法二：直接使用 ONNX Runtime（更快）

在 `chrome_tou_fixed_v4_network_optimized.py` 開頭新增：

```python
# OCR 模型相關
import onnxruntime as ort
onnx_session = None
OCR_CHARSET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
OCR_CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(OCR_CHARSET)}
OCR_IDX_TO_CHAR = {idx: char for char, idx in OCR_CHAR_TO_IDX.items()}
OCR_IDX_TO_CHAR[0] = '_'

def init_onnx_ocr():
    """初始化 ONNX OCR 模型"""
    global onnx_session
    try:
        model_path = os.path.join(util.get_app_root(), 'ocr_training', 'mobilenet_crnn_model.onnx')
        if os.path.exists(model_path):
            onnx_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            print(f"[{INSTANCE_ID}] ONNX OCR 模型載入成功")
            return True
    except Exception as e:
        print(f"[{INSTANCE_ID}] ONNX OCR 載入失敗: {e}")
    return False

def ocr_preprocess(img_data):
    """預處理驗證碼圖片"""
    from PIL import Image
    import io
    
    # 載入圖片
    image = Image.open(io.BytesIO(img_data)).convert('L')
    img_array = np.array(image)
    
    # 二值化
    _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 去噪
    img_array = cv2.medianBlur(img_array, 3)
    
    # 調整尺寸
    h, w = img_array.shape
    scale = 32 / h
    new_w = int(w * scale)
    new_w = min(max(new_w, 100), 160)
    
    img_array = cv2.resize(img_array, (new_w, 32))
    
    # Padding 到 128 寬度
    if new_w < 128:
        pad_width = 128 - new_w
        img_array = np.pad(img_array, ((0, 0), (0, pad_width)), mode='constant', constant_values=255)
    elif new_w > 128:
        img_array = img_array[:, :128]
    
    # 正規化
    img_array = img_array.astype(np.float32) / 255.0
    
    return img_array

def ocr_decode_ctc(output):
    """解碼 CTC 輸出"""
    if output.ndim == 3:
        output = output[:, 0, :]
    
    predictions = np.argmax(output, axis=1)
    
    chars = []
    prev = 0
    for pred in predictions:
        if pred != 0 and pred != prev:
            if pred in OCR_IDX_TO_CHAR:
                chars.append(OCR_IDX_TO_CHAR[pred])
        prev = pred
    
    return ''.join(chars)

def onnx_ocr_predict(img_data):
    """使用 ONNX 模型預測"""
    global onnx_session
    
    if not onnx_session:
        return None
    
    try:
        # 預處理
        img_array = ocr_preprocess(img_data)
        
        # 準備輸入
        img_input = img_array[np.newaxis, np.newaxis, :, :].astype(np.float32)
        
        # 預測
        input_name = onnx_session.get_inputs()[0].name
        output = onnx_session.run(None, {input_name: img_input})[0]
        
        # 解碼
        result = ocr_decode_ctc(output)
        
        return result
    except Exception as e:
        print(f"[{INSTANCE_ID}] ONNX 預測錯誤: {e}")
        return None
```

在 `main()` 函數中初始化：

```python
def main(args):
    # ... 原有程式碼 ...
    
    # 初始化 ONNX OCR（在建立瀏覽器之前）
    if config_dict["ocr_captcha"]["enable"]:
        if init_onnx_ocr():
            print("使用 ONNX OCR 模型")
        else:
            print("使用 ddddocr")
    
    # ... 繼續原有程式碼 ...
```

修改 `tixcraft_get_ocr_answer` 使用 ONNX：

```python
def tixcraft_get_ocr_answer(driver, ocr, ocr_captcha_image_source, Captcha_Browser, domain_name):
    """取得OCR答案 - ONNX 優先版本"""
    global onnx_session
    
    ocr_answer = None
    
    try:
        # 取得驗證碼圖片
        image_element = WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#TicketForm_verifyCode-image'))
        )
        
        img_data = image_element.screenshot_as_png
        
        if img_data:
            # 優先使用 ONNX
            if onnx_session:
                ocr_answer = onnx_ocr_predict(img_data)
                if ocr_answer:
                    print(f"[{INSTANCE_ID}] ONNX OCR: {ocr_answer}")
            
            # 退回使用 ddddocr
            if not ocr_answer and ocr:
                try:
                    ocr_answer = ocr.classification(img_data)
                    if ocr_answer:
                        ocr_answer = ocr_answer.strip()
                        import re
                        ocr_answer = re.sub(r'[^A-Za-z0-9]', '', ocr_answer)
                        print(f"[{INSTANCE_ID}] ddddocr: {ocr_answer}")
                except Exception as e:
                    print(f"[{INSTANCE_ID}] ddddocr 錯誤: {e}")
                    
    except Exception as e:
        print(f"[{INSTANCE_ID}] 取得驗證碼錯誤: {e}")
    
    return ocr_answer
```

## 效能優化建議

### 1. 資料收集優化
- 收集至少 5,000 張驗證碼圖片
- 確保涵蓋各種變形、噪點情況
- 平衡各字元的出現頻率

### 2. 訓練優化
- 使用 RTX 3060 以上 GPU 訓練
- Batch size 設為 64-128
- 使用混合精度訓練（fp16）
- 早停策略避免過擬合

### 3. 部署優化
- 使用 ONNX Runtime 而非 PyTorch
- INT8 量化進一步加速
- 預載模型避免冷啟動
- 使用進程池處理並發請求

### 4. 整合優化
- API 服務與搶票程式部署在同一台機器
- 使用 localhost 減少網路延遲
- 設定合理的超時時間（0.5-1秒）
- 失敗時退回使用 ddddocr

## 預期效果

| 指標 | 目標值 | 實際測試 |
|------|--------|----------|
| 準確率 | >95% | 待測試 |
| 推理時間（CPU） | <200ms | 待測試 |
| 推理時間（GPU） | <50ms | 待測試 |
| API 延遲 | <300ms | 待測試 |
| 並發處理 | >10 req/s | 待測試 |

## 常見問題

### Q1: 訓練時記憶體不足
- 減少 batch_size
- 使用梯度累積
- 凍結更多 MobileNet 層

### Q2: API 服務無法啟動
- 檢查端口是否被佔用
- 確認模型檔案路徑正確
- 安裝所需依賴套件

### Q3: 準確率不高
- 增加訓練資料量
- 調整資料增強參數
- 使用更大的模型（MobileNetV3-Large）

### Q4: 推理速度慢
- 使用 ONNX 而非 PyTorch
- 啟用 ONNX Runtime 優化
- 考慮 TensorRT（NVIDIA GPU）

## 進階功能

### 1. 模型版本管理
```python
# 使用時間戳記管理模型版本
model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/ocr_model_{model_version}.onnx"
```

### 2. A/B 測試
```python
# 同時運行多個模型進行比較
models = {
    "v1": load_model("model_v1.onnx"),
    "v2": load_model("model_v2.onnx"),
}

# 隨機選擇或根據規則選擇模型
selected_model = random.choice(list(models.values()))
```

### 3. 持續學習
```python
# 收集預測錯誤的案例
failed_captchas = []

# 定期重新訓練
if len(failed_captchas) > 1000:
    retrain_model(failed_captchas)
```

## 總結

本指南提供了完整的 OCR 模型訓練、部署和整合流程。關鍵步驟：

1. **資料收集**：使用自動標註工具快速建立資料集
2. **模型訓練**：MobileNetV3 + BiLSTM 架構，針對速度優化
3. **API 部署**：FastAPI 提供高效能服務
4. **程式整合**：多種整合方式，可根據需求選擇

建議先使用 API 方式整合，穩定後再考慮直接嵌入 ONNX Runtime 以獲得最佳效能。