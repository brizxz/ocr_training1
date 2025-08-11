# OCR 驗證碼識別準確度提升方案

## 目標
- **準確度**: 95% 以上
- **處理速度**: 0.2 秒內完成識別
- **部署環境**: 本地無 GPU，訓練環境有 GPU

## 當前問題分析
1. **ddddocr 準確度不足**: 當前準確率約 60-70%
2. **速度瓶頸**: 圖片擷取和預處理耗時
3. **驗證碼特性**: TixCraft 驗證碼為 4 位字母數字組合，有扭曲和雜訊

## 解決方案

### 方案一：自訂訓練模型（推薦）

#### 1. 數據收集策略
```python
# 驗證碼收集腳本範例
import os
import time
import base64
from selenium import webdriver

def collect_captcha_images(driver, save_dir, count=1000):
    """自動收集驗證碼圖片作為訓練資料"""
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(count):
        # 重新載入驗證碼
        driver.execute_script("""
            var img = document.getElementById('TicketForm_verifyCode-image');
            if(img) img.click();  // 點擊重新載入
        """)
        time.sleep(0.5)
        
        # 擷取驗證碼圖片
        img_base64 = driver.execute_script("""
            var img = document.getElementById('TicketForm_verifyCode-image');
            var canvas = document.createElement('canvas');
            var context = canvas.getContext('2d');
            canvas.height = img.naturalHeight;
            canvas.width = img.naturalWidth;
            context.drawImage(img, 0, 0);
            return canvas.toDataURL();
        """)
        
        # 儲存圖片
        if img_base64:
            img_data = base64.b64decode(img_base64.split(',')[1])
            with open(f"{save_dir}/captcha_{i:04d}.png", "wb") as f:
                f.write(img_data)
            print(f"已收集 {i+1}/{count} 張驗證碼")
```

#### 2. 標註工具
- **半自動標註**: 使用現有 OCR 預標註，人工修正
- **標註格式**: `filename,label` (例如: `captcha_0001.png,A3K9`)

#### 3. 模型訓練架構

##### A. 使用 CRNN + CTC Loss（輕量級）
```python
import torch
import torch.nn as nn

class LightweightCRNN(nn.Module):
    def __init__(self, img_height=32, num_classes=37):
        super().__init__()
        # CNN 特徵提取器（簡化版）
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        
        # RNN 序列建模
        self.rnn = nn.LSTM(128*8, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # CNN
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.view(b, c*h, w).permute(0, 2, 1)
        
        # RNN
        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)
        return output
```

##### B. 使用 TrOCR（高準確度）
```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# 使用預訓練模型微調
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")

# 針對 4 位驗證碼微調
model.config.max_length = 4
model.config.min_length = 4
```

#### 4. 訓練流程
```python
# 訓練配置
config = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 50,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "early_stopping_patience": 5,
    "model_save_path": "models/captcha_ocr_best.pth"
}

# 資料增強（提高泛化能力）
transforms = [
    RandomRotation(degrees=5),
    RandomAffine(degrees=0, translate=(0.1, 0.1)),
    GaussianNoise(mean=0, std=0.01),
    RandomBrightness(factor=0.2)
]
```

### 方案二：優化現有 ddddocr

#### 1. 圖片預處理優化
```python
import cv2
import numpy as np

def preprocess_captcha(image):
    """增強驗證碼圖片質量"""
    # 轉灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 去噪
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    
    # 二值化
    _, binary = cv2.threshold(denoised, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 形態學操作
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned
```

#### 2. 多模型投票機制
```python
def multi_model_ocr(image):
    """使用多個 OCR 模型投票"""
    results = []
    
    # 模型1: ddddocr
    results.append(ddddocr_ocr.classification(image))
    
    # 模型2: EasyOCR
    results.append(easyocr_reader.readtext(image)[0][1])
    
    # 模型3: Tesseract with config
    results.append(pytesseract.image_to_string(
        image, config='--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ))
    
    # 投票或選擇最可信的結果
    from collections import Counter
    vote = Counter(results)
    return vote.most_common(1)[0][0]
```

### 方案三：邊緣部署優化

#### 1. 模型量化
```python
# 使用 ONNX 轉換和量化
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic

# 轉換為 ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# 動態量化（減少模型大小，提升速度）
quantize_dynamic("model.onnx", "model_quant.onnx", 
                 weight_type=QuantType.QInt8)
```

#### 2. 推理優化
```python
import onnxruntime as ort

class FastOCR:
    def __init__(self, model_path):
        # 使用 ONNX Runtime 加速推理
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
    def predict(self, image):
        # 預處理
        processed = self.preprocess(image)
        
        # 推理（批次處理以提高效率）
        inputs = {self.session.get_inputs()[0].name: processed}
        outputs = self.session.run(None, inputs)
        
        # 後處理
        return self.decode(outputs[0])
```

## 實施計劃

### 第一階段：數據收集（1-2 天）
1. 部署收集腳本，自動收集 5000+ 張驗證碼
2. 建立標註系統，完成 80% 自動標註
3. 人工校正標註結果

### 第二階段：模型訓練（2-3 天）
1. 在 GPU 環境訓練 CRNN 模型
2. 調整超參數，達到 95% 準確率
3. 模型壓縮和量化

### 第三階段：本地部署（1 天）
1. 將訓練好的模型轉換為 ONNX
2. 整合到現有系統
3. 效能測試和優化

## 預期成果

| 指標 | 當前 | 目標 | 預期 |
|-----|------|------|------|
| 準確率 | 60-70% | 95% | 96-98% |
| 處理時間 | 0.3-0.5s | 0.2s | 0.15s |
| 模型大小 | - | <10MB | 5-8MB |
| CPU 使用率 | 高 | 中 | 低 |

## 備選方案

### 使用商業 API
- **優點**: 高準確率、無需維護
- **缺點**: 延遲較高、成本問題
- **選項**: 
  - 百度 OCR API
  - 騰訊 OCR API
  - Google Cloud Vision API

### 混合方案
1. 本地快速識別（0.1s）
2. 置信度低於閾值時呼叫 API
3. 持續收集失敗案例優化本地模型

## 結論

推薦採用**方案一**（自訂訓練模型）：
1. 可以針對 TixCraft 驗證碼特性優化
2. 一次訓練，長期使用
3. 本地推理速度快，無網路延遲
4. 可持續改進

實施要點：
- 收集足夠多樣的訓練數據
- 使用資料增強提高泛化能力
- 模型量化減少資源消耗
- 建立持續改進機制