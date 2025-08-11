#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合訓練好的 OCR 模型到 chrome_tou_fixed_v4_network_optimized.py
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

# =====================================================
# 方法 1: 使用 ONNX 模型（推薦，速度快）
# =====================================================

class CustomOCR:
    """自訂 OCR 模型類別"""
    
    def __init__(self, model_path="ocr_training/ocr_model.onnx"):
        """
        初始化自訂 OCR 模型
        
        Args:
            model_path: ONNX 模型檔案路徑
        """
        # 載入 ONNX 模型
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 字元集
        self.charset = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
    def preprocess(self, image_data):
        """
        預處理圖片
        
        Args:
            image_data: 圖片資料（bytes 或 PIL Image）
        
        Returns:
            numpy array: 預處理後的圖片
        """
        # 如果是 bytes，轉換為 numpy array
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_data, Image.Image):
            image = np.array(image.convert('L'))
        else:
            image = image_data
        
        # 調整大小為 128x32
        image = cv2.resize(image, (128, 32), interpolation=cv2.INTER_LINEAR)
        
        # 正規化
        image = (image.astype(np.float32) - 127.5) / 127.5
        
        # 增加維度 [batch, channel, height, width]
        image = image[np.newaxis, np.newaxis, :, :]
        
        return image
    
    def decode_prediction(self, output):
        """
        解碼模型輸出
        
        Args:
            output: 模型輸出 [batch, seq_len, num_classes]
        
        Returns:
            str: 識別結果
        """
        # 取得最大機率的類別
        preds = output[0].argmax(axis=1)
        
        # 移除重複和空白
        result = []
        prev = 0
        for p in preds:
            if p != 0 and p != prev:  # 非空白且不重複
                if p <= len(self.charset):
                    result.append(self.charset[p-1])
            prev = p
        
        return ''.join(result)
    
    def classification(self, image_data):
        """
        識別驗證碼（與 ddddocr 相同的介面）
        
        Args:
            image_data: 圖片資料
        
        Returns:
            str: 識別結果
        """
        # 預處理
        image = self.preprocess(image_data)
        
        # 推論
        output = self.session.run([self.output_name], {self.input_name: image})
        
        # 解碼
        result = self.decode_prediction(output)
        
        # 確保輸出是 4 位字元
        if len(result) > 4:
            result = result[:4]
        elif len(result) < 4:
            # 補 0 或使用其他策略
            result = result.ljust(4, '0')
        
        return result

# =====================================================
# 方法 2: 混合使用多個 OCR 模型
# =====================================================

class HybridOCR:
    """混合 OCR 系統（結合自訂模型和 ddddocr）"""
    
    def __init__(self, custom_model_path=None):
        """初始化混合 OCR"""
        import ddddocr
        
        # 載入自訂模型
        self.custom_ocr = None
        if custom_model_path and os.path.exists(custom_model_path):
            self.custom_ocr = CustomOCR(custom_model_path)
        
        # 載入 ddddocr 作為備用
        self.ddddocr = ddddocr.DdddOcr()
        
        # 信心度閾值
        self.confidence_threshold = 0.8
        
    def classification(self, image_data):
        """
        使用混合策略識別驗證碼
        
        Args:
            image_data: 圖片資料
        
        Returns:
            str: 識別結果
        """
        results = []
        
        # 1. 嘗試自訂模型
        if self.custom_ocr:
            try:
                custom_result = self.custom_ocr.classification(image_data)
                results.append(custom_result)
            except:
                pass
        
        # 2. 使用 ddddocr
        try:
            dddd_result = self.ddddocr.classification(image_data)
            results.append(dddd_result)
        except:
            pass
        
        # 3. 選擇最佳結果
        if len(results) == 0:
            return "0000"  # 預設值
        elif len(results) == 1:
            return results[0]
        else:
            # 如果兩個結果相同，信心度高
            if results[0] == results[1]:
                return results[0]
            # 否則使用自訂模型（假設準確度更高）
            elif self.custom_ocr:
                return results[0]
            else:
                return results[1]

# =====================================================
# 整合到 chrome_tou_fixed_v4_network_optimized.py
# =====================================================

"""
修改 chrome_tou_fixed_v4_network_optimized.py 的方法：

1. 在檔案開頭加入匯入：
```python
import os
import sys
# 加入 ocr_training 目錄到路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'ocr_training'))
from integrate_model import CustomOCR, HybridOCR
```

2. 修改 ocr_captcha_image 函數（約第 1369 行）：

原本的程式碼：
```python
def ocr_captcha_image(self, image_source, image_id):
    ...
    if self.config_dict["ocr_captcha"]["enable"]:
        ocr = ddddocr.DdddOcr()
        ...
```

修改為：
```python
def ocr_captcha_image(self, image_source, image_id):
    ...
    if self.config_dict["ocr_captcha"]["enable"]:
        # 使用自訂 OCR 模型
        model_path = "ocr_training/ocr_model.onnx"
        
        # 方法 1: 只使用自訂模型
        if os.path.exists(model_path):
            ocr = CustomOCR(model_path)
        else:
            # 如果沒有自訂模型，使用原本的 ddddocr
            ocr = ddddocr.DdddOcr()
        
        # 方法 2: 使用混合模型（推薦）
        # ocr = HybridOCR(model_path)
        
        # 其餘程式碼不變
        ...
```

3. 或者建立一個 OCR 管理器（更優雅的方式）：

在 __init__ 方法中初始化（約第 250 行）：
```python
def __init__(self):
    ...
    # 初始化 OCR 模型
    self.init_ocr_model()
    ...

def init_ocr_model(self):
    '''初始化 OCR 模型'''
    model_path = "ocr_training/ocr_model.onnx"
    
    if os.path.exists(model_path):
        print(f"[OCR] 載入自訂模型: {model_path}")
        self.ocr_model = CustomOCR(model_path)
    else:
        print("[OCR] 使用預設 ddddocr 模型")
        import ddddocr
        self.ocr_model = ddddocr.DdddOcr()
```

然後在 ocr_captcha_image 中使用：
```python
def ocr_captcha_image(self, image_source, image_id):
    ...
    if self.config_dict["ocr_captcha"]["enable"]:
        # 使用初始化的模型
        ocr = self.ocr_model
        ...
```
"""

# =====================================================
# 測試程式
# =====================================================

def test_custom_ocr():
    """測試自訂 OCR 模型"""
    import os
    
    model_path = "ocr_model.onnx"
    test_image = "test_captcha.png"
    
    if not os.path.exists(model_path):
        print(f"找不到模型: {model_path}")
        print("請先執行 train_ocr_model.py 訓練模型")
        return
    
    if not os.path.exists(test_image):
        print(f"找不到測試圖片: {test_image}")
        return
    
    # 載入模型
    ocr = CustomOCR(model_path)
    
    # 讀取測試圖片
    with open(test_image, 'rb') as f:
        image_data = f.read()
    
    # 識別
    result = ocr.classification(image_data)
    print(f"識別結果: {result}")
    
    # 測試混合模型
    hybrid = HybridOCR(model_path)
    hybrid_result = hybrid.classification(image_data)
    print(f"混合模型結果: {hybrid_result}")

if __name__ == "__main__":
    test_custom_ocr()