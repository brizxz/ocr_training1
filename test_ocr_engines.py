#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試不同 OCR 引擎的準確率
比較 ddddocr、EasyOCR、Tesseract、PaddleOCR 等
"""

import os
import time
import base64
from PIL import Image
import numpy as np

# 測試圖片路徑
TEST_IMAGE = "captcha_sample.png"

def test_ddddocr():
    """測試 ddddocr"""
    print("=== 測試 ddddocr ===")
    try:
        import ddddocr
        ocr = ddddocr.DdddOcr(show_ad=False, use_gpu=False)
        
        if os.path.exists(TEST_IMAGE):
            with open(TEST_IMAGE, 'rb') as f:
                image_data = f.read()
            
            start_time = time.time()
            result = ocr.classification(image_data)
            elapsed_time = time.time() - start_time
            
            print(f"結果: {result}")
            print(f"耗時: {elapsed_time:.3f} 秒")
        else:
            print(f"測試圖片不存在: {TEST_IMAGE}")
    except ImportError:
        print("ddddocr 未安裝")
    except Exception as e:
        print(f"錯誤: {e}")
    print()

def test_easyocr():
    """測試 EasyOCR"""
    print("=== 測試 EasyOCR ===")
    try:
        import easyocr
        # 初始化 reader，只使用英文模型
        reader = easyocr.Reader(['en'], gpu=False)
        
        if os.path.exists(TEST_IMAGE):
            start_time = time.time()
            # EasyOCR 可以直接讀取檔案路徑
            result = reader.readtext(TEST_IMAGE, detail=0)
            elapsed_time = time.time() - start_time
            
            # 合併所有識別的文字
            text = ''.join(result)
            print(f"結果: {text}")
            print(f"耗時: {elapsed_time:.3f} 秒")
        else:
            print(f"測試圖片不存在: {TEST_IMAGE}")
    except ImportError:
        print("EasyOCR 未安裝，請執行: pip install easyocr")
    except Exception as e:
        print(f"錯誤: {e}")
    print()

def test_tesseract():
    """測試 Tesseract OCR"""
    print("=== 測試 Tesseract OCR ===")
    try:
        import pytesseract
        from PIL import Image
        
        # Windows 系統需要指定 tesseract.exe 路徑
        # 預設安裝路徑
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\%USERNAME%\AppData\Local\Tesseract-OCR\tesseract.exe"
        ]
        
        tesseract_found = False
        for path in tesseract_paths:
            expanded_path = os.path.expandvars(path)
            if os.path.exists(expanded_path):
                pytesseract.pytesseract.tesseract_cmd = expanded_path
                tesseract_found = True
                break
        
        if not tesseract_found:
            print("Tesseract 未安裝，請先安裝 Tesseract-OCR")
            print("下載連結: https://github.com/UB-Mannheim/tesseract/wiki")
            return
        
        if os.path.exists(TEST_IMAGE):
            image = Image.open(TEST_IMAGE)
            
            start_time = time.time()
            # 使用英文配置，限制字符集為英文字母
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            result = pytesseract.image_to_string(image, config=custom_config).strip()
            elapsed_time = time.time() - start_time
            
            print(f"結果: {result}")
            print(f"耗時: {elapsed_time:.3f} 秒")
        else:
            print(f"測試圖片不存在: {TEST_IMAGE}")
    except ImportError:
        print("pytesseract 未安裝，請執行: pip install pytesseract")
    except Exception as e:
        print(f"錯誤: {e}")
    print()

def test_paddleocr():
    """測試 PaddleOCR"""
    print("=== 測試 PaddleOCR ===")
    try:
        from paddleocr import PaddleOCR
        
        # 使用英文模型，關閉 GPU
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
        
        if os.path.exists(TEST_IMAGE):
            start_time = time.time()
            result = ocr.ocr(TEST_IMAGE, cls=True)
            elapsed_time = time.time() - start_time
            
            # 提取文字
            text = ""
            if result and result[0]:
                for line in result[0]:
                    text += line[1][0]
            
            print(f"結果: {text}")
            print(f"耗時: {elapsed_time:.3f} 秒")
        else:
            print(f"測試圖片不存在: {TEST_IMAGE}")
    except ImportError:
        print("PaddleOCR 未安裝，請執行: pip install paddlepaddle paddleocr")
    except Exception as e:
        print(f"錯誤: {e}")
    print()

def test_muggle_ocr():
    """測試 Muggle OCR（專門用於驗證碼）"""
    print("=== 測試 Muggle OCR ===")
    try:
        import muggle_ocr
        
        # 創建 SDK 實例
        sdk = muggle_ocr.SDK(model_type=muggle_ocr.ModelType.Captcha)
        
        if os.path.exists(TEST_IMAGE):
            with open(TEST_IMAGE, 'rb') as f:
                image_data = f.read()
            
            start_time = time.time()
            result = sdk.predict(image_bytes=image_data)
            elapsed_time = time.time() - start_time
            
            print(f"結果: {result}")
            print(f"耗時: {elapsed_time:.3f} 秒")
        else:
            print(f"測試圖片不存在: {TEST_IMAGE}")
    except ImportError:
        print("Muggle OCR 未安裝，請執行: pip install muggle-ocr")
    except Exception as e:
        print(f"錯誤: {e}")
    print()

def test_trocr():
    """測試 TrOCR (Transformers-based OCR)"""
    print("=== 測試 TrOCR (Microsoft) ===")
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image
        
        # 載入預訓練模型
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
        
        if os.path.exists(TEST_IMAGE):
            image = Image.open(TEST_IMAGE).convert("RGB")
            
            start_time = time.time()
            # 處理圖片
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            # 生成文字
            generated_ids = model.generate(pixel_values)
            result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            elapsed_time = time.time() - start_time
            
            print(f"結果: {result}")
            print(f"耗時: {elapsed_time:.3f} 秒")
        else:
            print(f"測試圖片不存在: {TEST_IMAGE}")
    except ImportError:
        print("TrOCR 未安裝，請執行: pip install transformers torch")
    except Exception as e:
        print(f"錯誤: {e}")
    print()

def create_sample_image():
    """創建測試用的範例圖片"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import random
        import string
        
        # 創建圖片
        width, height = 120, 40
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # 生成4個隨機英文字母
        text = ''.join(random.choices(string.ascii_uppercase, k=4))
        
        # 嘗試使用系統字體
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except:
            font = ImageFont.load_default()
        
        # 繪製文字
        draw.text((10, 5), text, font=font, fill='black')
        
        # 加入一些噪點
        for _ in range(100):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            draw.point((x, y), fill='gray')
        
        # 儲存圖片
        image.save(TEST_IMAGE)
        print(f"已創建範例圖片: {TEST_IMAGE}")
        print(f"驗證碼內容: {text}")
        return text
    except Exception as e:
        print(f"創建範例圖片失敗: {e}")
        return None

def main():
    """主程式"""
    print("=" * 60)
    print("OCR 引擎準確率比較測試")
    print("=" * 60)
    
    # 檢查或創建測試圖片
    if not os.path.exists(TEST_IMAGE):
        print("測試圖片不存在，創建範例圖片...")
        answer = create_sample_image()
        if answer:
            print(f"正確答案: {answer}")
        print("-" * 60)
    
    # 測試各種 OCR 引擎
    test_ddddocr()
    test_easyocr()
    test_tesseract()
    test_paddleocr()
    test_muggle_ocr()
    test_trocr()
    
    print("=" * 60)
    print("測試完成！")
    print("\n建議：")
    print("1. ddddocr - 速度快，適合簡單驗證碼，免費")
    print("2. EasyOCR - 準確率高，支援多語言，適合複雜場景")
    print("3. Tesseract - 開源經典，需要預處理，可高度客製化")
    print("4. PaddleOCR - 百度開源，中文識別優秀，支援多種場景")
    print("5. Muggle OCR - 專門針對驗證碼優化")
    print("6. TrOCR - 基於 Transformer，最新技術，準確率高但較慢")
    print("\n對於4個英文字母的驗證碼，建議優先嘗試：")
    print("- Muggle OCR（專門處理驗證碼）")
    print("- EasyOCR（準確率較高）")
    print("- 訓練專屬的 ddddocr 模型")

if __name__ == "__main__":
    main()