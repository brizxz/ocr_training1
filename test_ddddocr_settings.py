#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 ddddocr 設定功能
測試 set_ranges 限制字符集和長度驗證
"""

import ddddocr
import os

def test_basic_ocr():
    """測試基本 OCR 功能"""
    print("=== 測試基本 OCR ===")
    ocr = ddddocr.DdddOcr()
    
    # 測試文字
    test_text = "TEST1234"
    print(f"測試文字: {test_text}")
    
    # 這裡應該用實際的驗證碼圖片
    # result = ocr.classification(image_data)
    print("需要實際圖片來測試")
    print()

def test_set_ranges():
    """測試 set_ranges 功能"""
    print("=== 測試 set_ranges 限制字符集 ===")
    ocr = ddddocr.DdddOcr()
    
    # 設定只識別英文字母
    ocr.set_ranges("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    print("已設定只識別英文字母 (a-z, A-Z)")
    
    # 測試圖片路徑（需要替換為實際圖片）
    test_image_path = "captcha_sample.png"
    
    if os.path.exists(test_image_path):
        with open(test_image_path, 'rb') as f:
            image_data = f.read()
        
        result = ocr.classification(image_data)
        print(f"識別結果: {result}")
        
        # 驗證結果
        if len(result) == 4 and result.isalpha():
            print(f"✓ 符合4個英文字母格式")
        else:
            print(f"✗ 不符合格式 (長度: {len(result)}, 全英文: {result.isalpha()})")
    else:
        print(f"測試圖片不存在: {test_image_path}")
    print()

def test_validation():
    """測試驗證邏輯"""
    print("=== 測試驗證邏輯 ===")
    
    test_cases = [
        ("ABCD", True, "正確格式"),
        ("ABC", False, "長度不足"),
        ("ABCDE", False, "長度超過"),
        ("AB12", False, "包含數字"),
        ("AB#D", False, "包含特殊字符"),
        ("abcd", True, "小寫字母"),
        ("AbCd", True, "混合大小寫"),
    ]
    
    for text, expected, description in test_cases:
        # 驗證邏輯
        is_valid = len(text) == 4 and text.isalpha()
        result = "✓" if is_valid == expected else "✗"
        print(f"{result} {text:8} - {description} (預期: {expected}, 實際: {is_valid})")
    print()

def test_processing():
    """測試處理邏輯"""
    print("=== 測試處理邏輯 ===")
    
    test_inputs = [
        "ABCD",      # 正確格式
        "ABCDEF",    # 太長
        "AB",        # 太短
        "AB12",      # 包含數字
        "abcd",      # 小寫
    ]
    
    for initial_guess in test_inputs:
        processed = initial_guess
        print(f"原始: {initial_guess:8}", end=" -> ")
        
        # 處理邏輯（與 auto_label_captcha.py 相同）
        if len(processed) != 4 or not processed.isalpha():
            if len(processed) > 4:
                processed = processed[:4]  # 截取前4個字元
                print(f"截取: {processed:8}", end=" ")
            elif len(processed) < 4:
                print(f"長度不足，跳過", end="")
                continue
            
            if not processed.isalpha():
                print(f"包含非英文，跳過", end="")
                continue
        
        # 統一轉為大寫
        processed = processed.upper()
        print(f"最終: {processed}")
    print()

if __name__ == "__main__":
    print("ddddocr 設定測試")
    print("=" * 50)
    
    # 執行測試
    test_basic_ocr()
    test_set_ranges()
    test_validation()
    test_processing()
    
    print("=" * 50)
    print("測試完成")