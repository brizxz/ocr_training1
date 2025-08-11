#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 OCR 整合
用於測試 OCR API 和本地模型的效能
"""

import os
import time
import base64
import requests
from PIL import Image
import numpy as np
import io
from typing import List, Tuple
import concurrent.futures
import statistics

def test_api_service(image_path: str, api_url: str = "http://localhost:8000") -> dict:
    """測試 API 服務"""
    try:
        # 讀取圖片
        with open(image_path, "rb") as f:
            img_data = f.read()
        
        # 轉為 base64
        img_base64 = base64.b64encode(img_data).decode()
        
        # 測量時間
        start_time = time.time()
        
        # 發送請求
        response = requests.post(
            f"{api_url}/predict",
            json={"image": img_base64, "format": "base64"},
            timeout=5
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            result['total_time'] = end_time - start_time
            return result
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "total_time": end_time - start_time
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "total_time": 0
        }


def test_onnx_local(image_path: str, model_path: str = "mobilenet_crnn_model.onnx") -> dict:
    """測試本地 ONNX 模型"""
    try:
        import onnxruntime as ort
        import cv2
        
        # 載入模型
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # 讀取和預處理圖片
        image = Image.open(image_path).convert('L')
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
        
        # Padding
        if new_w < 128:
            pad_width = 128 - new_w
            img_array = np.pad(img_array, ((0, 0), (0, pad_width)), mode='constant', constant_values=255)
        elif new_w > 128:
            img_array = img_array[:, :128]
        
        # 正規化
        img_array = img_array.astype(np.float32) / 255.0
        
        # 準備輸入
        img_input = img_array[np.newaxis, np.newaxis, :, :].astype(np.float32)
        
        # 測量推理時間
        start_time = time.time()
        
        # 預測
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: img_input})[0]
        
        end_time = time.time()
        
        # 解碼 CTC
        CHARSET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHARSET)}
        IDX_TO_CHAR[0] = '_'
        
        if output.ndim == 3:
            output = output[:, 0, :]
        
        predictions = np.argmax(output, axis=1)
        
        chars = []
        prev = 0
        for pred in predictions:
            if pred != 0 and pred != prev:
                if pred in IDX_TO_CHAR:
                    chars.append(IDX_TO_CHAR[pred])
            prev = pred
        
        result = ''.join(chars)
        
        return {
            "success": True,
            "result": result,
            "inference_time": end_time - start_time
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "inference_time": 0
        }


def test_ddddocr_local(image_path: str) -> dict:
    """測試 ddddocr"""
    try:
        import ddddocr
        
        # 初始化
        ocr = ddddocr.DdddOcr(show_ad=False)
        
        # 讀取圖片
        with open(image_path, "rb") as f:
            img_data = f.read()
        
        # 測量時間
        start_time = time.time()
        
        # 預測
        result = ocr.classification(img_data)
        
        end_time = time.time()
        
        return {
            "success": True,
            "result": result,
            "inference_time": end_time - start_time
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "inference_time": 0
        }


def batch_test(image_dir: str, ground_truth_file: str = None) -> None:
    """批量測試"""
    print("=" * 60)
    print("OCR 效能測試")
    print("=" * 60)
    
    # 載入真實標籤（如果有）
    ground_truth = {}
    if ground_truth_file and os.path.exists(ground_truth_file):
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            for line in f:
                if ',' in line:
                    filename, label = line.strip().split(',')
                    if label:  # 只載入有標籤的
                        ground_truth[filename] = label.upper()
    
    # 收集測試圖片
    test_images = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            test_images.append(os.path.join(image_dir, filename))
    
    if not test_images:
        print("找不到測試圖片！")
        return
    
    print(f"找到 {len(test_images)} 張測試圖片")
    if ground_truth:
        print(f"載入 {len(ground_truth)} 個標籤")
    print("-" * 60)
    
    # 測試結果
    results = {
        "api": {"times": [], "correct": 0, "total": 0, "errors": 0},
        "onnx": {"times": [], "correct": 0, "total": 0, "errors": 0},
        "ddddocr": {"times": [], "correct": 0, "total": 0, "errors": 0}
    }
    
    # 測試每張圖片
    for i, image_path in enumerate(test_images[:10], 1):  # 只測試前10張
        filename = os.path.basename(image_path)
        true_label = ground_truth.get(filename, "")
        
        print(f"\n[{i}/{min(10, len(test_images))}] 測試: {filename}")
        if true_label:
            print(f"  真實答案: {true_label}")
        
        # 測試 API
        print("  測試 API...", end="")
        api_result = test_api_service(image_path)
        if api_result['success']:
            pred = api_result.get('result', '')
            time_taken = api_result.get('total_time', 0)
            results['api']['times'].append(time_taken)
            results['api']['total'] += 1
            if true_label and pred == true_label:
                results['api']['correct'] += 1
            print(f" {pred} ({time_taken:.3f}s)")
        else:
            results['api']['errors'] += 1
            print(f" 失敗: {api_result.get('error', 'Unknown')}")
        
        # 測試 ONNX
        print("  測試 ONNX...", end="")
        if os.path.exists("mobilenet_crnn_model.onnx"):
            onnx_result = test_onnx_local(image_path)
            if onnx_result['success']:
                pred = onnx_result.get('result', '')
                time_taken = onnx_result.get('inference_time', 0)
                results['onnx']['times'].append(time_taken)
                results['onnx']['total'] += 1
                if true_label and pred == true_label:
                    results['onnx']['correct'] += 1
                print(f" {pred} ({time_taken:.3f}s)")
            else:
                results['onnx']['errors'] += 1
                print(f" 失敗: {onnx_result.get('error', 'Unknown')}")
        else:
            print(" 模型不存在")
        
        # 測試 ddddocr
        print("  測試 ddddocr...", end="")
        dddd_result = test_ddddocr_local(image_path)
        if dddd_result['success']:
            pred = dddd_result.get('result', '').upper()
            time_taken = dddd_result.get('inference_time', 0)
            results['ddddocr']['times'].append(time_taken)
            results['ddddocr']['total'] += 1
            if true_label and pred == true_label:
                results['ddddocr']['correct'] += 1
            print(f" {pred} ({time_taken:.3f}s)")
        else:
            results['ddddocr']['errors'] += 1
            print(f" 失敗: {dddd_result.get('error', 'Unknown')}")
    
    # 統計結果
    print("\n" + "=" * 60)
    print("測試結果統計")
    print("=" * 60)
    
    for method in ['api', 'onnx', 'ddddocr']:
        res = results[method]
        print(f"\n{method.upper()}:")
        
        if res['total'] > 0:
            accuracy = res['correct'] / res['total'] * 100 if ground_truth else 0
            avg_time = statistics.mean(res['times']) if res['times'] else 0
            min_time = min(res['times']) if res['times'] else 0
            max_time = max(res['times']) if res['times'] else 0
            
            print(f"  成功率: {res['total']}/{res['total'] + res['errors']}")
            if ground_truth:
                print(f"  準確率: {accuracy:.1f}% ({res['correct']}/{res['total']})")
            print(f"  平均時間: {avg_time:.3f}s")
            print(f"  最快時間: {min_time:.3f}s")
            print(f"  最慢時間: {max_time:.3f}s")
        else:
            print(f"  無有效測試結果")


def stress_test(api_url: str = "http://localhost:8000", num_requests: int = 100, num_workers: int = 10):
    """壓力測試 API"""
    print("\n" + "=" * 60)
    print("API 壓力測試")
    print("=" * 60)
    print(f"請求數: {num_requests}")
    print(f"並發數: {num_workers}")
    
    # 準備測試圖片
    test_image = Image.new('L', (128, 32), 255)
    buffer = io.BytesIO()
    test_image.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    def single_request():
        try:
            start = time.time()
            response = requests.post(
                f"{api_url}/predict",
                json={"image": img_base64, "format": "base64"},
                timeout=5
            )
            elapsed = time.time() - start
            return response.status_code == 200, elapsed
        except:
            return False, 0
    
    # 執行壓力測試
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(single_request) for _ in range(num_requests)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    total_time = time.time() - start_time
    
    # 統計結果
    successful = sum(1 for success, _ in results if success)
    failed = num_requests - successful
    response_times = [time for success, time in results if success and time > 0]
    
    print(f"\n測試完成！總耗時: {total_time:.2f}s")
    print(f"成功: {successful}/{num_requests}")
    print(f"失敗: {failed}/{num_requests}")
    
    if response_times:
        print(f"平均響應時間: {statistics.mean(response_times):.3f}s")
        print(f"最快響應: {min(response_times):.3f}s")
        print(f"最慢響應: {max(response_times):.3f}s")
        print(f"QPS: {successful / total_time:.2f} req/s")


def main():
    """主程式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='測試 OCR 整合')
    parser.add_argument('--mode', type=str, default='batch', 
                       choices=['batch', 'stress', 'single'],
                       help='測試模式')
    parser.add_argument('--image_dir', type=str, 
                       default='captcha_auto_label/20250808_173711',
                       help='圖片目錄')
    parser.add_argument('--labels', type=str,
                       help='標籤檔案 (training_data.txt)')
    parser.add_argument('--api_url', type=str, default='http://localhost:8000',
                       help='API 服務地址')
    parser.add_argument('--image', type=str,
                       help='單張圖片路徑（single 模式）')
    
    args = parser.parse_args()
    
    if args.mode == 'batch':
        # 批量測試
        labels_file = args.labels
        if not labels_file and os.path.exists(os.path.join(args.image_dir, 'training_data.txt')):
            labels_file = os.path.join(args.image_dir, 'training_data.txt')
        
        batch_test(args.image_dir, labels_file)
        
    elif args.mode == 'stress':
        # 壓力測試
        stress_test(args.api_url)
        
    elif args.mode == 'single':
        # 單張測試
        if not args.image:
            print("請提供圖片路徑 (--image)")
            return
        
        print("測試單張圖片:", args.image)
        
        # API 測試
        print("\nAPI 測試:")
        api_result = test_api_service(args.image, args.api_url)
        print(f"  結果: {api_result.get('result', 'N/A')}")
        print(f"  信心: {api_result.get('confidence', 0):.2f}")
        print(f"  時間: {api_result.get('total_time', 0):.3f}s")
        
        # ONNX 測試
        if os.path.exists("mobilenet_crnn_model.onnx"):
            print("\nONNX 測試:")
            onnx_result = test_onnx_local(args.image)
            print(f"  結果: {onnx_result.get('result', 'N/A')}")
            print(f"  時間: {onnx_result.get('inference_time', 0):.3f}s")
        
        # ddddocr 測試
        print("\nddddocr 測試:")
        dddd_result = test_ddddocr_local(args.image)
        print(f"  結果: {dddd_result.get('result', 'N/A')}")
        print(f"  時間: {dddd_result.get('inference_time', 0):.3f}s")


if __name__ == "__main__":
    main()