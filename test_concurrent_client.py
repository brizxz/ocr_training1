#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
並發測試客戶端
測試 OCR API 的並發處理能力
"""

import asyncio
import aiohttp
import time
import sys
from pathlib import Path
import random
import argparse
from typing import List
import json

async def predict_single_async(session, url, image_path):
    """異步預測單張圖片"""
    start_time = time.time()
    
    try:
        with open(image_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=image_path.name)
            
            async with session.post(url, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    total_time = (time.time() - start_time) * 1000
                    
                    return {
                        "success": True,
                        "image": image_path.name,
                        "result": result["result"],
                        "confidence": result["confidence"],
                        "server_time": result["inference_time_ms"],
                        "total_time": round(total_time, 2),
                        "network_time": round(total_time - result["inference_time_ms"], 2)
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "image": image_path.name,
                        "error": f"HTTP {response.status}: {error_text}",
                        "total_time": round((time.time() - start_time) * 1000, 2)
                    }
    except Exception as e:
        return {
            "success": False,
            "image": image_path.name,
            "error": str(e),
            "total_time": round((time.time() - start_time) * 1000, 2)
        }


async def test_concurrent_requests(url, image_paths, concurrent_count):
    """測試並發請求"""
    print(f"\n開始並發測試:")
    print(f"  API URL: {url}")
    print(f"  圖片數量: {len(image_paths)}")
    print(f"  並發數: {concurrent_count}")
    print("-" * 60)
    
    # 創建會話
    connector = aiohttp.TCPConnector(limit=concurrent_count)
    async with aiohttp.ClientSession(connector=connector) as session:
        # 創建所有任務
        tasks = []
        for i, image_path in enumerate(image_paths):
            # 添加隨機延遲以模擬真實場景
            delay = random.uniform(0, 0.1) * i / concurrent_count
            await asyncio.sleep(delay)
            
            task = predict_single_async(session, url, image_path)
            tasks.append(task)
            
            # 控制並發數
            if len(tasks) >= concurrent_count:
                # 等待一批完成
                batch_results = await asyncio.gather(*tasks[:concurrent_count])
                for result in batch_results:
                    print_result(result)
                tasks = tasks[concurrent_count:]
        
        # 處理剩餘的任務
        if tasks:
            batch_results = await asyncio.gather(*tasks)
            for result in batch_results:
                print_result(result)


def print_result(result):
    """打印單個結果"""
    if result["success"]:
        print(f"✓ {result['image']}: {result['result']} "
              f"(信心: {result['confidence']:.2f}, "
              f"服務器: {result['server_time']}ms, "
              f"總耗時: {result['total_time']}ms)")
    else:
        print(f"✗ {result['image']}: {result['error']}")


async def benchmark_concurrent_performance(url, image_path, test_counts):
    """性能基準測試"""
    print("\n性能基準測試:")
    print("=" * 60)
    
    results = {}
    
    for count in test_counts:
        print(f"\n測試 {count} 個並發請求...")
        
        # 準備圖片列表
        image_paths = [image_path] * count
        
        start_time = time.time()
        
        # 創建會話
        connector = aiohttp.TCPConnector(limit=count)
        async with aiohttp.ClientSession(connector=connector) as session:
            # 創建所有任務並同時發送
            tasks = [predict_single_async(session, url, img) for img in image_paths]
            results_list = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # 統計結果
        success_count = sum(1 for r in results_list if r["success"])
        avg_server_time = sum(r.get("server_time", 0) for r in results_list if r["success"]) / max(success_count, 1)
        avg_total_time = sum(r["total_time"] for r in results_list) / len(results_list)
        
        results[count] = {
            "concurrent": count,
            "total_time": round(total_time, 3),
            "requests_per_second": round(count / total_time, 2),
            "success_rate": success_count / count,
            "avg_server_time": round(avg_server_time, 2),
            "avg_total_time": round(avg_total_time, 2)
        }
        
        print(f"  完成: {success_count}/{count} 成功")
        print(f"  總耗時: {total_time:.3f} 秒")
        print(f"  吞吐量: {results[count]['requests_per_second']} req/s")
        print(f"  平均服務器處理: {avg_server_time:.2f} ms")
        print(f"  平均總耗時: {avg_total_time:.2f} ms")
    
    # 打印總結
    print("\n" + "=" * 60)
    print("性能測試總結:")
    print("-" * 60)
    print(f"{'並發數':>8} | {'吞吐量(req/s)':>15} | {'成功率':>8} | {'平均延遲(ms)':>12}")
    print("-" * 60)
    
    for count, data in results.items():
        print(f"{data['concurrent']:>8} | {data['requests_per_second']:>15.2f} | "
              f"{data['success_rate']:>7.1%} | {data['avg_total_time']:>12.2f}")
    
    return results


def test_synchronous_requests(url, image_paths):
    """同步請求測試（對比用）"""
    import requests
    
    print("\n同步請求測試（對比）:")
    print("-" * 60)
    
    start_time = time.time()
    
    for image_path in image_paths:
        with open(image_path, 'rb') as f:
            response = requests.post(url, files={'file': f})
            if response.status_code == 200:
                result = response.json()
                print(f"✓ {image_path.name}: {result['result']} "
                      f"(耗時: {result['inference_time_ms']}ms)")
            else:
                print(f"✗ {image_path.name}: HTTP {response.status_code}")
    
    total_time = time.time() - start_time
    print(f"\n同步處理 {len(image_paths)} 個請求總耗時: {total_time:.3f} 秒")
    print(f"平均每個請求: {total_time/len(image_paths)*1000:.2f} ms")


async def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='OCR API 並發測試客戶端')
    parser.add_argument('--url', type=str, default='http://localhost:8000/predict',
                       help='API URL (可以使用外網地址)')
    parser.add_argument('--image-dir', type=str, 
                       default='captcha_auto_label/merged_20250811_155009',
                       help='圖片目錄')
    parser.add_argument('--count', type=int, default=10,
                       help='測試圖片數量')
    parser.add_argument('--concurrent', type=int, default=5,
                       help='並發數')
    parser.add_argument('--benchmark', action='store_true',
                       help='執行性能基準測試')
    
    args = parser.parse_args()
    
    # 獲取圖片列表
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"錯誤: 找不到目錄 {image_dir}")
        sys.exit(1)
    
    image_paths = list(image_dir.glob("*.png"))[:args.count]
    
    if not image_paths:
        print(f"錯誤: 在 {image_dir} 中找不到圖片")
        sys.exit(1)
    
    print("=" * 60)
    print("OCR API 並發測試")
    print("=" * 60)
    
    # 檢查服務器狀態
    import aiohttp
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(args.url.replace('/predict', '/health')) as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"服務器狀態: {health['status']}")
                    print(f"設備: {health['device']}")
                    print(f"當前活躍請求: {health.get('active_requests', 'N/A')}")
                    print(f"最大並發: {health.get('max_concurrent', 'N/A')}")
        except Exception as e:
            print(f"警告: 無法連接到服務器 - {e}")
    
    if args.benchmark:
        # 執行基準測試
        test_image = image_paths[0]
        test_counts = [1, 2, 5, 10, 20]
        await benchmark_concurrent_performance(args.url, test_image, test_counts)
    else:
        # 一般並發測試
        await test_concurrent_requests(args.url, image_paths, args.concurrent)
        
        # 對比同步請求
        if len(image_paths) <= 5:
            test_synchronous_requests(args.url, image_paths)
    
    # 顯示服務器統計
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(args.url.replace('/predict', '/stats')) as response:
                if response.status == 200:
                    stats = await response.json()
                    print("\n" + "=" * 60)
                    print("服務器統計:")
                    print(f"  總請求數: {stats['total_requests']}")
                    print(f"  成功率: {stats.get('success_rate', 0):.1%}")
                    print(f"  平均推理時間: {stats['avg_inference_time']:.2f} ms")
                    print(f"  最大並發數: {stats.get('concurrent_high_water_mark', 'N/A')}")
        except:
            pass


if __name__ == "__main__":
    # 安裝必要套件: pip install aiohttp requests
    asyncio.run(main())