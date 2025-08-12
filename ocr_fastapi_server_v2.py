#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高性能 OCR FastAPI 服務 v2
優化並發處理和外網訪問
"""

import os
import io
import time
import asyncio
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
import logging
import warnings
import multiprocessing
from datetime import datetime
import threading
import queue

warnings.filterwarnings('ignore')

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全域變數
model = None
device = None
executor = None
model_lock = threading.Lock()  # 用於模型同步
request_queue = queue.Queue()  # 請求佇列
active_requests = 0
max_concurrent_requests = 10  # 最大並發請求數

# 字元集
CHARSET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
IDX_TO_CHAR[0] = '_'
NUM_CLASSES = len(CHARSET) + 1


class LightweightCRNN(nn.Module):
    """輕量化 CRNN 模型"""
    
    def __init__(self, num_classes=NUM_CLASSES, rnn_hidden=128, rnn_layers=2):
        super(LightweightCRNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        
        self.rnn = nn.LSTM(
            256 * 2,
            rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if rnn_layers > 1 else 0
        )
        
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.permute(0, 3, 1, 2)
        conv = conv.reshape(b, w, c * h)
        rnn_out, _ = self.rnn(conv)
        rnn_out = self.dropout(rnn_out)
        output = self.fc(rnn_out)
        output = output.permute(1, 0, 2)
        return output


def decode_predictions(preds, blank_idx=0):
    """解碼 CTC 預測結果"""
    if preds.dim() == 3:
        preds = preds.permute(1, 0, 2)
    
    preds = preds.argmax(2)
    batch_size = preds.size(0)
    
    decoded = []
    for i in range(batch_size):
        pred = preds[i]
        chars = []
        prev = blank_idx
        
        for p in pred:
            if p != blank_idx and p != prev:
                if p.item() in IDX_TO_CHAR:
                    chars.append(IDX_TO_CHAR[p.item()])
            prev = p
        
        decoded.append(''.join(chars))
    
    return decoded


def preprocess_image(image: Image.Image, img_height=32, img_width=128) -> torch.Tensor:
    """預處理圖片"""
    # 轉為灰階
    if image.mode != 'L':
        image = image.convert('L')
    
    # 轉為 numpy array
    img_array = np.array(image)
    
    # 二值化
    _, img_array = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
    
    # 調整尺寸
    h, w = img_array.shape
    scale = img_height / h
    new_w = int(w * scale)
    new_w = min(max(new_w, 80), 150)
    
    img_array = cv2.resize(img_array, (new_w, img_height), interpolation=cv2.INTER_LINEAR)
    
    # Padding
    if new_w < img_width:
        pad_width = img_width - new_w
        img_array = np.pad(img_array, ((0, 0), (0, pad_width)), mode='constant', constant_values=255)
    elif new_w > img_width:
        img_array = img_array[:, :img_width]
    
    # 正規化
    img_array = img_array.astype(np.float32) / 255.0
    
    # 轉為 tensor
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor


def predict_single(image: Image.Image) -> Dict[str, Any]:
    """預測單張圖片 - 線程安全版本"""
    start_time = time.time()
    
    # 預處理
    img_tensor = preprocess_image(image)
    img_tensor = img_tensor.to(device)
    
    # 使用鎖確保線程安全
    with model_lock:
        with torch.no_grad():
            output = model(img_tensor)
            result = decode_predictions(output)[0]
            confidence = calculate_confidence(output)
    
    inference_time = (time.time() - start_time) * 1000  # 轉為毫秒
    
    return {
        "result": result,
        "confidence": confidence,
        "inference_time_ms": round(inference_time, 2)
    }


def calculate_confidence(output: torch.Tensor) -> float:
    """計算預測信心分數"""
    # 使用 softmax 後的最大機率作為信心分數
    probs = F.softmax(output, dim=2)
    max_probs = probs.max(dim=2)[0]
    
    # 排除 blank token 的機率
    non_blank_probs = max_probs[max_probs > 0.5]
    
    if len(non_blank_probs) > 0:
        confidence = non_blank_probs.mean().item()
    else:
        confidence = 0.0
    
    return round(confidence, 3)


# 生命週期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    global model, device, executor
    
    logger.info("正在載入模型...")
    
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用設備: {device}")
    
    # 載入模型
    model = LightweightCRNN()
    
    # 嘗試載入權重
    model_path = 'best_lightweight_crnn_model.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"已載入模型權重: {model_path}")
        
        # 顯示訓練結果
        if 'val_seq_acc' in checkpoint:
            logger.info(f"模型驗證準確率: {checkpoint['val_seq_acc']:.2%}")
    else:
        logger.warning(f"找不到模型權重檔案: {model_path}")
    
    model = model.to(device)
    model.eval()
    
    # 創建執行緒池（增加工作線程數以處理並發）
    executor = ThreadPoolExecutor(max_workers=20)
    
    logger.info("模型載入完成，服務已啟動")
    logger.info(f"最大並發請求數: {max_concurrent_requests}")
    
    yield
    
    # 關閉時清理資源
    executor.shutdown(wait=True)
    logger.info("服務已關閉")


# 創建 FastAPI 應用
app = FastAPI(
    title="OCR API Service",
    description="高性能驗證碼識別服務 - 支援高並發",
    version="2.0.0",
    lifespan=lifespan
)

# 添加 CORS 中間件（允許所有來源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加信任主機中間件（防止 Host Header 攻擊）
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # 允許所有主機
)


# API 路由
@app.get("/")
async def root():
    """根路徑"""
    return {
        "service": "OCR API v2",
        "status": "running",
        "device": str(device),
        "active_requests": active_requests,
        "max_concurrent": max_concurrent_requests,
        "endpoints": [
            "/predict - 預測單張圖片（支援高並發）",
            "/health - 健康檢查",
            "/stats - 服務統計",
            "/status - 即時狀態"
        ]
    }


@app.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "active_requests": active_requests,
        "max_concurrent": max_concurrent_requests
    }


# 請求統計
request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_inference_time": 0,
    "avg_inference_time": 0,
    "min_inference_time": float('inf'),
    "max_inference_time": 0,
    "start_time": datetime.now().isoformat(),
    "concurrent_high_water_mark": 0
}


@app.get("/stats")
async def get_stats():
    """獲取服務統計"""
    uptime = (datetime.now() - datetime.fromisoformat(request_stats["start_time"])).total_seconds()
    
    return {
        **request_stats,
        "uptime_seconds": uptime,
        "requests_per_second": request_stats["total_requests"] / uptime if uptime > 0 else 0,
        "success_rate": request_stats["successful_requests"] / request_stats["total_requests"] 
                       if request_stats["total_requests"] > 0 else 0
    }


@app.get("/status")
async def get_status():
    """即時狀態"""
    return {
        "timestamp": datetime.now().isoformat(),
        "active_requests": active_requests,
        "queue_size": request_queue.qsize() if hasattr(request_queue, 'qsize') else 0,
        "available_slots": max_concurrent_requests - active_requests,
        "model_device": str(device),
        "thread_pool_active": executor._threads if hasattr(executor, '_threads') else "N/A"
    }


class PredictResponse(BaseModel):
    result: str
    confidence: float
    inference_time_ms: float
    request_id: Optional[str] = None


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    預測單張圖片 - 優化並發處理
    支援多個客戶端同時請求
    """
    global active_requests, request_stats
    
    # 檢查並發限制
    if active_requests >= max_concurrent_requests:
        raise HTTPException(
            status_code=503, 
            detail=f"服務忙碌中，當前活躍請求: {active_requests}/{max_concurrent_requests}"
        )
    
    # 生成請求 ID
    request_id = f"{datetime.now().timestamp()}_{id(file)}"
    
    try:
        # 增加活躍請求計數
        active_requests += 1
        request_stats["total_requests"] += 1
        
        # 更新並發高水位
        if active_requests > request_stats["concurrent_high_water_mark"]:
            request_stats["concurrent_high_water_mark"] = active_requests
        
        logger.info(f"處理請求 {request_id}, 當前並發: {active_requests}")
        
        # 讀取圖片
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 在執行緒池中執行預測
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            predict_single,
            image
        )
        
        # 更新統計
        request_stats["successful_requests"] += 1
        request_stats["total_inference_time"] += result["inference_time_ms"]
        request_stats["avg_inference_time"] = (
            request_stats["total_inference_time"] / request_stats["successful_requests"]
        )
        
        if result["inference_time_ms"] < request_stats["min_inference_time"]:
            request_stats["min_inference_time"] = result["inference_time_ms"]
        if result["inference_time_ms"] > request_stats["max_inference_time"]:
            request_stats["max_inference_time"] = result["inference_time_ms"]
        
        result["request_id"] = request_id
        
        logger.info(f"請求 {request_id} 完成, 結果: {result['result']}, 耗時: {result['inference_time_ms']}ms")
        
        return result
        
    except Exception as e:
        request_stats["failed_requests"] += 1
        logger.error(f"請求 {request_id} 失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 減少活躍請求計數
        active_requests -= 1


# 批次處理端點（保留但說明用途不同）
@app.post("/predict_batch")
async def predict_batch_deprecated():
    """
    此端點已棄用
    請使用 /predict 端點進行並發請求
    """
    return {
        "message": "此端點已棄用。請直接對 /predict 端點發送多個並發請求。",
        "recommendation": "使用多線程或異步方式同時發送多個請求到 /predict 端點"
    }


def run_server(host="0.0.0.0", port=8000, workers=1):
    """
    啟動服務器
    host="0.0.0.0" 允許外網訪問
    """
    logger.info(f"啟動服務器: {host}:{port}")
    logger.info("外網訪問: 確保防火牆開放對應端口")
    
    if workers > 1:
        # 多進程模式（生產環境）
        uvicorn.run(
            "ocr_fastapi_server_v2:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info",
            access_log=True
        )
    else:
        # 單進程模式（開發環境）
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )


def main():
    """主程式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR FastAPI 服務 v2 - 高並發版本')
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                       help='服務主機 (0.0.0.0 允許外網訪問)')
    parser.add_argument('--port', type=int, default=8000, help='服務埠')
    parser.add_argument('--workers', type=int, default=1, 
                       help='工作進程數 (>1 時使用多進程模式)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OCR API 服務 v2 - 高並發版本")
    print("=" * 60)
    print(f"監聽地址: {args.host}:{args.port}")
    print(f"工作進程: {args.workers}")
    print()
    print("外網訪問說明:")
    print("1. 確保防火牆開放端口", args.port)
    print("2. 使用公網 IP 或域名訪問")
    print("3. 範例: http://your-public-ip:8000/predict")
    print("=" * 60)
    
    run_server(args.host, args.port, args.workers)


if __name__ == "__main__":
    main()