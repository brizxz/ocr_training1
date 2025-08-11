#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR API 服務
使用 FastAPI 提供高效能的驗證碼識別服務
支援 PyTorch 和 ONNX Runtime
"""

import os
import io
import base64
import time
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from contextlib import asynccontextmanager

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全域變數
model = None
onnx_session = None
device = None
CHARSET = None
CHAR_TO_IDX = None
IDX_TO_CHAR = None
use_onnx = False

# 請求模型
class OCRRequest(BaseModel):
    image: str  # Base64 編碼的圖片
    format: Optional[str] = "base64"  # 圖片格式: base64 或 url

class OCRResponse(BaseModel):
    success: bool
    result: Optional[str]
    confidence: Optional[float]
    processing_time: Optional[float]
    error: Optional[str]

# 從 train_ocr_model_v2.py 複製模型定義
class MobileNetV3_CRNN(nn.Module):
    """MobileNetV3-Small + BiLSTM + CTC 模型"""
    
    def __init__(self, num_classes=37, rnn_hidden=256, rnn_layers=2):
        super(MobileNetV3_CRNN, self).__init__()
        
        from torchvision import models
        
        # 使用預訓練的 MobileNetV3-Small
        mobilenet = models.mobilenet_v3_small(pretrained=False)
        self.cnn = mobilenet.features
        
        # 修改第一層以接受單通道輸入
        original_conv = self.cnn[0][0]
        self.cnn[0][0] = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # 調整特徵圖
        self.adapter = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # 雙向 LSTM
        self.rnn = nn.LSTM(
            256, rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if rnn_layers > 1 else 0
        )
        
        # 輸出層
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        import torch.nn.functional as F
        
        conv = self.cnn(x)
        conv = self.adapter(conv)
        
        b, c, h, w = conv.size()
        if h > 1:
            conv = F.adaptive_avg_pool2d(conv, (1, None))
        
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        
        rnn_out, _ = self.rnn(conv)
        rnn_out = self.dropout(rnn_out)
        output = self.fc(rnn_out)
        output = output.permute(1, 0, 2)
        
        return output


def load_pytorch_model(model_path: str, device_type: str = 'cpu'):
    """載入 PyTorch 模型"""
    global model, device, CHARSET, CHAR_TO_IDX, IDX_TO_CHAR
    
    device = torch.device(device_type)
    
    # 載入檢查點
    checkpoint = torch.load(model_path, map_location=device)
    
    # 載入字元集
    if 'charset' in checkpoint:
        CHARSET = checkpoint['charset']
    else:
        CHARSET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}
    IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
    IDX_TO_CHAR[0] = '_'
    
    # 創建模型
    model = MobileNetV3_CRNN(num_classes=len(CHARSET) + 1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"PyTorch 模型載入成功，使用設備: {device}")
    return model


def load_onnx_model(model_path: str):
    """載入 ONNX 模型"""
    global onnx_session, CHARSET, CHAR_TO_IDX, IDX_TO_CHAR
    
    try:
        import onnxruntime as ort
        
        # 設定 ONNX Runtime
        providers = ['CPUExecutionProvider']
        
        # 如果有 GPU，優先使用
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        onnx_session = ort.InferenceSession(model_path, providers=providers)
        
        # 設定字元集（需要與訓練時一致）
        CHARSET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}
        IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
        IDX_TO_CHAR[0] = '_'
        
        logger.info(f"ONNX 模型載入成功，使用 providers: {providers}")
        return onnx_session
    except ImportError:
        logger.error("onnxruntime 未安裝，請執行: pip install onnxruntime")
        return None


def preprocess_image(image: Image.Image, target_height: int = 32, target_width: int = 128) -> np.ndarray:
    """預處理圖片"""
    # 轉為灰階
    if image.mode != 'L':
        image = image.convert('L')
    
    img_array = np.array(image)
    
    # 二值化
    _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 去噪
    img_array = cv2.medianBlur(img_array, 3)
    
    # 調整尺寸
    h, w = img_array.shape
    scale = target_height / h
    new_w = int(w * scale)
    new_w = min(max(new_w, 100), 160)
    
    img_array = cv2.resize(img_array, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
    
    # Padding
    if new_w < target_width:
        pad_width = target_width - new_w
        img_array = np.pad(img_array, ((0, 0), (0, pad_width)), mode='constant', constant_values=255)
    elif new_w > target_width:
        img_array = img_array[:, :target_width]
    
    # 正規化
    img_array = img_array.astype(np.float32) / 255.0
    
    return img_array


def decode_ctc_output(output: np.ndarray, blank_idx: int = 0) -> tuple:
    """解碼 CTC 輸出"""
    # output: [T, B, C] or [T, C] for single batch
    if output.ndim == 3:
        output = output[:, 0, :]  # 取第一個 batch
    
    # 取最大機率的類別
    predictions = np.argmax(output, axis=1)
    
    # CTC 解碼
    chars = []
    confidence_scores = []
    prev = blank_idx
    
    for i, pred in enumerate(predictions):
        if pred != blank_idx and pred != prev:
            if pred in IDX_TO_CHAR:
                chars.append(IDX_TO_CHAR[pred])
                # 計算該字元的信心分數
                confidence_scores.append(output[i, pred])
        prev = pred
    
    result = ''.join(chars)
    
    # 計算平均信心分數
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
    
    return result, float(avg_confidence)


def predict_pytorch(image: Image.Image) -> tuple:
    """使用 PyTorch 模型預測"""
    global model, device
    
    if model is None:
        raise ValueError("PyTorch 模型未載入")
    
    # 預處理
    img_array = preprocess_image(image)
    
    # 轉為 tensor [B, C, H, W]
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # 預測
    with torch.no_grad():
        output = model(img_tensor)  # [T, B, C]
        output = output.cpu().numpy()
    
    # 解碼
    result, confidence = decode_ctc_output(output)
    
    return result, confidence


def predict_onnx(image: Image.Image) -> tuple:
    """使用 ONNX 模型預測"""
    global onnx_session
    
    if onnx_session is None:
        raise ValueError("ONNX 模型未載入")
    
    # 預處理
    img_array = preprocess_image(image)
    
    # 轉為 [B, C, H, W] 格式
    img_input = img_array[np.newaxis, np.newaxis, :, :].astype(np.float32)
    
    # 預測
    input_name = onnx_session.get_inputs()[0].name
    output = onnx_session.run(None, {input_name: img_input})[0]
    
    # 解碼
    result, confidence = decode_ctc_output(output)
    
    return result, confidence


# 應用程式生命週期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式啟動和關閉時的處理"""
    # 啟動時載入模型
    global use_onnx
    
    # 優先嘗試載入 ONNX 模型（更快）
    if os.path.exists('mobilenet_crnn_model.onnx'):
        if load_onnx_model('mobilenet_crnn_model.onnx'):
            use_onnx = True
            logger.info("使用 ONNX 模型")
    
    # 如果沒有 ONNX，載入 PyTorch 模型
    if not use_onnx and os.path.exists('best_mobilenet_crnn_model.pth'):
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        load_pytorch_model('best_mobilenet_crnn_model.pth', device_type)
        logger.info("使用 PyTorch 模型")
    
    if not use_onnx and model is None and onnx_session is None:
        logger.error("找不到模型檔案！請先訓練模型。")
    
    yield
    
    # 關閉時清理資源
    logger.info("API 服務關閉")


# 創建 FastAPI 應用
app = FastAPI(
    title="TixCraft OCR API",
    description="高效能驗證碼識別服務",
    version="2.0",
    lifespan=lifespan
)

# 添加 CORS 支援
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """根路徑"""
    return {
        "service": "TixCraft OCR API",
        "version": "2.0",
        "status": "running",
        "model": "ONNX" if use_onnx else "PyTorch",
        "endpoints": {
            "/predict": "POST - 預測驗證碼",
            "/health": "GET - 健康檢查",
            "/stats": "GET - 服務統計"
        }
    }


@app.get("/health")
async def health_check():
    """健康檢查"""
    model_loaded = (model is not None) or (onnx_session is not None)
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_type": "ONNX" if use_onnx else "PyTorch",
        "model_loaded": model_loaded
    }


# 統計資訊
stats = {
    "total_requests": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "average_processing_time": 0,
    "total_processing_time": 0
}


@app.get("/stats")
async def get_stats():
    """獲取服務統計"""
    return stats


@app.post("/predict", response_model=OCRResponse)
async def predict(request: OCRRequest):
    """預測驗證碼"""
    global stats
    
    start_time = time.time()
    stats["total_requests"] += 1
    
    try:
        # 解碼圖片
        if request.format == "base64":
            # Base64 解碼
            image_data = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(image_data))
        else:
            return OCRResponse(
                success=False,
                result=None,
                confidence=None,
                processing_time=None,
                error="不支援的圖片格式"
            )
        
        # 預測
        if use_onnx:
            result, confidence = predict_onnx(image)
        else:
            result, confidence = predict_pytorch(image)
        
        # 計算處理時間
        processing_time = time.time() - start_time
        
        # 更新統計
        stats["successful_predictions"] += 1
        stats["total_processing_time"] += processing_time
        stats["average_processing_time"] = stats["total_processing_time"] / stats["successful_predictions"]
        
        # 驗證結果（TixCraft 驗證碼應該是4位）
        if len(result) != 4:
            logger.warning(f"預測結果長度異常: {result} (長度: {len(result)})")
        
        return OCRResponse(
            success=True,
            result=result,
            confidence=confidence,
            processing_time=processing_time,
            error=None
        )
        
    except Exception as e:
        stats["failed_predictions"] += 1
        logger.error(f"預測錯誤: {e}")
        return OCRResponse(
            success=False,
            result=None,
            confidence=None,
            processing_time=None,
            error=str(e)
        )


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    """從上傳的檔案預測驗證碼"""
    global stats
    
    start_time = time.time()
    stats["total_requests"] += 1
    
    try:
        # 讀取圖片
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 預測
        if use_onnx:
            result, confidence = predict_onnx(image)
        else:
            result, confidence = predict_pytorch(image)
        
        # 計算處理時間
        processing_time = time.time() - start_time
        
        # 更新統計
        stats["successful_predictions"] += 1
        stats["total_processing_time"] += processing_time
        stats["average_processing_time"] = stats["total_processing_time"] / stats["successful_predictions"]
        
        return {
            "success": True,
            "result": result,
            "confidence": confidence,
            "processing_time": processing_time
        }
        
    except Exception as e:
        stats["failed_predictions"] += 1
        logger.error(f"預測錯誤: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """啟動服務"""
    uvicorn.run(
        "ocr_api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR API 服務')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服務主機')
    parser.add_argument('--port', type=int, default=8000, help='服務端口')
    parser.add_argument('--reload', action='store_true', help='自動重載（開發模式）')
    
    args = parser.parse_args()
    
    print(f"啟動 OCR API 服務: http://{args.host}:{args.port}")
    print(f"API 文檔: http://{args.host}:{args.port}/docs")
    
    run_server(args.host, args.port, args.reload)