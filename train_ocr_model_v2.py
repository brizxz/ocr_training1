#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
訓練自訂 OCR 模型 v2
基於 MobileNetV3-Small + BiLSTM + CTC Loss 架構
針對 TixCraft 驗證碼優化
"""

import os
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from datetime import datetime

# 字元集（數字 + 大寫字母）
CHARSET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}  # 0 保留給 CTC blank
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
IDX_TO_CHAR[0] = '_'  # blank token
NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank

class CaptchaDataset(Dataset):
    """驗證碼資料集 - 增強版"""
    
    def __init__(self, data_dir, labels_file, training=True, img_height=32, img_width=128):
        """
        Args:
            data_dir: 圖片目錄
            labels_file: 標註檔案 (training_data.txt)
            training: 是否為訓練模式（決定是否使用資料增強）
            img_height: 圖片高度（固定）
            img_width: 目標圖片寬度
        """
        self.data_dir = data_dir
        self.training = training
        self.img_height = img_height
        self.img_width = img_width
        self.samples = []
        
        # 載入標註
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and ',' in line:
                    parts = line.split(',')
                    filename = parts[0]
                    label = parts[1] if len(parts) > 1 else ''
                    # 只載入有標註的資料
                    if label and len(label) == 4:  # TixCraft 驗證碼固定4位
                        # 確保標籤都是大寫
                        label = label.upper()
                        # 檢查標籤字元是否都在字元集中
                        if all(c in CHARSET for c in label):
                            self.samples.append((filename, label))
        
        print(f"載入 {len(self.samples)} 個有效樣本")
    
    def __len__(self):
        return len(self.samples)
    
    def preprocess_image(self, image):
        """預處理圖片 - 參考 GPT-5 建議"""
        # 轉為 numpy array
        img_array = np.array(image)
        
        # 轉灰階
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 二值化 (OTSU 自適應閾值)
        _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 去噪（中值濾波）
        img_array = cv2.medianBlur(img_array, 3)
        
        # 調整尺寸（保持長寬比）
        h, w = img_array.shape
        scale = self.img_height / h
        new_w = int(w * scale)
        
        # 限制寬度在合理範圍內
        new_w = min(max(new_w, 100), 160)
        
        img_array = cv2.resize(img_array, (new_w, self.img_height), interpolation=cv2.INTER_LINEAR)
        
        # Padding 到固定寬度
        if new_w < self.img_width:
            pad_width = self.img_width - new_w
            img_array = np.pad(img_array, ((0, 0), (0, pad_width)), mode='constant', constant_values=255)
        elif new_w > self.img_width:
            img_array = img_array[:, :self.img_width]
        
        return img_array
    
    def augment_image(self, image):
        """資料增強 - 訓練時使用"""
        if not self.training:
            return image
        
        # 隨機旋轉（小角度）
        if random.random() > 0.5:
            angle = random.uniform(-3, 3)
            image = image.rotate(angle, fillcolor=255)
        
        # 隨機扭曲
        if random.random() > 0.5:
            # 使用 PIL 的透視變換
            width, height = image.size
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            
            # 定義變換點
            coeffs = [
                1 + random.uniform(-0.05, 0.05),  # a
                random.uniform(-0.05, 0.05),       # b
                dx,                                 # c
                random.uniform(-0.05, 0.05),       # d
                1 + random.uniform(-0.05, 0.05),  # e
                dy,                                 # f
                random.uniform(-0.0001, 0.0001),   # g
                random.uniform(-0.0001, 0.0001)    # h
            ]
            image = image.transform(image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
        
        # 隨機噪點
        if random.random() > 0.5:
            img_array = np.array(image)
            noise = np.random.randint(0, 50, img_array.shape, dtype=np.uint8)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        
        # 隨機亮度調整
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        return image
    
    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        
        # 載入圖片
        img_path = os.path.join(self.data_dir, filename)
        try:
            image = Image.open(img_path).convert('L')  # 確保是灰階
        except Exception as e:
            print(f"無法載入圖片 {img_path}: {e}")
            # 返回空白圖片
            image = Image.new('L', (self.img_width, self.img_height), 255)
        
        # 資料增強（訓練時）
        if self.training:
            image = self.augment_image(image)
        
        # 預處理
        img_array = self.preprocess_image(image)
        
        # 正規化到 [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        # 轉為 tensor [C, H, W]
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0)
        
        # 標籤編碼
        label_encoded = [CHAR_TO_IDX.get(char, 0) for char in label]
        label_tensor = torch.LongTensor(label_encoded)
        
        return img_tensor, label_tensor, label


class MobileNetV3_CRNN(nn.Module):
    """MobileNetV3-Small + BiLSTM + CTC 模型"""
    
    def __init__(self, num_classes=NUM_CLASSES, rnn_hidden=256, rnn_layers=2):
        super(MobileNetV3_CRNN, self).__init__()
        
        # 使用預訓練的 MobileNetV3-Small 作為特徵提取器
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        
        # 移除分類層，保留特徵提取部分
        # MobileNetV3 的特徵提取器到 features 層
        self.cnn = mobilenet.features
        
        # 修改第一層以接受單通道輸入（灰階圖）
        original_conv = self.cnn[0][0]
        self.cnn[0][0] = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # 凍結部分層以加速訓練（可選）
        for param in self.cnn[:8].parameters():
            param.requires_grad = False
        
        # 計算 CNN 輸出維度
        # MobileNetV3-Small 最後輸出通道數是 576
        cnn_output_channels = 576
        
        # 調整特徵圖以適應序列模型
        self.adapter = nn.Sequential(
            nn.Conv2d(cnn_output_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # 雙向 LSTM
        self.rnn = nn.LSTM(
            256,  # 輸入特徵維度
            rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if rnn_layers > 1 else 0
        )
        
        # 輸出層
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)
        
        # Dropout 層
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # CNN 特徵提取
        conv = self.cnn(x)  # [B, 576, H', W']
        
        # 適配層
        conv = self.adapter(conv)  # [B, 256, H', W']
        
        # 獲取維度
        b, c, h, w = conv.size()
        
        # 確保高度為1（透過池化）
        if h > 1:
            conv = F.adaptive_avg_pool2d(conv, (1, None))
            h = 1
        
        # 轉換為序列格式 [B, W, C]
        conv = conv.squeeze(2)  # [B, C, W]
        conv = conv.permute(0, 2, 1)  # [B, W, C]
        
        # RNN
        rnn_out, _ = self.rnn(conv)  # [B, W, hidden*2]
        
        # Dropout
        rnn_out = self.dropout(rnn_out)
        
        # 輸出層
        output = self.fc(rnn_out)  # [B, W, num_classes]
        
        # 轉換為 CTC 需要的格式 [T, B, C]
        output = output.permute(1, 0, 2)
        
        return output


def decode_predictions(preds, blank_idx=0):
    """解碼 CTC 預測結果"""
    # preds: [T, B, C] -> [B, T, C]
    if preds.dim() == 3:
        preds = preds.permute(1, 0, 2)
    
    preds = preds.argmax(2)  # [B, T]
    batch_size = preds.size(0)
    
    decoded = []
    for i in range(batch_size):
        pred = preds[i]
        chars = []
        prev = blank_idx
        
        for p in pred:
            # CTC 解碼：移除重複和空白
            if p != blank_idx and p != prev:
                if p.item() in IDX_TO_CHAR:
                    chars.append(IDX_TO_CHAR[p.item()])
            prev = p
        
        decoded.append(''.join(chars))
    
    return decoded


def calculate_accuracy(predictions, targets):
    """計算字元級和序列級準確率"""
    correct_sequences = 0
    total_chars = 0
    correct_chars = 0
    
    for pred, target in zip(predictions, targets):
        if pred == target:
            correct_sequences += 1
        
        # 字元級準確率
        for i in range(min(len(pred), len(target))):
            total_chars += 1
            if i < len(pred) and i < len(target) and pred[i] == target[i]:
                correct_chars += 1
        
        # 計算多餘或缺少的字元
        total_chars += abs(len(pred) - len(target))
    
    seq_acc = correct_sequences / len(predictions) if predictions else 0
    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    
    return seq_acc, char_acc


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    for batch_idx, (images, labels, label_strs) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向傳播
        outputs = model(images)  # [T, B, C]
        
        # 計算 CTC Loss
        T = outputs.size(0)  # 序列長度
        B = outputs.size(1)  # batch size
        
        input_lengths = torch.full((B,), T, dtype=torch.long)
        target_lengths = torch.full((B,), labels.size(1), dtype=torch.long)
        
        # Flatten labels for CTC
        labels_flat = labels.view(-1)
        
        loss = criterion(
            outputs.log_softmax(2),
            labels_flat,
            input_lengths,
            target_lengths
        )
        
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # 解碼預測結果
        with torch.no_grad():
            decoded = decode_predictions(outputs)
            all_predictions.extend(decoded)
            all_targets.extend(label_strs)
        
        # 每10個batch顯示一次進度
        if batch_idx % 10 == 0:
            seq_acc, char_acc = calculate_accuracy(decoded, label_strs)
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, "
                  f"Seq Acc: {seq_acc:.2%}, Char Acc: {char_acc:.2%}")
    
    # 計算整體準確率
    seq_acc, char_acc = calculate_accuracy(all_predictions, all_targets)
    
    return total_loss / len(dataloader), seq_acc, char_acc


def evaluate(model, dataloader, criterion, device):
    """評估模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels, label_strs in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            T = outputs.size(0)
            B = outputs.size(1)
            
            input_lengths = torch.full((B,), T, dtype=torch.long)
            target_lengths = torch.full((B,), labels.size(1), dtype=torch.long)
            labels_flat = labels.view(-1)
            
            loss = criterion(
                outputs.log_softmax(2),
                labels_flat,
                input_lengths,
                target_lengths
            )
            
            total_loss += loss.item()
            
            decoded = decode_predictions(outputs)
            all_predictions.extend(decoded)
            all_targets.extend(label_strs)
    
    seq_acc, char_acc = calculate_accuracy(all_predictions, all_targets)
    
    # 顯示一些預測範例
    print("\n預測範例:")
    for i in range(min(5, len(all_predictions))):
        print(f"  真實: {all_targets[i]}, 預測: {all_predictions[i]}")
    
    return total_loss / len(dataloader), seq_acc, char_acc


def train_model(data_dir, labels_file, epochs=100, batch_size=64, lr=1e-3, device='cuda'):
    """訓練模型主函數"""
    # 檢查設備
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA 不可用，使用 CPU")
    
    device = torch.device(device)
    print(f"使用裝置: {device}")
    
    # 載入資料集
    dataset = CaptchaDataset(data_dir, labels_file, training=True)
    
    if len(dataset) == 0:
        print("錯誤：沒有有效的訓練資料！")
        return None
    
    # 分割訓練集和驗證集 (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 驗證集不使用資料增強
    val_dataset.dataset.training = False
    
    print(f"訓練集: {len(train_dataset)} 樣本")
    print(f"驗證集: {len(val_dataset)} 樣本")
    
    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 建立模型
    model = MobileNetV3_CRNN().to(device)
    
    # 計算模型參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數量: {total_params:,}")
    print(f"可訓練參數量: {trainable_params:,}")
    
    # 損失函數和優化器
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # 使用 AdamW 優化器（GPT-5 推薦）
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 學習率調度器 - Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 訓練歷史
    history = {
        'train_loss': [],
        'train_seq_acc': [],
        'train_char_acc': [],
        'val_loss': [],
        'val_seq_acc': [],
        'val_char_acc': []
    }
    
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    print("\n開始訓練...")
    print("=" * 60)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"學習率: {scheduler.get_last_lr()[0]:.6f}")
        
        # 訓練
        train_loss, train_seq_acc, train_char_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 驗證
        val_loss, val_seq_acc, val_char_acc = evaluate(
            model, val_loader, criterion, device
        )
        
        # 更新學習率
        scheduler.step()
        
        # 記錄歷史
        history['train_loss'].append(train_loss)
        history['train_seq_acc'].append(train_seq_acc)
        history['train_char_acc'].append(train_char_acc)
        history['val_loss'].append(val_loss)
        history['val_seq_acc'].append(val_seq_acc)
        history['val_char_acc'].append(val_char_acc)
        
        print(f"\n訓練 - Loss: {train_loss:.4f}, Seq Acc: {train_seq_acc:.2%}, Char Acc: {train_char_acc:.2%}")
        print(f"驗證 - Loss: {val_loss:.4f}, Seq Acc: {val_seq_acc:.2%}, Char Acc: {val_char_acc:.2%}")
        
        # 儲存最佳模型
        if val_seq_acc > best_val_acc:
            best_val_acc = val_seq_acc
            patience_counter = 0
            
            # 儲存完整模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_seq_acc': val_seq_acc,
                'val_char_acc': val_char_acc,
                'charset': CHARSET,
                'history': history
            }, 'best_mobilenet_crnn_model.pth')
            
            print(f"✓ 儲存最佳模型 (序列準確率: {val_seq_acc:.2%})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停：驗證準確率已經 {patience} 個 epoch 沒有改善")
                break
        
        print("-" * 60)
    
    print(f"\n訓練完成！最佳驗證序列準確率: {best_val_acc:.2%}")
    
    # 繪製訓練曲線
    plot_training_history(history)
    
    return model, history


def plot_training_history(history):
    """繪製訓練歷史圖表"""
    plt.figure(figsize=(15, 5))
    
    # Loss 曲線
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # 序列準確率曲線
    plt.subplot(1, 3, 2)
    plt.plot(history['train_seq_acc'], label='Train Seq Acc')
    plt.plot(history['val_seq_acc'], label='Val Seq Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Sequence Accuracy')
    plt.title('Sequence Accuracy Curve')
    plt.legend()
    plt.grid(True)
    
    # 字元準確率曲線
    plt.subplot(1, 3, 3)
    plt.plot(history['train_char_acc'], label='Train Char Acc')
    plt.plot(history['val_char_acc'], label='Val Char Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Character Accuracy')
    plt.title('Character Accuracy Curve')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100)
    print("訓練曲線已儲存: training_history.png")


def export_onnx(model, save_path='mobilenet_crnn_model.onnx', device='cpu'):
    """匯出 ONNX 模型以供部署"""
    model.eval()
    model = model.to(device)
    
    # 創建範例輸入
    dummy_input = torch.randn(1, 1, 32, 128).to(device)
    
    # 匯出 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'seq_length', 1: 'batch_size'}
        }
    )
    
    print(f"ONNX 模型已匯出: {save_path}")
    
    # 驗證 ONNX 模型
    try:
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型驗證成功")
    except Exception as e:
        print(f"ONNX 模型驗證失敗: {e}")


def main():
    """主程式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='訓練 MobileNetV3-CRNN OCR 模型')
    parser.add_argument('--data_dir', type=str, required=True, help='圖片目錄')
    parser.add_argument('--labels', type=str, required=True, help='標註檔案 (training_data.txt)')
    parser.add_argument('--epochs', type=int, default=100, help='訓練 epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='學習率')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='訓練設備')
    parser.add_argument('--export_onnx', action='store_true', help='匯出 ONNX 模型')
    
    args = parser.parse_args()
    
    # 訓練模型
    model, history = train_model(
        args.data_dir,
        args.labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
    
    if model and args.export_onnx:
        # 載入最佳模型
        checkpoint = torch.load('best_mobilenet_crnn_model.pth', map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 匯出 ONNX
        export_onnx(model, device=args.device)


if __name__ == "__main__":
    main()