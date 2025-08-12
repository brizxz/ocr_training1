#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
訓練自訂 OCR 模型 v3 - 改進版
解決過擬合問題，提升泛化能力和推理速度
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
from torchvision import transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 字元集（數字 + 大寫字母）
CHARSET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}  # 0 保留給 CTC blank
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
IDX_TO_CHAR[0] = '_'  # blank token
NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank

class CaptchaDataset(Dataset):
    """驗證碼資料集 - 強化版"""
    
    def __init__(self, data_dir, labels_file, training=True, img_height=32, img_width=128):
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
                    
                    # 確保標籤都是大寫（修正小寫標籤問題）
                    label = label.upper()
                    
                    # 只載入有效的4位標註
                    if label and len(label) == 4:
                        # 檢查標籤字元是否都在字元集中
                        if all(c in CHARSET for c in label):
                            self.samples.append((filename, label))
        
        print(f"載入 {len(self.samples)} 個有效樣本")
        
        # 分析標籤分布
        if len(self.samples) > 0:
            labels = [s[1] for s in self.samples[:100]]
            print(f"標籤範例: {labels[:5]}")
    
    def __len__(self):
        return len(self.samples)
    
    def preprocess_image(self, image):
        """預處理圖片 - 簡化版本減少過處理"""
        # 轉為 numpy array
        img_array = np.array(image)
        
        # 轉灰階
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 簡單的二值化（避免過度處理）
        _, img_array = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
        
        # 調整尺寸（保持長寬比）
        h, w = img_array.shape
        scale = self.img_height / h
        new_w = int(w * scale)
        
        # 限制寬度
        new_w = min(max(new_w, 80), 150)
        
        img_array = cv2.resize(img_array, (new_w, self.img_height), interpolation=cv2.INTER_LINEAR)
        
        # Padding 到固定寬度
        if new_w < self.img_width:
            pad_width = self.img_width - new_w
            img_array = np.pad(img_array, ((0, 0), (0, pad_width)), mode='constant', constant_values=255)
        elif new_w > self.img_width:
            img_array = img_array[:, :self.img_width]
        
        return img_array
    
    def augment_image(self, image):
        """強化的資料增強 - 提升泛化能力"""
        if not self.training:
            return image
        
        # 更強的隨機旋轉
        if random.random() > 0.3:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, fillcolor=255)
        
        # 隨機縮放
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            width, height = image.size
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            # 裁剪或填充回原始大小
            if scale > 1:
                left = (new_width - width) // 2
                top = (new_height - height) // 2
                image = image.crop((left, top, left + width, top + height))
            else:
                new_img = Image.new('L', (width, height), 255)
                left = (width - new_width) // 2
                top = (height - new_height) // 2
                new_img.paste(image, (left, top))
                image = new_img
        
        # 隨機模糊
        if random.random() > 0.6:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        # 隨機噪點（更強）
        if random.random() > 0.4:
            img_array = np.array(image)
            noise = np.random.randint(-30, 30, img_array.shape, dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        
        # 隨機亮度和對比度調整
        if random.random() > 0.3:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.7, 1.3))
        
        if random.random() > 0.3:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.7, 1.3))
        
        # 隨機網格噪聲（模擬驗證碼干擾線）
        if random.random() > 0.5:
            img_array = np.array(image)
            h, w = img_array.shape
            # 添加隨機線條
            for _ in range(random.randint(1, 3)):
                x1, y1 = random.randint(0, w), random.randint(0, h)
                x2, y2 = random.randint(0, w), random.randint(0, h)
                cv2.line(img_array, (x1, y1), (x2, y2), 
                        random.randint(100, 200), 1)
            image = Image.fromarray(img_array)
        
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


class LightweightCRNN(nn.Module):
    """輕量化 CRNN 模型 - 快速且準確"""
    
    def __init__(self, num_classes=NUM_CLASSES, rnn_hidden=128, rnn_layers=2):
        super(LightweightCRNN, self).__init__()
        
        # 輕量化 CNN 特徵提取器
        self.cnn = nn.Sequential(
            # 第一層
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x16x64
            
            # 第二層
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x8x32
            
            # 第三層
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 128x4x32
            
            # 第四層
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 256x2x32
        )
        
        # 雙向 LSTM（減少隱藏層大小以加速）
        self.rnn = nn.LSTM(
            256 * 2,  # 高度為2，所以是256*2
            rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if rnn_layers > 1 else 0
        )
        
        # 輸出層
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)
        
        # Dropout 層（增加 dropout 率以減少過擬合）
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # CNN 特徵提取
        conv = self.cnn(x)  # [B, 256, 2, W']
        
        # 獲取維度
        b, c, h, w = conv.size()
        
        # 轉換為序列格式 [B, W, C*H]
        conv = conv.permute(0, 3, 1, 2)  # [B, W, C, H]
        conv = conv.reshape(b, w, c * h)  # [B, W, C*H]
        
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


def train_model(data_dir, labels_file, epochs=50, batch_size=32, lr=5e-4, device='cuda'):
    """訓練模型主函數 - 改進版"""
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
    
    # 分割訓練集和驗證集 (85/15) - 給訓練集更多資料
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    
    # 使用固定的隨機種子以確保可重現性
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 創建獨立的驗證集物件（不使用資料增強）
    val_dataset_clean = CaptchaDataset(data_dir, labels_file, training=False)
    val_indices = val_dataset.indices
    val_dataset_clean.samples = [val_dataset_clean.samples[i] for i in val_indices]
    
    print(f"訓練集: {len(train_dataset)} 樣本")
    print(f"驗證集: {len(val_dataset_clean)} 樣本")
    
    # 建立 DataLoader（減小 batch size 以增加更新頻率）
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset_clean, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 建立輕量化模型
    model = LightweightCRNN().to(device)
    
    # 計算模型參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數量: {total_params:,}")
    print(f"可訓練參數量: {trainable_params:,}")
    
    # 損失函數和優化器
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # 使用 Adam 優化器 + L2 正則化
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 學習率調度器 - ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
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
    patience = 20
    patience_counter = 0
    
    print("\n開始訓練...")
    print("=" * 60)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"學習率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 訓練
        train_loss, train_seq_acc, train_char_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 驗證
        val_loss, val_seq_acc, val_char_acc = evaluate(
            model, val_loader, criterion, device
        )
        
        # 更新學習率
        scheduler.step(val_seq_acc)
        
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
            }, 'best_lightweight_crnn_model.pth')
            
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
    plt.savefig('training_history_v3.png', dpi=100)
    print("訓練曲線已儲存: training_history_v3.png")


def test_inference_speed(model, device='cpu', num_tests=100):
    """測試推理速度"""
    import time
    
    model.eval()
    model = model.to(device)
    
    # 創建測試輸入
    dummy_input = torch.randn(1, 1, 32, 128).to(device)
    
    # 熱身
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # 測試速度
    times = []
    for _ in range(num_tests):
        start = time.time()
        with torch.no_grad():
            output = model(dummy_input)
            decoded = decode_predictions(output)
        end = time.time()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n推理速度測試 ({device}):")
    print(f"  平均時間: {avg_time*1000:.2f} ms")
    print(f"  標準差: {std_time*1000:.2f} ms")
    print(f"  最快: {min(times)*1000:.2f} ms")
    print(f"  最慢: {max(times)*1000:.2f} ms")
    
    return avg_time


def main():
    """主程式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='訓練輕量化 CRNN OCR 模型')
    parser.add_argument('--data_dir', type=str, required=True, help='圖片目錄')
    parser.add_argument('--labels', type=str, required=True, help='標註檔案')
    parser.add_argument('--epochs', type=int, default=50, help='訓練 epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='學習率')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='訓練設備')
    parser.add_argument('--test_speed', action='store_true', help='測試推理速度')
    
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
    
    if model and args.test_speed:
        # 測試推理速度
        test_inference_speed(model, device='cpu', num_tests=100)
        if torch.cuda.is_available():
            test_inference_speed(model, device='cuda', num_tests=100)


if __name__ == "__main__":
    main()