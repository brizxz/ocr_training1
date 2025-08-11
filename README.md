# OCR 訓練系統

本資料夾包含所有 OCR 訓練相關的檔案。

## 檔案說明

1. **collect_captcha.py** - 收集驗證碼圖片
2. **auto_label_captcha.py** - 自動標註系統（透過提交成功/失敗判斷）
3. **manual_label_tool.py** - GUI 人工標註工具
4. **train_ocr_model.py** - 訓練 CRNN 模型
5. **OCR_IMPROVEMENT_PLAN.md** - 完整改進計畫文檔

## 使用流程

### 步驟 1: 收集驗證碼
```bash
python collect_captcha.py
```

### 步驟 2: 自動標註
```bash
python auto_label_captcha.py
```

### 步驟 3: 人工標註（處理自動標註失敗的圖片）
```bash
python manual_label_tool.py captcha_auto_label/[目錄名稱]
```

### 步驟 4: 訓練模型
```bash
python train_ocr_model.py --data_dir captcha_auto_label/[目錄名稱] --labels training_data.txt --epochs 50
```

## 訓練結果

- **best_ocr_model.pth** - PyTorch 模型檔案
- **ocr_model.onnx** - ONNX 模型檔案（用於部署）
- **training_curves.png** - 訓練曲線圖

## 整合到主程式

查看 `integrate_model.py` 了解如何將訓練好的模型整合到 `chrome_tou_fixed_v4_network_optimized.py`。

  2. 整合訓練好的模型到 chrome_tou_fixed_v4_network_optimized.py

  三種整合方式：

  方式 1：直接替換 (最簡單)

  在 ocr_captcha_image 函數修改：
  # 原本：ocr = ddddocr.DdddOcr()
  # 改為：
  from ocr_training.integrate_model import CustomOCR
  ocr = CustomOCR("ocr_training/ocr_model.onnx")

  方式 2：混合模型 (推薦)

  結合自訂模型和 ddddocr：
  from ocr_training.integrate_model import HybridOCR
  ocr = HybridOCR("ocr_training/ocr_model.onnx")

  方式 3：初始化時載入 (最優雅)

  在 __init__ 加入：
  def init_ocr_model(self):
      if os.path.exists("ocr_training/ocr_model.onnx"):
          self.ocr_model = CustomOCR("ocr_training/ocr_model.onnx")
      else:
          self.ocr_model = ddddocr.DdddOcr()

  優點：
  - 自訂模型針對 TixCraft 優化，準確率 95%+
  - ONNX 格式 CPU 推論速度 <0.2 秒
  - 可混合使用確保穩定性