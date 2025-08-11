#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自動標註驗證碼系統
透過實際提交結果來自動標註訓練資料
"""

import os
import json
import time
import base64
import hashlib
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import undetected_chromedriver as uc
import ddddocr

class AutoCaptchaLabeler:
    def __init__(self, save_dir="captcha_auto_label"):
        """
        初始化自動標註系統
        
        Args:
            save_dir: 儲存目錄
        """
        self.save_dir = save_dir
        self.session_dir = os.path.join(save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.session_dir, exist_ok=True)
        
        # 標註資料庫
        self.label_db_file = os.path.join(self.session_dir, "labels.json")
        self.label_db = self.load_label_db()
        
        # 統計資訊
        self.stats = {
            "total_attempts": 0,
            "success": 0,
            "failed": 0,
            "manual_required": 0
        }
        
        # 圖片計數器
        self.image_counter = 0
        
        # OCR 模型 - 延遲初始化
        self.ocr = None
        
    def load_label_db(self):
        """載入標註資料庫"""
        if os.path.exists(self.label_db_file):
            with open(self.label_db_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_label_db(self):
        """儲存標註資料庫"""
        with open(self.label_db_file, 'w', encoding='utf-8') as f:
            json.dump(self.label_db, f, indent=2, ensure_ascii=False)
    
    def get_image_hash(self, image_data):
        """計算圖片雜湊值作為唯一標識"""
        return hashlib.md5(image_data).hexdigest()
    
    def collect_and_label(self, driver, url, max_samples=5000):
        """
        收集並自動標註驗證碼
        
        Args:
            driver: Selenium WebDriver
            url: 目標頁面
            max_samples: 最大樣本數
        """
        # 延遲初始化 OCR（在瀏覽器操作之後）
        if self.ocr is None:
            print("初始化 OCR 模型...")
            self.ocr = ddddocr.DdddOcr(show_ad=False, use_gpu=False)
            print("OCR 模型初始化完成")
        
        try:
            # 確保瀏覽器視窗存在
            current = driver.current_url
            # 如果不在正確頁面，重新載入
            if "ticket/ticket" not in current:
                driver.get(url)
                time.sleep(3)
        except Exception as e:
            print(f"錯誤：瀏覽器視窗問題 - {e}")
            return
        
        print(f"開始自動收集和標註驗證碼")
        print(f"儲存目錄: {self.session_dir}")
        print("-" * 50)
        
        while self.stats["total_attempts"] < max_samples:
            try:
                # 檢查瀏覽器視窗是否還存在
                try:
                    driver.current_url
                except:
                    print("瀏覽器視窗已關閉，結束程式")
                    break
                # 步驟1: 擷取驗證碼圖片
                captcha_data = self.capture_captcha(driver)
                if not captcha_data:
                    time.sleep(1)
                    continue
                
                image_hash = captcha_data['hash']
                image_path = captcha_data['path']
                
                # 步驟2: OCR 初步識別
                initial_guess = self.ocr.classification(captcha_data['data'])
                print(f"\n[{self.stats['total_attempts']+1}] OCR 識別: {initial_guess}")
                
                # 步驟3: 準備表單（設定票數、勾選同意）
                self.prepare_form(driver)
                
                # 步驟4: 嘗試提交驗證碼
                submit_result = self.submit_captcha(driver, initial_guess)
                
                # 步驟5: 根據結果標註
                if submit_result['success']:
                    # 成功 - 自動標註為正確
                    self.label_db[image_hash] = {
                        "filename": os.path.basename(image_path),
                        "label": initial_guess,
                        "confidence": "high",
                        "method": "auto_success",
                        "timestamp": datetime.now().isoformat()
                    }
                    print(f"✓ 成功！標註為: {initial_guess}")
                    self.stats["success"] += 1
                    
                    # 如果跳轉到登入頁面，直接返回票券頁面（不需要登入）
                    if submit_result.get('redirect_to_login'):
                        print("  返回票券頁面（不登入以避免偵測）")
                        driver.get(url)
                        time.sleep(1)
                    else:
                        # 正常跳轉後返回
                        time.sleep(1)
                        driver.get(url)
                        time.sleep(1)
                    
                elif submit_result['retry_count'] >= 1:
                    # 失敗1次 - 需要人工標註
                    self.label_db[image_hash] = {
                        "filename": os.path.basename(image_path),
                        "label": None,
                        "failed_attempts": [initial_guess] + submit_result.get('other_attempts', []),
                        "confidence": "manual_required",
                        "method": "multiple_failures",
                        "timestamp": datetime.now().isoformat()
                    }
                    print(f"✗ 失敗1次，需要人工標註")
                    print(f"   圖片已儲存: {os.path.basename(image_path)}")
                    print(f"   OCR 識別為: {initial_guess}")
                    self.stats["manual_required"] += 1
                    
                    # 重新載入驗證碼繼續下一張
                    self.reload_captcha(driver)
                    time.sleep(0.5)
                    
                else:
                    # 單次失敗 - 記錄但繼續嘗試
                    self.stats["failed"] += 1
                    print(f"✗ 失敗，重新載入驗證碼")
                    
                    # 重新載入驗證碼
                    self.reload_captcha(driver)
                    time.sleep(0.5)
                    continue
                
                self.stats["total_attempts"] += 1
                
                # 定期儲存
                if self.stats["total_attempts"] % 50 == 0:
                    self.save_label_db()
                    self.save_stats()
                    
            except Exception as e:
                print(f"錯誤: {e}")
                time.sleep(1)
        
        # 最終儲存
        self.save_label_db()
        self.save_stats()
        self.print_summary()
    
    def capture_captcha(self, driver):
        """擷取驗證碼圖片"""
        try:
            img_base64 = driver.execute_script("""
                var img = document.getElementById('TicketForm_verifyCode-image');
                if(img && img.complete && img.naturalHeight !== 0) {
                    var canvas = document.createElement('canvas');
                    var context = canvas.getContext('2d');
                    canvas.height = img.naturalHeight;
                    canvas.width = img.naturalWidth;
                    context.drawImage(img, 0, 0);
                    return canvas.toDataURL();
                }
                return null;
            """)
            
            if img_base64:
                img_data = base64.b64decode(img_base64.split(',')[1])
                img_hash = self.get_image_hash(img_data)
                
                # 檢查是否已經標註過
                if img_hash in self.label_db and self.label_db[img_hash].get('label'):
                    print(f"已標註過: {self.label_db[img_hash]['label']}")
                    self.reload_captcha(driver)
                    return None
                
                # 使用序號命名圖片
                filename = f"captcha_{self.image_counter:05d}.png"
                filepath = os.path.join(self.session_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(img_data)
                
                # 增加計數器
                self.image_counter += 1
                
                return {
                    "data": img_data,
                    "hash": img_hash,
                    "path": filepath
                }
        except Exception as e:
            print(f"擷取驗證碼失敗: {e}")
        return None
    
    def prepare_form(self, driver):
        """準備表單（設定票數、勾選同意）"""
        try:
            driver.execute_script("""
                // 設定票數
                var selects = document.querySelectorAll('select[id^="TicketForm_ticketPrice_"]');
                if (selects.length > 0) {
                    selects[0].value = '1';
                    selects[0].dispatchEvent(new Event('change'));
                }
                
                // 勾選同意
                var agree = document.getElementById('TicketForm_agree');
                if (agree && !agree.checked) {
                    agree.checked = true;
                    agree.dispatchEvent(new Event('change'));
                }
            """)
            time.sleep(0.1)
        except:
            pass
    
    def submit_captcha(self, driver, captcha_text, max_retries=1):
        """
        提交驗證碼並檢查結果
        
        Returns:
            dict: {success: bool, retry_count: int, other_attempts: list}
        """
        retry_count = 0
        other_attempts = []
        
        while retry_count < max_retries:
            try:
                # 輸入驗證碼
                verify_input = driver.find_element(By.ID, 'TicketForm_verifyCode')
                verify_input.clear()
                verify_input.send_keys(captcha_text)
                time.sleep(0.1)
                
                # 點擊提交按鈕
                from selenium.webdriver.common.action_chains import ActionChains
                submit_btn = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"].btn-green')
                actions = ActionChains(driver)
                actions.move_to_element(submit_btn).click().perform()
                
                time.sleep(0.3)  # 等待結果
                
                # 檢查是否成功
                # 方法1: 檢查是否有 alert
                try:
                    alert = driver.switch_to.alert
                    alert_text = alert.text
                    if "驗證碼" in alert_text and "不正確" in alert_text:
                        alert.accept()
                        retry_count += 1
                        
                        # 如果還有重試機會，嘗試其他 OCR 方法
                        if retry_count < max_retries:
                            # 這裡可以使用其他 OCR 或策略
                            new_guess = self.alternative_ocr(driver)
                            if new_guess and new_guess != captcha_text:
                                other_attempts.append(new_guess)
                                captcha_text = new_guess
                                print(f"  重試 {retry_count}: {new_guess}")
                                time.sleep(0.5)
                                continue
                        return {"success": False, "retry_count": retry_count, "other_attempts": other_attempts}
                except:
                    pass
                
                # 方法2: 檢查是否跳轉到登入頁面（表示驗證碼正確但未登入）
                current_url = driver.current_url
                if "login" in current_url.lower() or "/login" in current_url:
                    # 驗證碼正確但需要登入，視為成功
                    print("  ✓ 驗證碼正確！（跳轉至登入頁面）")
                    return {"success": True, "retry_count": retry_count, "redirect_to_login": True}
                
                # 方法3: 檢查 URL 是否改變到會員驗證或訂單頁
                if "verify" in current_url or "order" in current_url:
                    return {"success": True, "retry_count": retry_count}
                
                # 方法4: 檢查頁面元素
                time.sleep(0.2)
                if self.check_success_elements(driver):
                    return {"success": True, "retry_count": retry_count}
                
                retry_count += 1
                
            except Exception as e:
                print(f"提交錯誤: {e}")
                retry_count += 1
        
        return {"success": False, "retry_count": retry_count, "other_attempts": other_attempts}
    
    def alternative_ocr(self, driver):
        """使用替代 OCR 方法"""
        # 這裡可以實現其他 OCR 策略
        # 例如：圖片預處理後重新識別、使用其他 OCR 引擎等
        return None
    
    def check_success_elements(self, driver):
        """檢查是否有成功的頁面元素"""
        try:
            # 檢查是否進入下一步（會員驗證頁面）
            if driver.find_elements(By.CSS_SELECTOR, '.zone-verify'):
                return True
            # 檢查是否進入訂單頁面
            if driver.find_elements(By.CSS_SELECTOR, '.ticket-order'):
                return True
        except:
            pass
        return False
    
    def reload_captcha(self, driver):
        """重新載入驗證碼"""
        try:
            driver.execute_script("""
                var img = document.getElementById('TicketForm_verifyCode-image');
                if(img) img.click();
            """)
        except:
            pass
    
    
    def save_stats(self):
        """儲存統計資訊"""
        stats_file = os.path.join(self.session_dir, "stats.json")
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def print_summary(self):
        """列印總結"""
        print("\n" + "=" * 50)
        print("自動標註完成！")
        print("-" * 50)
        print(f"總嘗試次數: {self.stats['total_attempts']}")
        print(f"成功標註: {self.stats['success']}")
        print(f"失敗次數: {self.stats['failed']}")
        print(f"需要人工標註: {self.stats['manual_required']}")
        print(f"成功率: {self.stats['success']/max(self.stats['total_attempts'],1)*100:.1f}%")
        print("-" * 50)
        print(f"標註檔案: {self.label_db_file}")
        print(f"圖片目錄: {self.session_dir}")
        
        # 列出需要人工標註的檔案
        if self.stats['manual_required'] > 0:
            print("\n需要人工標註的檔案：")
            for img_hash, data in self.label_db.items():
                if data.get('label') is None:
                    print(f"  - {data['filename']} (OCR: {data.get('failed_attempts', ['未知'])[0]})")
            print("\n請手動開啟圖片檔案並在 labels.json 中填入正確答案")
        
    def export_training_data(self):
        """匯出訓練資料格式"""
        training_file = os.path.join(self.session_dir, "training_data.txt")
        with open(training_file, 'w', encoding='utf-8') as f:
            for img_hash, data in self.label_db.items():
                if data.get('label'):
                    # 有答案的條目
                    f.write(f"{data['filename']},{data['label']}\n")
                else:
                    # 失敗的條目，答案留空
                    f.write(f"{data['filename']},\n")
        print(f"訓練資料已匯出: {training_file}")

def main():
    """主程式"""
    TICKET_URL = "https://tixcraft.com/ticket/ticket/25_elijah/19065/1/8"
    
    driver = None
    try:
        # 創建瀏覽器
        options = uc.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        driver = uc.Chrome(options=options, version_main=None)
        
        print("=" * 50)
        print("自動標註驗證碼系統")
        print("=" * 50)
        print("工作原理：")
        print("1. OCR 初步識別驗證碼")
        print("2. 實際提交測試是否正確")
        print("3. 成功則自動標註，失敗則儲存圖片供本地標註")
        print("-" * 50)
        print("重要提示：")
        print("- 不需要登入帳號（避免帳號被偵測異常）")
        print("- 驗證碼正確時會跳轉到登入頁面")
        print("- 失敗的驗證碼會儲存到本地資料夾")
        print("- 請事後手動標註失敗的驗證碼")
        print("-" * 50)
        
        # 先開啟瀏覽器並載入頁面
        print("正在開啟瀏覽器...")
        driver.get(TICKET_URL)
        time.sleep(3)
        
        print("\n請確認頁面已正確載入驗證碼")
        print("按 Enter 開始自動標註...")
        input()
        
        # 先測試瀏覽器是否正常
        try:
            test_url = driver.current_url
            print(f"當前頁面: {test_url}")
        except Exception as e:
            print(f"瀏覽器已關閉或異常: {e}")
            return
        
        # 創建自動標註器（在瀏覽器開啟後）
        labeler = AutoCaptchaLabeler()
        
        # 開始自動標註（不再重新載入頁面）
        labeler.collect_and_label(driver, TICKET_URL, max_samples=5000)
        
        # 匯出訓練資料
        labeler.export_training_data()
        
    except Exception as e:
        print(f"程式錯誤: {e}")
    finally:
        if driver:
            print("\n按 Enter 結束...")
            input()
            driver.quit()

if __name__ == "__main__":
    main()