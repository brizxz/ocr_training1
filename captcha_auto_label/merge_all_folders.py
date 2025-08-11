import os
import json
import shutil
from datetime import datetime
import hashlib

def merge_captcha_folders():
    base_dir = r"C:\Users\ian.su\Desktop\ticket1\ocr_training\captcha_auto_label"
    
    folders = [
        "20250810_014535",
        "20250810_030221", 
        "20250810_170612",
        "20250811_113936",
        "20250811_132206"
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_folder = os.path.join(base_dir, f"merged_{timestamp}")
    os.makedirs(merged_folder, exist_ok=True)
    
    print(f"Creating merged folder: {merged_folder}")
    
    merged_labels = {}
    merged_stats = {
        "total_attempts": 0,
        "success": 0,
        "failed": 0,
        "manual_required": 0
    }
    merged_training_data = []
    
    file_counter = 0
    filename_mapping = {}
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder} not found, skipping...")
            continue
            
        print(f"\nProcessing folder: {folder}")
        
        labels_path = os.path.join(folder_path, "labels.json")
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                folder_labels = json.load(f)
                
            for md5_hash, label_data in folder_labels.items():
                old_filename = label_data['filename']
                old_filepath = os.path.join(folder_path, old_filename)
                
                if os.path.exists(old_filepath):
                    new_filename = f"captcha_{file_counter:05d}.png"
                    new_filepath = os.path.join(merged_folder, new_filename)
                    
                    shutil.copy2(old_filepath, new_filepath)
                    
                    filename_mapping[f"{folder}/{old_filename}"] = new_filename
                    
                    new_label_data = label_data.copy()
                    new_label_data['filename'] = new_filename
                    new_label_data['original_folder'] = folder
                    new_label_data['original_filename'] = old_filename
                    
                    merged_labels[md5_hash] = new_label_data
                    
                    label = label_data.get('label', '')
                    if label and label != 'null':
                        merged_training_data.append(f"{new_filename},{label}")
                    else:
                        merged_training_data.append(f"{new_filename},")
                    
                    file_counter += 1
                    
                    if file_counter % 100 == 0:
                        print(f"  Processed {file_counter} files...")
        
        stats_path = os.path.join(folder_path, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                folder_stats = json.load(f)
                merged_stats['total_attempts'] += folder_stats.get('total_attempts', 0)
                merged_stats['success'] += folder_stats.get('success', 0)
                merged_stats['failed'] += folder_stats.get('failed', 0)
                merged_stats['manual_required'] += folder_stats.get('manual_required', 0)
    
    print(f"\nTotal files processed: {file_counter}")
    
    labels_output = os.path.join(merged_folder, "labels.json")
    with open(labels_output, 'w', encoding='utf-8') as f:
        json.dump(merged_labels, f, indent=2, ensure_ascii=False)
    print(f"Saved merged labels to: {labels_output}")
    
    stats_output = os.path.join(merged_folder, "stats.json")
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(merged_stats, f, indent=2, ensure_ascii=False)
    print(f"Saved merged stats to: {stats_output}")
    
    training_output = os.path.join(merged_folder, "training_data.txt")
    with open(training_output, 'w', encoding='utf-8') as f:
        for line in merged_training_data:
            f.write(line + '\n')
    print(f"Saved merged training data to: {training_output}")
    
    mapping_output = os.path.join(merged_folder, "filename_mapping.json")
    with open(mapping_output, 'w', encoding='utf-8') as f:
        json.dump(filename_mapping, f, indent=2, ensure_ascii=False)
    print(f"Saved filename mapping to: {mapping_output}")
    
    print(f"\nMerge complete!")
    print(f"Merged folder: {merged_folder}")
    print(f"Total images: {file_counter}")
    print(f"Stats: {merged_stats}")
    
    return merged_folder

if __name__ == "__main__":
    merge_captcha_folders()