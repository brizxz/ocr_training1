import json
import os

def rebuild_training_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    labels_file = os.path.join(base_dir, 'labels.json')
    output_file = os.path.join(base_dir, 'training_data.txt')
    
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    training_lines = []
    
    for md5_hash, data in labels_data.items():
        filename = data['filename']
        label = data.get('label', '')
        
        if label and label != 'null':
            training_lines.append(f"{filename},{label}")
        else:
            training_lines.append(f"{filename},")
    
    training_lines.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in training_lines:
            f.write(line + '\n')
    
    print(f"Successfully created training_data.txt with {len(training_lines)} entries")
    return len(training_lines)

if __name__ == "__main__":
    count = rebuild_training_data()