import os
import json
import cv2
import glob
from tqdm import tqdm
from rembg import remove
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_file(label_file, pill_image_dir, output_dir):
    try:
        with open(label_file, 'r') as f:
            pill_data = json.load(f)
            
        base_name = os.path.basename(label_file).replace('.json', '')
        img_path = os.path.join(pill_image_dir, base_name + '.jpg')
        
        if not os.path.exists(img_path):
            return 0
            
        img = cv2.imread(img_path)
        if img is None:
            return 0
            
        h_img, w_img = img.shape[:2]
        processed_count = 0
        
        for i, p in enumerate(pill_data):
            out_path = os.path.join(output_dir, f"{base_name}_box_{i}.png")
            
            # Skip if already processed
            if os.path.exists(out_path):
                processed_count += 1
                continue
                
            x, y, w, h = p.get('x', 0), p.get('y', 0), p.get('w', 0), p.get('h', 0)
            xmin = max(0, int(x))
            ymin = max(0, int(y))
            xmax = min(w_img, int(x + w))
            ymax = min(h_img, int(y + h))
            
            if xmax > xmin and ymax > ymin:
                crop = img[ymin:ymax, xmin:xmax]
                try:
                    crop_nobg = remove(crop)
                    cv2.imwrite(out_path, crop_nobg)
                    processed_count += 1
                except Exception as e:
                    pass
        return processed_count
    except Exception:
        return 0

def preprocess_pills(data_dir):
    pill_label_dir = os.path.join(data_dir, 'pill/label')
    pill_image_dir = os.path.join(data_dir, 'pill/image')
    output_dir = os.path.join(data_dir, 'pill_crops')
    
    os.makedirs(output_dir, exist_ok=True)
    
    label_files = glob.glob(os.path.join(pill_label_dir, '*.json'))
    print(f"Found {len(label_files)} pill label files.")
    
    max_workers = os.cpu_count() or 4
    print(f"Starting multiprocessing with {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, lf, pill_image_dir, output_dir) for lf in label_files]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            pass

if __name__ == "__main__":
    data_dir = '/home/lab/son/lap/vaipepill2022/public_train'
    preprocess_pills(data_dir)
