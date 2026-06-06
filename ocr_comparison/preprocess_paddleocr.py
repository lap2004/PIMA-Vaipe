import os
import json
import cv2
import glob
from tqdm import tqdm
from paddleocr import PaddleOCR
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/lab/son/lap/vaipe_pima_format/')
    parser.add_argument('--vaipe2022-dir', type=str, default='/home/lab/son/lap/vaipepill2022/')
    args = parser.parse_args()

    ocr = PaddleOCR(use_angle_cls=False, det=False, use_gpu=False, lang='vi')

    # Process both train and test sets
    for split in ['train', 'test']:
        label_dir = os.path.join(args.data_dir, f'prescriptions/{split}')
        
        if split == 'train':
            image_dir = os.path.join(args.vaipe2022_dir, 'public_train/prescription/image')
        else:
            image_dir = os.path.join(args.vaipe2022_dir, 'public_test/prescription/image')

        json_files = glob.glob(os.path.join(label_dir, '*.json'))
        print(f"Extracting OCR for {len(json_files)} prescriptions in {split} set...")

        for json_file in tqdm(json_files, desc=f"Processing OCR {split}"):
            base_name = os.path.basename(json_file).replace('.json', '')
            # Original image has .png (or .jpg) extension
            img_path = os.path.join(image_dir, f"{base_name}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(image_dir, f"{base_name}.jpg")
                if not os.path.exists(img_path):
                    continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            modified = False
            for box_data in data:
                # PIMA format is: "box": [x_min, y_min, x_max, y_max]
                if 'box' not in box_data:
                    box_data['paddleocr_text'] = "unknown"
                    modified = True
                    continue
                
                coords = box_data['box']
                if len(coords) != 4:
                    box_data['paddleocr_text'] = "unknown"
                    modified = True
                    continue

                xmin, ymin, xmax, ymax = coords
                xmin = max(0, int(xmin))
                ymin = max(0, int(ymin))
                xmax = min(img.shape[1], int(xmax))
                ymax = min(img.shape[0], int(ymax))

                if xmax > xmin and ymax > ymin:
                    crop = img[ymin:ymax, xmin:xmax]
                    try:
                        result = ocr.ocr(crop, det=False, cls=False)
                        if result and len(result) > 0 and len(result[0]) > 0:
                            text = result[0][0][0]
                            box_data['paddleocr_text'] = text
                        else:
                            box_data['paddleocr_text'] = "unknown"
                    except Exception as e:
                        box_data['paddleocr_text'] = "unknown"
                else:
                    box_data['paddleocr_text'] = "unknown"
                
                modified = True

            if modified:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
