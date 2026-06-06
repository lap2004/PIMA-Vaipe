import os
import json
import glob
import random
from PIL import Image, ImageFile
from tqdm import tqdm
import shutil

# Cho phep load anh bi thieu/khuyet byte
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- CONFIG PATHS ---
SRC_PRES_LABEL_DIR = r"D:\HocTap\3\archive\public_train\prescription\label"
SRC_PILL_IMAGE_DIR = r"D:\HocTap\3\archive\public_train\pill\image"
SRC_PILL_LABEL_DIR = r"D:\HocTap\3\archive\public_train\pill\label"

DEST_DATA_DIR = "./data"
DEST_PRES_TRAIN = os.path.join(DEST_DATA_DIR, "prescriptions", "train")
DEST_PRES_TEST = os.path.join(DEST_DATA_DIR, "prescriptions", "test")
DEST_PILLS_TRAIN = os.path.join(DEST_DATA_DIR, "pills", "train")
DEST_PILLS_TEST = os.path.join(DEST_DATA_DIR, "pills", "test")
DEST_ALL_IMGS_TRAIN = os.path.join(DEST_DATA_DIR, "all_imgs", "train")

def create_dirs():
    os.makedirs(DEST_PRES_TRAIN, exist_ok=True)
    os.makedirs(DEST_PRES_TEST, exist_ok=True)
    os.makedirs(DEST_PILLS_TRAIN, exist_ok=True)
    os.makedirs(DEST_PILLS_TEST, exist_ok=True)
    os.makedirs(DEST_ALL_IMGS_TRAIN, exist_ok=True)

def process_item(pres_file, mode):
    filename = os.path.basename(pres_file)
    pres_id = filename.split(".")[0].split("_")[-1]
    
    dest_pres_path = os.path.join(DEST_PRES_TRAIN if mode == "train" else DEST_PRES_TEST, filename)
    shutil.copy(pres_file, dest_pres_path)
    
    pill_label_pattern = os.path.join(SRC_PILL_LABEL_DIR, f"VAIPE_P_{pres_id}_*.json")
    pill_label_files = glob.glob(pill_label_pattern)
    
    dest_pill_folder = os.path.join(
        DEST_PILLS_TRAIN if mode == "train" else DEST_PILLS_TEST,
        filename.split(".")[0]
    )
    
    for label_file in pill_label_files:
        image_filename = os.path.basename(label_file).replace(".json", ".jpg")
        image_path = os.path.join(SRC_PILL_IMAGE_DIR, image_filename)
        
        if not os.path.exists(image_path):
            continue
            
        try:
            with open(label_file, "r") as f:
                boxes = json.load(f)
                
            img = Image.open(image_path)
            pill_img_idx = os.path.basename(label_file).split(".")[0].split("_")[-1]
            
            for box_idx, box in enumerate(boxes):
                x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                label = str(box["label"])
                
                if w <= 0 or h <= 0:
                    continue
                    
                cropped_img = img.crop((x, y, x + w, y + h))
                
                dest_label_folder = os.path.join(dest_pill_folder, label)
                os.makedirs(dest_label_folder, exist_ok=True)
                cropped_img.save(os.path.join(dest_label_folder, f"{pres_id}_{pill_img_idx}_{box_idx}.jpg"))
                
                if mode == "train":
                    dest_all_imgs_folder = os.path.join(DEST_ALL_IMGS_TRAIN, label)
                    os.makedirs(dest_all_imgs_folder, exist_ok=True)
                    cropped_img.save(os.path.join(dest_all_imgs_folder, f"{pres_id}_{pill_img_idx}_{box_idx}.jpg"))
                    
        except Exception as e:
            # Dung tieng Anh khong dau de tranh crash terminal encoding
            print(f"[Error] Failed to process {label_file}: {e}")

def main():
    print(">>> Initializing target directories...")
    create_dirs()
    
    print(">>> Scanning prescription JSON files...")
    pres_files = glob.glob(os.path.join(SRC_PRES_LABEL_DIR, "*.json"))
    print(f"Found {len(pres_files)} prescriptions.")
    
    if len(pres_files) == 0:
        print("[Error] No data found! Please double check D drive paths.")
        return
        
    random.seed(42)
    random.shuffle(pres_files)
    
    split_idx = int(len(pres_files) * 0.9)
    train_files = pres_files[:split_idx]
    test_files = pres_files[split_idx:]
    
    print(f"Dataset split: {len(train_files)} TRAIN, {len(test_files)} VAL/TEST.")
    print(">>> Starting pill image cropping and directory structure generation...")
    
    for pres_file in tqdm(train_files, desc="Processing Train"):
        process_item(pres_file, "train")
        
    for pres_file in tqdm(test_files, desc="Processing Test"):
        process_item(pres_file, "test")
        
    print("\n" + "="*50)
    print(" DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print(" You can now run: python train.py")
    print("="*50)

if __name__ == "__main__":
    main()
