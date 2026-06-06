import os
import json
import shutil
import random

def main():
    src_dir = "/home/lab/son/lap/vaipepill2022/public_train"
    dst_dir = "/home/lab/son/lap/vaipe_pima_format"
    
    # We will recreate the destination directory to ensure clean state
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    
    # Create required directories
    all_imgs_dir = os.path.join(dst_dir, "all_imgs", "train")
    
    pills_train_dir = os.path.join(dst_dir, "pills", "train")
    pills_test_dir = os.path.join(dst_dir, "pills", "test")
    
    pres_train_dir = os.path.join(dst_dir, "prescriptions", "train")
    pres_test_dir = os.path.join(dst_dir, "prescriptions", "test")
    
    os.makedirs(all_imgs_dir, exist_ok=True)
    os.makedirs(pills_train_dir, exist_ok=True)
    os.makedirs(pills_test_dir, exist_ok=True)
    os.makedirs(pres_train_dir, exist_ok=True)
    os.makedirs(pres_test_dir, exist_ok=True)
    
    # Read mapping
    map_file = os.path.join(src_dir, "pill_pres_map.json")
    with open(map_file, "r") as f:
        mappings = json.load(f)
        
    # Split 80/20
    random.seed(42)
    random.shuffle(mappings)
    split_idx = int(len(mappings) * 0.8)
    train_mappings = mappings[:split_idx]
    test_mappings = mappings[split_idx:]
    
    print(f"Total mappings: {len(mappings)}. Train: {len(train_mappings)}, Test: {len(test_mappings)}")
    
    def process_split(split_mappings, pills_out_dir, pres_out_dir):
        for item in split_mappings:
            pres_file = item["pres"]
            pres_id = pres_file.split(".")[0]
            
            # Copy prescription label
            src_pres_json = os.path.join(src_dir, "prescription", "label", pres_file)
            dst_pres_json = os.path.join(pres_out_dir, pres_file)
            if os.path.exists(src_pres_json):
                shutil.copy(src_pres_json, dst_pres_json)
            
            pill_files = item["pill"]
            pres_pills_dir = os.path.join(pills_out_dir, pres_id)
            
            for pill_json_file in pill_files:
                pill_id = pill_json_file.split(".")[0]
                
                src_pill_json = os.path.join(src_dir, "pill", "label", pill_json_file)
                if not os.path.exists(src_pill_json):
                    continue
                    
                with open(src_pill_json, "r") as f:
                    try:
                        pill_data = json.load(f)
                        if isinstance(pill_data, list) and len(pill_data) > 0:
                            label = str(pill_data[0].get("label", "unknown"))
                        else:
                            label = "unknown"
                    except:
                        label = "unknown"
                
                if label == "unknown":
                    continue
                    
                # Create class dirs
                class_all_dir = os.path.join(all_imgs_dir, label)
                os.makedirs(class_all_dir, exist_ok=True)
                
                class_pres_dir = os.path.join(pres_pills_dir, label)
                os.makedirs(class_pres_dir, exist_ok=True)
                
                # Copy image
                src_img = os.path.join(src_dir, "pill", "image", f"{pill_id}.jpg")
                if os.path.exists(src_img):
                    shutil.copy(src_img, os.path.join(class_all_dir, f"{pill_id}.jpg"))
                    shutil.copy(src_img, os.path.join(class_pres_dir, f"{pill_id}.jpg"))

    print("Processing train set...")
    process_split(train_mappings, pills_train_dir, pres_train_dir)
    
    print("Processing test set...")
    process_split(test_mappings, pills_test_dir, pres_test_dir)

    print("Data preparation complete.")

if __name__ == "__main__":
    main()
