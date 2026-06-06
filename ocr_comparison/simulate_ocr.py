import json
import glob
import os
import random
import string

def introduce_noise(text, noise_level=0.15):
    """
    Giả lập lỗi OCR bằng cách thêm nhiễu vào text.
    noise_level: Xác suất một từ bị thay đổi.
    """
    if not text or random.random() > noise_level:
        return text

    # Chỉ xáo trộn nhẹ các từ dài hơn 2 ký tự
    if len(text) <= 2:
        return text
    
    chars = list(text)
    num_errors = max(1, int(len(text) * 0.1)) # Tạo khoảng 10% lỗi trên số ký tự của chuỗi
    
    for _ in range(num_errors):
        idx = random.randint(0, len(chars) - 1)
        action = random.choice(["replace", "drop", "insert"])
        
        if action == "replace":
            # Thay thế bằng một ký tự ngẫu nhiên gần giống (giả lập lỗi OCR)
            chars[idx] = random.choice(string.ascii_letters + "0123456789")
        elif action == "drop":
            chars.pop(idx)
        elif action == "insert":
            chars.insert(idx, random.choice(string.ascii_letters))
            
    return "".join(chars)

def process_directory(directory):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for box in data:
            original_text = box.get('text', '')
            
            # Tạo paddleocr_text bằng cách thêm nhiễu
            simulated_ocr = introduce_noise(original_text)
            
            # Cập nhật vào dictionary
            box['paddleocr_text'] = simulated_ocr
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

if __name__ == "__main__":
    train_dir = "/home/lab/son/lap/vaipe_pima_format/prescriptions/train"
    test_dir = "/home/lab/son/lap/vaipe_pima_format/prescriptions/test"
    
    print("Bắt đầu giả lập OCR text cho tập Train...")
    process_directory(train_dir)
    print(f"Hoàn tất tập Train.")
    
    print("Bắt đầu giả lập OCR text cho tập Test...")
    process_directory(test_dir)
    print(f"Hoàn tất tập Test.")
    
    print("Đã giả lập OCR xong! Giờ bạn có thể train bằng lệnh train_with_paddleocr.")
