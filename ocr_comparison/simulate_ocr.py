import json
import glob
import os
import random
import string

def introduce_noise(text, noise_level=0.15):
    """
    Simulate OCR error by adding noise to text.
    noise_level: Probability of a word being modified.
    """
    if not text or random.random() > noise_level:
        return text

    # Only slightly perturb words longer than 2 characters
    if len(text) <= 2:
        return text
    
    chars = list(text)
    num_errors = max(1, int(len(text) * 0.1)) # Generate about 10% errors based on string length
    
    for _ in range(num_errors):
        idx = random.randint(0, len(chars) - 1)
        action = random.choice(["replace", "drop", "insert"])
        
        if action == "replace":
            # Replace with a similar random character (simulate OCR error)
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
            
            # Create paddleocr_text by adding noise
            simulated_ocr = introduce_noise(original_text)
            
            # Update to dictionary
            box['paddleocr_text'] = simulated_ocr
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

if __name__ == "__main__":
    train_dir = "/home/lab/son/lap/vaipe_pima_format/prescriptions/train"
    test_dir = "/home/lab/son/lap/vaipe_pima_format/prescriptions/test"
    
    print("Start simulating OCR text for Train set...")
    process_directory(train_dir)
    print(f"Finished Train set.")
    
    print("Start simulating OCR text for Test set...")
    process_directory(test_dir)
    print(f"Finished Test set.")
    
    print("Finished simulating OCR! You can now train the model.")
