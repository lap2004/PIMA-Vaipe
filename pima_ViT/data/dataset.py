import os
import json
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import cv2

class PimaDataset(Dataset):
    def __init__(self, split_file, data_dir, is_train=True):
        """
        split_file: path to json file containing split data (e.g. train_scenario_1-1.json)
        data_dir: base directory of VAIPE pill dataset
        """
        with open(split_file, 'r') as f:
            self.data = json.load(f)
            
        self.data_dir = data_dir
        self.is_train = is_train
        self.samples = []
        for item in self.data:
            pres = item['pres']
            for pill in item['pill']:
                pres_label_path = os.path.join(data_dir, 'prescription/label', pres)
                pill_label_path = os.path.join(data_dir, 'pill/label', pill)
                pill_img_path = os.path.join(data_dir, 'pill/image', pill.replace('.json', '.jpg'))
                if os.path.exists(pres_label_path) and os.path.exists(pill_label_path) and os.path.exists(pill_img_path):
                    self.samples.append((pres, pill))
                    
        print(f"Loaded {len(self.samples)} samples from {split_file}")
        
        # Advanced Augmentation for Unconstrained Environments
        if self.is_train:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ColorJitter(p=0.4),
                A.GaussNoise(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pres_file, pill_file = self.samples[idx]
        
        pres_label_path = os.path.join(self.data_dir, 'prescription/label', pres_file)
        pill_label_path = os.path.join(self.data_dir, 'pill/label', pill_file)
        pill_img_path = os.path.join(self.data_dir, 'pill/image', pill_file.replace('.json', '.jpg'))
        
        with open(pill_label_path, 'r') as f:
            pill_data = json.load(f)
            
        class_labels = []
        raw_boxes = []
        for i, p in enumerate(pill_data):
            class_labels.append(p.get('label', -1))
            x, y, w, h = p.get('x', 0), p.get('y', 0), p.get('w', 0), p.get('h', 0)
            raw_boxes.append([x, y, x+w, y+h])
            
        cropped_pill_images = []
        for i, p in enumerate(pill_data):
            crop_name = pill_file.replace('.json', f'_box_{i}.png')
            crop_path = os.path.join(self.data_dir, 'pill_crops', crop_name)
            
            if os.path.exists(crop_path):
                img = cv2.imread(crop_path)
                if img is not None:
                    try:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        augmented = self.transform(image=img)
                        cropped_pill_images.append(augmented['image'])
                    except Exception as e:
                        print(f"Error processing image {crop_path}: {e}")
                        
        if len(cropped_pill_images) == 0:
            cropped_pill_images.append(torch.zeros((3, 224, 224), dtype=torch.float32))
            class_labels = [-1]
            
        images = torch.stack(cropped_pill_images) # [Num_Pills, 3, 224, 224]
        pill_boxes = []
            
        # Load prescription
        with open(pres_label_path, 'r') as f:
            pres_data = json.load(f)
            
        texts = []
        true_pill_name_labels = []
        pres_mappings = []
        for t in pres_data:
            text_val = t.get('text', '')
            if not text_val:
                continue
            texts.append(text_val)
            if t.get('label') == 'drugname':
                true_pill_name_labels.append(1.0)
                pres_mappings.append(t.get('mapping', -1))
            else:
                true_pill_name_labels.append(0.0)
                pres_mappings.append(-1)
                
        if len(texts) == 0:
            texts = [""]
            true_pill_name_labels = [0.0]
            pres_mappings = [-1]
            
        true_pill_name_labels = torch.tensor(true_pill_name_labels, dtype=torch.float32)
        
        positive_pairs = []
        for i, p_label in enumerate(class_labels):
            for j, t_map in enumerate(pres_mappings):
                if p_label == t_map and p_label != -1 and t_map != -1:
                    positive_pairs.append((i, j))
                    
        num_texts = len(texts)
        edge_index = []
        edge_type = []
        for i in range(num_texts - 1):
            edge_index.append([i, i+1])
            edge_index.append([i+1, i])
            edge_type.extend([0, 0])
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        if edge_index.numel() == 0:
             edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        
        return {
            'image': images,
            'pill_boxes': pill_boxes,
            'texts': texts,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'true_pill_name_labels': true_pill_name_labels,
            'positive_pairs': positive_pairs
        }

def collate_fn(batch):
    images = [item['image'] for item in batch]
    pill_boxes = [item['pill_boxes'] for item in batch]
    texts = [item['texts'] for item in batch]
    edge_indices = [item['edge_index'] for item in batch]
    edge_types = [item['edge_type'] for item in batch]
    true_pill_name_labels = [item['true_pill_name_labels'] for item in batch]
    positive_pairs = [item['positive_pairs'] for item in batch]
    
    return images, pill_boxes, texts, edge_indices, edge_types, true_pill_name_labels, positive_pairs
