import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import numpy as np

# Note: For PP-OCRv3, usually it is better to extract bounding boxes and texts offline
# or as a pre-processing step to avoid the overhead during the PyTorch training loop.
# Here we define a wrapper that can optionally call PaddleOCR if texts are not provided.
try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False

class PrescriptionRecognizer(nn.Module):
    def __init__(self, embed_dim=256, use_ocr=False):
        super(PrescriptionRecognizer, self).__init__()
        
        # 1. PP-OCRv3 Initialization (Optional, for end-to-end inference)
        self.use_ocr = use_ocr
        if self.use_ocr and HAS_PADDLEOCR:
            # Initialize PP-OCRv3
            self.ocr = PaddleOCR(use_angle_cls=True, lang='vi', version='PP-OCRv3', show_log=False)
            
        # 2. Text Embedding with MiniLM-v6
        # Upgrade from MiniLM-L12-v2 to MiniLM-v6 (all-MiniLM-L6-v2)
        # Freeze the transformer to use as a feature extractor, or fine-tune if needed.
        self.text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.text_encoder_out_dim = 384 # Dimension for all-MiniLM-L6-v2
        
        # Projection layer to match visual embedding dimension (e.g., 256)
        self.projection = nn.Sequential(
            nn.Linear(self.text_encoder_out_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        
    def extract_ocr(self, image_path_or_numpy):
        """
        Runs PP-OCRv3 on the input image to extract text and bounding boxes.
        Returns: texts (list of str), boxes (list of [x1, y1, x2, y2])
        """
        if not self.use_ocr or not HAS_PADDLEOCR:
            raise RuntimeError("PaddleOCR is not initialized or not installed.")
            
        result = self.ocr.ocr(image_path_or_numpy, cls=True)
        texts = []
        boxes = []
        # result is a list of lists. For single image it's result[0]
        if result is not None and len(result) > 0 and result[0] is not None:
            for line in result[0]:
                box = line[0] # [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                text = line[1][0]
                # Convert poly to min/max box
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
                boxes.append([x1, y1, x2, y2])
                texts.append(text)
                
        return texts, boxes

    def forward(self, texts):
        """
        texts: list of strings (e.g. ['Paracetamol 500mg', 'Vitamin C', ...])
        Returns: text embeddings of shape [len(texts), embed_dim]
        """
        if not texts:
            return torch.empty(0, self.projection[0].out_features)
            
        # Extract embeddings using MiniLM-v6
        # By default sentence-transformers encodes to numpy array or torch tensor on CPU/GPU
        device = next(self.projection.parameters()).device
        
        with torch.no_grad():
            # encode returns tensor if convert_to_tensor=True
            embeddings = self.text_encoder.encode(texts, convert_to_tensor=True, device=device)
            
        embeddings = embeddings.clone()
        # Project to target dimension
        projected_embeddings = self.projection(embeddings)
        return projected_embeddings

if __name__ == "__main__":
    recognizer = PrescriptionRecognizer(embed_dim=256)
    sample_texts = ["Aspirin 100mg", "Paracetamol 500mg", "Amoxicillin"]
    embeddings = recognizer(sample_texts)
    print("Text Embeddings shape:", embeddings.shape)
