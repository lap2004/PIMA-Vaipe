# PIMA_NEW: Comprehensive Upgrade (Transformer-based Multimodal Architecture)

The `pima_ViT` (or `PIMA_NEW`) directory is a **breakthrough upgrade** of the PIMA-VAIPE system. This version replaces almost all old modules with State-Of-The-Art (SOTA) Deep Learning technologies, shifting from a simple graph comparison model to a **Transformer-based Multimodal Cross-Attention Framework**.

This is the most deeply optimized version for the VAIPE dataset.

---

This version uses a **Vision Transformer (ViT-B/16)** for the visual processing stream:

### Vision Transformer (ViT-B/16) - *The "Clean Crop" Approach*
- **Operational Principle:** Assumes the pill has been cleanly separated from its background (using a tool like `rembg`). The clean pill image is passed through the `ViT-B/16` model.
- **Pros:** Since ViT is not affected by background noise, it can focus entirely on capturing morphological features, yielding exceptionally high accuracy.
- **Cons:** Requires a separate image preprocessing step. Cannot automatically detect pills on raw prescription photographs.
- **Accuracy:** `82.46%`

---

## 🧠 Core Innovations 

Other modules built around ViT include:

### 1. Language Branch & OCR: MiniLM + PP-OCRv3
- **Replacing BERT with `all-MiniLM-L6-v2`:** Provides sentence embedding vectors much faster and more accurately than the previous version.
- **End-to-End OCR Integration:** Supported by `PaddleOCR` (PP-OCRv3), allowing the network to automatically extract text and Bounding Boxes directly from prescription photographs.

### 2. Graph Branch: Relational Graph Attention (R-GAT)
- **Replacing `GraphSAGE` with `RGATConv`:** 
- Unlike standard GCNs, R-GAT can understand *directional spatial relations* (left, right, top, bottom) between text boxes.
- **Pseudo-Classifier:** The graph branch includes a binary classifier predicting the probability that a text box contains a drug name, acting as an attention gate.

### 3. Fusion: Multi-Modal Cross-Attention
- **Replacing basic Cosine Similarity with a powerful `MultiModalCrossAttention` module.**
- Visual features act as **Queries**, while textual features from the GNN act as **Keys/Values**. This allows the network to dynamically "search" for the most matching text on the prescription based on the pill's appearance.

### 4. Objective Function: InfoNCE Loss
- **Replacing standard Contrastive Loss with `InfoNCE Loss`.**
- Optimizes Cross-entropy over the entire Batch (with a temperature coefficient), pulling the correct multimodal pairs closer while simultaneously pushing all incorrect pairs further apart.

---

## ⚙️ Source Code Structure

- `train.py`: Supports multi-GPU training with `torchrun` (Distributed Data Parallel - DDP). 
- `eval.py`: Evaluation script, measuring Top-1 Matching accuracy on the dataset.
- `preprocess_pills.py`, `data_split.py`: Scripts supporting preparation and splitting of the training dataset.

---

## 📈 Evaluation Results & Experiments (Details)

The evaluation process is conducted on the test set to measure the ability to match real pill images with drug names.

| Model | Method | Total Evaluations | Top-1 Matches | Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **Vision Transformer (ViT-B/16)** | *Clean Crop* | 3,627 | 2,991 | **82.46%** |

### Vision Transformer (ViT-B/16) Model
- **Total evaluated pills (with matching labels):** 3,627
- **Exact Matches (Top-1):** 2,991
- **Average Loss (Early Stopping at Epoch 45):** 0.2075
> **Remarks:** Eliminating background noise helps the Transformer network focus completely on capturing the morphological features of the pills, resulting in very high accuracy.
