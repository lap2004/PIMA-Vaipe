# PIMA Faster R-CNN: End-to-End Approach

The `pima_Faster-R-CNN` directory contains the image processing model version based on **Faster R-CNN**. Unlike the Vision Transformer (ViT) approach, which requires cropping the pill out of the background, this version extracts features directly from the raw prescription image (End-to-End).

---

## 🌟 System Architecture

### 1. Vision Branch - Faster R-CNN
- **Core Model:** `FasterRCNN_ResNet50_FPN`.
- **Operational Principle:** Takes the raw image and the bounding box coordinates of the pills as input. The network uses a ResNet50 backbone combined with a Feature Pyramid Network (FPN). Then, the **RoI Align** module is used to extract feature vectors directly for each pill from those bounding boxes.
- **Pros:** Fully End-to-End, easily integrated into real-world systems as it requires no auxiliary background removal tool.
- **Challenges:** The extracted features contain background noise, requiring the model to learn better noise filtering.

### 2. Language, OCR & Graph Branch
- Inherits the best upgrades from the PIMA_NEW version:
  - **OCR:** End-to-End integration with `PaddleOCR` (PP-OCRv3) to automatically extract text and bounding boxes from prescription images, efficiently reading natural text.
  - **Language:** Uses `all-MiniLM-L6-v2` for fast and accurate textual embeddings.
  - **Graph:** Uses Relational Graph Attention (`RGATConv`) to understand the directional spatial logic of the text boxes on the prescription, combined with a Pseudo-Classifier to predict the probability of containing a drug name.

---

## 📊 Training & Evaluation Results

Using the End-to-End method means retaining a larger number of samples (since none are discarded during a cropping step).

- **Total evaluated pill pairs:** `7,058`
- **Top-1 Matches:** Up to `5,635` pairs
- **Accuracy:** Ranges between **79.84%** and **83.47%** (depending on the best epoch).

> **Remarks:** Compared to the old Baseline version, the Faster R-CNN system combined with R-GAT and InfoNCE Loss has significantly improved performance. Even when processing raw images with noise, the model still achieves approximately ~80-83% accuracy, making this the ideal version for real-world deployment.

---

## ⚙️ Source Code Structure

- `train.py`: Supports multi-GPU training with `torchrun` (DDP). Utilizes Gradient Accumulation (`--accumulate-steps`) and Early Stopping.
- `eval.py`: Evaluates the model's accuracy on the validation set.
- `plot_from_json.py`: Helps plot Loss and Accuracy charts from the `train_results.json` (`train_0306.json`) file.
- `preprocess_pills.py`, `data_split.py`: Prepare and split the data (Train/Val).
- `models/`: Contains the definitions for the entire network architecture (Faster R-CNN, R-GAT, MiniLM, Cross-Attention).
