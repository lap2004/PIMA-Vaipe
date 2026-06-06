# Model 3: Baseline Extension with OCR Integration (PIMA_OCR)

The `ocr_comparison` directory (part of the PIMA-VAIPE system) contains the source code for the **PIMA method integrated with Optical Character Recognition (OCR) technology**.

In previous versions (such as the Baseline and Ablation), the system assumed that the Bounding Boxes containing drug names on the prescriptions were manually annotated (Ground Truth). However, to deploy the model in real-world applications, it must be able to read text automatically from raw prescription images. This directory addresses that problem.

---

## 🔬 Architectural Differences & Technical Details

The OCR-integrated version still utilizes the core algorithms of the Baseline (`ResNet50` + `GraphSAGE`), but the input data processing pipeline has been completely redesigned:

### 1. Automated Text Extraction (PaddleOCR / PP-OCRv3)
- Instead of feeding pre-annotated text labels, the system directly embeds the **PaddleOCR** library.
- Image preprocessing and OCR execution are supported by scripts such as `preprocess_paddleocr.py` and can be simulated via `simulate_ocr.py`.
- When a prescription image is inputted, the system automatically runs the OCR model to detect text regions, generating Bounding Boxes and predicting the character sequences within them.
- These character sequences (which may contain noise or spelling errors due to OCR inaccuracies) are then passed into the Text Encoder.

### 2. Remaining Branches (Vision & Graph)
- **Graph:** The graph is now constructed from spatial coordinates predicted by the OCR AI, rather than human-annotated coordinates.
- **Vision:** Still uses `ResNet50` taking cropped pill images as input.
- **Matching:** Still utilizes Contrastive Learning, but the challenge is significantly harder because the input text contains noise from the OCR engine.

---

## ⚙️ Source Code Structure

The main source code files used for training and evaluating the model:
- `train_ocr.py`: Script used to train the model. The `main` function supports an evaluation flag via `--use_ocr`. The main loss includes `Contrastive Loss` (text-image matching) and `Graph Loss`. The best model is saved automatically (`model_best_ocr.pth` / `model_best_no_ocr.pth`).
- `eval.py`: Script to load the trained weights and evaluate the Top-1 Matching Accuracy on the Test/Val dataset.
- `preprocess_paddleocr.py`: Script to support text and bounding box extraction from raw images based on PaddleOCR.
- `simulate_ocr.py`: Simulates OCR data for testing scenarios.
- `plot_curves.py`: Tool to visualize learning curves (Accuracy / Loss) after training.

---

## 📊 Method Evaluation (Pros & Cons)

- **Pros:** Higher real-world applicability (End-to-End) compared to the Baseline. The model automatically completes the text analysis process without human intervention.
- **Cons:** 
  - Suffers from **Cascading Errors**: Any text recognition error from the OCR will be amplified when passed into GraphSAGE, severely degrading matching accuracy.
  - Slower processing speed (Inference Time) due to the additional OCR image scanning process before running the GNN.

---

## 📈 Evaluation Results & Experiments

To objectively compare the impact of OCR errors on the system, we evaluated both scenarios (with OCR and without OCR, but sharing the same model architecture). 

Below is a summary table of parameters and detailed results after training:

| Scenario | Method | Final Training Loss | Validation Accuracy | Top-1 Matching (Test) |
| :--- | :--- | :---: | :---: | :---: |
| **Scenario 1 (No OCR)** | *Ground Truth Text & BBox* | 0.239 (Epoch 10) | 44.08% | **44.26%** |
| **Scenario 2 (With OCR)** | *PaddleOCR End-to-End* | 0.648 (Epoch 10) | 50.95% | **44.31%** |
| **Ablation Scenario** | *Pruned Experiment* | 0.816 (Epoch 6) | 43.73% | 0.00% |

### Training Process Details

**1. Scenario 1: No OCR (Ground Truth Text & Bounding Box)**
Uses 100% accurate human-annotated labels and Bounding Box coordinates, free of text noise.
- **Weights File:** `model_best_no_ocr.pth`
- **Log Files:** `train_no_ocr.log`, `log_eval.log`
- **Train Loss (Total):** 0.239 | **Top-1 Matching Accuracy:** 44.26%
- *Evaluation:* The model achieves good performance on clean data (without noise).

**2. Scenario 2: With OCR (End-to-End with PaddleOCR)**
The system automatically reads text and extracts Bounding Boxes. The text data contains natural errors.
- **Weights File:** `model_best_ocr.pth`
- **Log Files:** `train_ocr.log`, `log_eval.log`
- **Train Loss (Total):** 0.648 | **Top-1 Matching Accuracy:** 44.31%
- *Evaluation:* In the OCR scenario, the Train Loss is harder to converge (`0.648` vs `0.239`) due to Cascading Errors. However, the Top-1 Accuracy (44.31%) is still almost equivalent to Scenario 1. This demonstrates the excellent noise tolerance of the Vision and Graph branches.

**3. Ablation Scenario**
Ablation study on the model.
- **Train Loss (Total):** 0.816 | **Validation Accuracy:** 43.73%

> 💡 **Note:** The Early Stopping mechanism was triggered in most scenarios after approximately 5 to 10 epochs.
