# Model 1: PIMA Baseline

The `PIMA_Baseline` directory (part of the PIMA-VAIPE system) contains the original source code for the **PIMA Baseline** model. This is the standard foundational version used for comparison and serves as the basis for other variants like PIMA_OCR or PIMA_ablation.

This model aims to solve the **matching** problem between the Pill Image and the Drug Name on the prescription. Unlike the automatic OCR version, this Baseline model utilizes **Ground Truth**, assuming that the bounding box coordinates and the text on the prescription have been extracted with 100% accuracy.

---

## 🔬 Technical Architecture

The Baseline utilizes two main neural network branches to extract multimodal features:
1. **Vision Branch:** Uses a `ResNet50` network to extract image features from cropped pill images.
2. **Text & Spatial Graph Branch:** Uses a Graph Neural Network (`GraphSAGE`). The graph is constructed from text nodes, with edges linked based on spatial coordinates on the prescription.

**Matching Mechanism:**
- The two feature streams (image and text/graph) are projected into the same embedding space.
- A **Contrastive Loss** function is used to optimize the distance: pulling correct pairs (pill image - drug name) closer and pushing incorrect pairs further apart.
- An **NLLLoss** function is applied as a supplementary loss for graph classification.

---

## ⚙️ Source Code Structure

- `train.py`: The main training script for the Baseline model. It loads data from the configured directory, performs training across epochs, and saves the best weights as `model_best.pth`. It automatically evaluates on the Validation set during training.
- `inference.py`: Script for running real-world predictions. When given a prescription data structure, the model computes the Cosine Similarity between the image and the text graph, using a soft threshold (> 0.5) to predict the corresponding drug or return "No match".
- `preprocess.py`: Preprocessing functions and initial graph creation for the standard model.
- `config.py`: File containing constants, configurations, hyperparameters, and system paths.
