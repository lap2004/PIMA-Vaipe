# Model 2: Ablation Study Experiment (PIMA Ablation Study)

The `pima_ablation_study` directory (part of the PIMA-VAIPE system) contains the source code for the **Ablation Study (Component Pruning Experiment)**. 

In Deep Learning research, an Ablation Study is a crucial step to prove the effectiveness of a system. We fine-tune, modify, or remove some core modules of the original PIMA method (e.g., changing the Loss function, modifying how the graph is combined) to observe how performance changes on the **VAIPE** dataset.

---

## 🔬 Tested Architectures (Ablations)

In this version, the foundational technologies mostly remain identical to the PIMA Baseline (`ResNet50` for images, `GraphSAGE` for text graphs). However, several configurations have been altered for measurement:

### 1. Loss Function Variations
- The system experiments with fine-tuning the weights of the **Contrastive Loss** and **BCE/NLL Loss** to find the optimal balance between text node classification and image-text matching.
- During inference, samples are filtered using a strict softmax threshold (0.8) to evaluate the model's confidence in the learned features.

### 2. Graph Structure Variations
- Bypassing parts of the GraphSAGE algorithm to make a direct comparison: Is pure matching (pure Text - pure Image) better than using a Graph? This demonstrates the essential role of GNNs.

---

## ⚙️ Source Code Structure

- `train_ablation.py`: The main training script in the Ablation directory, integrating graph and image feature combination, and using `ContrastiveLoss` to measure the similarity between pill images and text. The best model is saved at `./logs/weights/model_best.pth`.
- `eval.py`: Loads the best weights after training and calculates the Top-1 Matching Accuracy on the Test dataset.
- `config.py`: General configuration file storing parameters.
- `models` and `utils` directories: Contain the `PrescriptionPill` model uniquely adjusted for the ablation study, along with utility functions like Data Loaders and Metrics.

---

## 📊 Experimental Analysis (Pros & Cons)

- **Role in research:** A perfect stepping stone to identify the core weaknesses of the old architecture. Thanks to the Ablation Study, we realized that minor tweaks (changing Loss, changing Data Augmentation) are **insufficient** to overcome the unique barriers of the VAIPE dataset.
- **Development foundation:** The limited results from this directory are the exact reason and compelling motivation to rebuild the entire architecture from scratch (leading to the creation of the most complete `PIMA_NEW` version).

---

## 📈 Evaluation Results & Experiments

Below are the detailed metrics recorded during the training of the Ablation model to measure performance changes:

| Model | Early Stopping Epoch | Train Loss | Validation Accuracy | Top-1 Matching |
| :--- | :---: | :---: | :---: | :---: |
| **Ablation Model** | Epoch 9 | 0.862 | 50.52% | **26.23%** |

### Training Process Details
- **Training log file:** `ablation.log`
- **Evaluation log file:** `log_eval.log`
- **Training Epochs reached:** 9 (The model converged fairly quickly and triggered Early Stopping at epoch 9)
- **Train Loss (Total Loss):** 0.862
- **Validation Accuracy (during training):** 50.52%
- **Top-1 Matching Accuracy (on Test set):** 26.23%

### Overall Remarks

- **Regarding Train Loss:** The achieved loss is `0.862`, indicating that the training process of the pruned network architecture faced certain difficulties compared to the original PIMA version. 
- **Regarding Accuracy:** Although the **Validation Accuracy** during training reached a fairly good level (over 50%), when performing actual matching on the Test set, the **Top-1 Matching Accuracy** dropped sharply, stopping at only **26.23%**. This discrepancy shows that the model is prone to overfitting or struggles to generalize to real-world matching data, partly due to the overly rigid softmax threshold filtering or the removal of essential graph features.
- **Conclusion from Ablation Study:** Modifying the graph structure or loss function in the Ablation experiment did not yield the ideal expected performance. This result is crucial empirical evidence, helping to detect core weaknesses in the feature extraction and combination process. As a result, we have sufficient grounds to comprehensively upgrade to a superior architecture (the `PIMA_NEW` version).