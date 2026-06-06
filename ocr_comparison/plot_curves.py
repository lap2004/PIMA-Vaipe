import re
import matplotlib.pyplot as plt
import os

def parse_log(log_file):
    if not os.path.exists(log_file):
        return [], []
        
    losses = []
    accuracies = []
    current_loss = None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        if "100%|" in line and "loss=" in line:
            match = re.search(r'loss=([0-9.]+)', line)
            if match:
                current_loss = float(match.group(1))
                
        if "Val accuracy:" in line:
            acc_match = re.search(r'Val accuracy:\s+([0-9.]+)', line)
            if acc_match:
                acc = float(acc_match.group(1)) * 100
                if current_loss is not None:
                    losses.append(current_loss)
                    accuracies.append(acc)
                    current_loss = None
                    
    return losses, accuracies

def plot_combined_curves():
    ocr_losses, ocr_accs = parse_log('ocr.log')
    ablation_losses, ablation_accs = parse_log('ocr_ablation.log')
    
    if not ocr_losses and not ablation_losses:
        print("No log files found yet.")
        return
        
    plt.figure(figsize=(12, 5))
    
    # Plot Losses
    plt.subplot(1, 2, 1)
    if ocr_losses:
        plt.plot(range(1, len(ocr_losses) + 1), ocr_losses, 'b-', label='OCR Model Loss')
    if ablation_losses:
        plt.plot(range(1, len(ablation_losses) + 1), ablation_losses, 'r-', label='No-OCR Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    
    # Plot Accuracies
    plt.subplot(1, 2, 2)
    if ocr_accs:
        plt.plot(range(1, len(ocr_accs) + 1), ocr_accs, 'b-', label='OCR Model Accuracy')
    if ablation_accs:
        plt.plot(range(1, len(ablation_accs) + 1), ablation_accs, 'r-', label='No-OCR Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Comparison (Ablation Study)')
    plt.legend()
    
    plt.tight_layout()
    output_file = 'learning_curves.png'
    plt.savefig(output_file)
    print(f"Successfully generated {output_file} comparing both models!")

if __name__ == '__main__':
    plot_combined_curves()
