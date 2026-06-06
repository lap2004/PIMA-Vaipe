import json
import matplotlib.pyplot as plt

# Set serif font to match previous styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

# Load data from the generated JSON
with open('train_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

epochs = [d['epoch'] for d in data]
train_losses = [d['train_loss'] for d in data]
val_losses = [d['val_loss'] for d in data]
train_accuracies = [d['train_accuracy'] for d in data]
val_accuracies = [d['val_accuracy'] for d in data]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

blue_color = '#385d8a'
red_color = '#c0504d'
lw = 2.5 # thicker lines like the image

# Plot Accuracy (Subplot 1)
ax1.plot(epochs, train_accuracies, marker='o', markersize=4, linewidth=lw, color=blue_color, label='Train Accuracy')
ax1.plot(epochs, val_accuracies, marker='s', markersize=4, linewidth=lw, color=red_color, label='Validation Accuracy')
ax1.set_title('(a) Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Matching Accuracy (%)')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower right', shadow=True)

# Plot Loss (Subplot 2)
ax2.plot(epochs, train_losses, marker='o', markersize=4, linewidth=lw, color=blue_color, label='Train Loss')
ax2.plot(epochs, val_losses, marker='s', markersize=4, linewidth=lw, color=red_color, label='Validation Loss')
ax2.set_title('(b) Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Total Loss')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', shadow=True)

plt.tight_layout()
plt.savefig('test.png', dpi=300)
print("Saved plot from JSON to test.png")
