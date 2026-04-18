import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. ЗАВАНТАЖЕННЯ ДАНИХ
# ==========================================
print("--- Loading Data ---")
try:
    train_vectors = np.load('train_latent_vectors.npy')
    train_labels = np.load('train_latent_labels.npy')
    val_vectors = np.load('val_latent_vectors.npy')
    val_labels = np.load('val_latent_labels.npy')
    test_vectors = np.load('test_latent_vectors.npy')
    test_labels = np.load('test_latent_labels.npy')
    class_names = np.load('latent_class_names.npy', allow_pickle=True)
    print("Files loaded successfully!")
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

# ==========================================
# 2. ТРЕНУВАННЯ ТА ВИБІР МОДЕЛІ
# ==========================================
print("\n--- Training MLP Model ---")
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    learning_rate_init=0.0005,
    max_iter=500,
    random_state=42
)

model.fit(train_vectors, train_labels)
val_acc = accuracy_score(val_labels, model.predict(val_vectors))
test_preds = model.predict(test_vectors)
test_acc = accuracy_score(test_labels, test_preds)

print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")

# ==========================================
# 3. ВІЗУАЛІЗАЦІЯ 1: CONFUSION MATRIX
# ==========================================
plt.figure(figsize=(16, 11)) 
cm = confusion_matrix(test_labels, test_preds)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)

plt.title(f'Confusion Matrix (Accuracy: {test_acc:.2%})', fontsize=16, pad=20)

plt.xlabel('Predicted Label', fontsize=14, labelpad=25) 
plt.ylabel('True Label', fontsize=14, labelpad=10)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.subplots_adjust(bottom=0.2, left=0.1, right=0.95, top=0.9)

plt.show()
plt.pause(0.5) 

# ==========================================
# 4. ВІЗУАЛІЗАЦІЯ 2: НАЙСКЛАДНІШІ ЗНАКИ (ПО ВПЕВНЕНОСТІ)
# ==========================================
print("\n--- Analyzing model confidence ---")
probs = model.predict_proba(test_vectors)
correct_probs = probs[np.arange(len(test_labels)), test_labels]

avg_confidences = []
for i in range(len(class_names)):
    idx = np.where(test_labels == i)[0]
    avg_confidences.append(np.mean(correct_probs[idx]))

sorted_idx = np.argsort(avg_confidences)[:10]
diff_labels = [class_names[i] for i in sorted_idx]
diff_values = [avg_confidences[i] for i in sorted_idx]

plt.figure(figsize=(13, 7))
colors = sns.color_palette("magma", len(diff_labels))
bars = plt.bar(diff_labels, diff_values, color=colors)

plt.ticklabel_format(useOffset=False, style='plain', axis='y')

min_v = min(diff_values)
plt.ylim(min_v - 0.0001, 1.0001) 

plt.title('Top 10 Most Difficult Signs (Confidence Analysis)', fontsize=15, pad=25)
plt.ylabel('Mean Confidence Score', fontsize=12)
plt.xlabel('Sign Class', fontsize=12)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, 
             f'{yval:.6f}', 
             va='bottom', ha='center', 
             fontsize=10, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

print("\nMost difficult sign statistics:")
for i in range(len(diff_labels)):
    print(f"Sign: {diff_labels[i]} | Confidence: {diff_values[i]:.8f}")

# ==========================================
# 5. ABLATION STUDY
# ==========================================
print("\n--- Running Ablation Study ---")
model_abl = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
model_abl.fit(train_vectors[:, :64], train_labels)
abl_acc = accuracy_score(val_labels, model_abl.predict(val_vectors[:, :64]))
print(f"Ablation Accuracy (64 features): {abl_acc:.4f}")
print("\nTask Complete.")

print("\n" + "#"*50 + "\nREPORT FOR: MAIN MODEL (128-dim Latent Space)\n" + "#"*50)