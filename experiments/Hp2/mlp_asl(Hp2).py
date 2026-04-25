import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# ==========================================
# 1. ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ
# ==========================================
print("--- Loading Data for PyTorch ---")
def load_to_tensor(vec_path, lab_path):
    v = np.load(vec_path).astype(np.float32)
    l = np.load(lab_path).astype(np.int64)
    return torch.from_numpy(v), torch.from_numpy(l)

train_X, train_y = load_to_tensor('train_latent_vectors.npy', 'train_latent_labels.npy')
test_X, test_y = load_to_tensor('test_latent_vectors.npy', 'test_latent_labels.npy')
class_names = np.load('latent_class_names.npy', allow_pickle=True)

# Створюємо DataLoader
train_ds = TensorDataset(train_X, train_y)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# ==========================================
# 2. АРХІТЕКТУРА MLP
# ==========================================
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

input_dim = train_X.shape[1]
num_classes = len(class_names)
model = MLP(input_dim, num_classes)

# ==========================================
# 3. НАЛАШТУВАННЯ ТРЕНУВАННЯ
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
epochs = 20

print("--- Training Start ---")
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# ==========================================
# 4. ОЦІНКА ТА ВІЗУАЛІЗАЦІЯ
# ==========================================
model.eval()
with torch.no_grad():
    outputs = model(test_X)
    _, predicted = torch.max(outputs, 1)
    probabilities = torch.softmax(outputs, dim=1)
    correct_probs = probabilities[torch.arange(len(test_y)), test_y].numpy()

test_acc = accuracy_score(test_y.numpy(), predicted.numpy())

# Побудова Confusion Matrix
plt.figure(figsize=(16, 11))
cm = confusion_matrix(test_y.numpy(), predicted.numpy())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'PyTorch Confusion Matrix (Acc: {test_acc:.2%})')
plt.xlabel('Predicted Label', labelpad=20)
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(bottom=0.2)
plt.show()

# Аналіз впевненості (Confidence)
model.eval()
with torch.no_grad():
    outputs = model(test_X)
    probabilities = torch.softmax(outputs, dim=1)
    correct_probs = probabilities[torch.arange(len(test_y)), test_y].numpy()

avg_conf = [np.mean(correct_probs[test_y.numpy() == i]) for i in range(num_classes)]
sorted_idx = np.argsort(avg_conf)[:10]

diff_labels = [class_names[i] for i in sorted_idx]
diff_values = [avg_conf[i] for i in sorted_idx]

plt.figure(figsize=(12, 7))

colors = sns.color_palette("flare", len(diff_labels))

bars = plt.bar(diff_labels, diff_values, color=colors, edgecolor='black', linewidth=0.5)

plt.ticklabel_format(useOffset=False, style='plain', axis='y')

plt.ylim(min(diff_values) - 0.00005, 1.00005)

plt.title('Top 10 Most Difficult Signs to Classify (Confidence Analysis)', fontsize=15, pad=25)
plt.ylabel('Mean Softmax Confidence', fontsize=12)
plt.xlabel('Sign Class', fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.3)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.000005, 
             f'{yval:.6f}', 
             va='bottom', ha='center', 
             fontsize=10, fontweight='bold', color='#333333')

plt.tight_layout()
plt.show()


print("\n" + "#"*50 + "\nREPORT FOR: HYPERPARAM 2 | Learning Rate: 5e-4\n" + "#"*50)
print(f"Final Accuracy: {test_acc:.4f}")