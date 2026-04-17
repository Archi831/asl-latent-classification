import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier

# -------------------- Завантаження даних --------------------
X = np.load("latent_vectors.npy")   
y = np.load("latent_labels.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# -------------------- Розділення на тренувальні та тестові --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# -------------------- Створення та тренування моделі --------------------
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42,
    verbose=True
)

model.fit(X_train, y_train)

# -------------------- Оцінка моделі --------------------
y_pred_classes = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Test Accuracy: {accuracy:.4f}")

# -------------------- Матриця конфузій --------------------
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues", annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# -------------------- Збереження матриці конфузій у CSV з підписами класів --------------------
class_labels = np.unique(y)  # отримуємо всі класи
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
cm_df.to_csv("confusion_matrix_labeled.csv")
print("Confusion matrix saved as confusion_matrix_labeled.csv")

# -------------------- Топ 5 помилок --------------------
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)

errors = [(i, j, cm_no_diag[i, j]) 
          for i in range(cm_no_diag.shape[0]) 
          for j in range(cm_no_diag.shape[1]) 
          if cm_no_diag[i, j] > 0]

errors = sorted(errors, key=lambda x: x[2], reverse=True)

print("Top 5 confused class pairs:")
for e in errors[:5]:
    print(f"True: {e[0]} -> Predicted: {e[1]} | Count: {e[2]}")

# -------------------- Графік втрат (loss_curve_) --------------------
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_, label="Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

# Беремо топ-5 помилок
top_errors = errors[:5]

# Унікальні класи серед цих помилок
classes = list(set([e[0] for e in top_errors] + [e[1] for e in top_errors]))
classes.sort()  # для порядку
class_to_idx = {c:i for i,c in enumerate(classes)}

# Створюємо маленьку матрицю тільки для цих класів
mini_cm = np.zeros((len(classes), len(classes)), dtype=int)
for e in top_errors:
    i = class_to_idx[e[0]]
    j = class_to_idx[e[1]]
    mini_cm[i, j] = e[2]

# ---------------- Heatmap ----------------
plt.figure(figsize=(6,6))
sns.heatmap(mini_cm, annot=True, fmt='d', cmap="Reds", xticklabels=classes, yticklabels=classes)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Top 5 Most Confused Sign Pairs")
plt.gca().invert_yaxis()  # щоб True класи були зверху
plt.show()