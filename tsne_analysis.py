import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load latent vectors from autoencoder
latents = np.load("latents/latent_vectors.npy")
labels  = np.load("latents/latent_labels.npy")

# Fix size to match test set (sometimes off by 1)
latents = latents[:17400]
labels  = labels[:17400]

print("Latents shape:", latents.shape)

# Reduce dimensions from 128D to 2D for visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=67)
z_2d = tsne.fit_transform(latents)

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab20', s=5)
plt.colorbar()
plt.title("t-SNE of Latent Space (ASL)")

# Save plot for report
plt.savefig("tsne_plot.png", dpi=300)

plt.show()