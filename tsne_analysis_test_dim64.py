import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# load data
latents = np.load("latentspace64/test_latent_vectors.npy")
labels  = np.load("latentspace64/test_latent_labels.npy")

# subsample 
n_samples = 10000
idx = np.random.choice(len(latents), n_samples, replace=False)
latents = latents[idx]
labels  = labels[idx]

print("Latents shape:", latents.shape)

# normalize
latents = StandardScaler().fit_transform(latents)

# t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=40,
    learning_rate=200,
    random_state=67
)

z_2d = tsne.fit_transform(latents)

# plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='nipy_spectral', s=5)
plt.colorbar(scatter, ticks=range(29))
plt.title("t-SNE of Latent Space (Dim = 64) - Test Set")

plt.savefig("tsne_dim64_test.png", dpi=300)
plt.show()