from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt

from data_prep import load_preprocessed_datasets
from autoencoder import build_autoencoder

DATA_DIR = r"D:\ns\asl_alphabet_train"
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
SEED = 67
LATENT_DIM = 64
EPOCHS = 21
LEARNING_RATE = 1e-3

MODELS_DIR = Path("models")
PLOTS_DIR = Path("outputs/plots")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def make_autoencoder_dataset(dataset):

    return dataset.map(lambda images, labels: (images, images),
                       num_parallel_calls=tf.data.AUTOTUNE)

def plot_history(history, save_path="outputs/plots/ae_loss_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def show_reconstructions(model, dataset, save_path="outputs/plots/ae_reconstructions.png", num_images=6):
    images, _ = next(iter(dataset))
    reconstructed = model.predict(images[:num_images], verbose=0)

    plt.figure(figsize=(12, 4))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(tf.squeeze(images[i]), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(tf.squeeze(reconstructed[i]), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":

    train_ds, val_ds, test_ds, class_names = load_preprocessed_datasets(
        DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    train_ae_ds = make_autoencoder_dataset(train_ds)
    val_ae_ds = make_autoencoder_dataset(val_ds)
    test_ae_ds = make_autoencoder_dataset(test_ds)

    encoder, decoder, autoencoder = build_autoencoder(
        input_shape=(64, 64, 1),
        latent_dim=LATENT_DIM
    )

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )

    autoencoder.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "best_autoencoder.keras"),
            monitor="val_loss",
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    history = autoencoder.fit(
        train_ae_ds,
        validation_data=val_ae_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    autoencoder.save(MODELS_DIR / "final_autoencoder.keras")
    encoder.save(MODELS_DIR / "encoder_only.keras")
    decoder.save(MODELS_DIR / "decoder_only.keras")

    test_loss = autoencoder.evaluate(test_ae_ds, verbose=1)
    print(f"\nTest MSE Loss: {test_loss:.6f}")

    plot_history(history, save_path=str(PLOTS_DIR / "ae_loss_curve.png"))
    show_reconstructions(
        autoencoder,
        test_ae_ds,
        save_path=str(PLOTS_DIR / "ae_reconstructions.png"),
        num_images=6
    )
