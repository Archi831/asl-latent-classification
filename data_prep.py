import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt


DATA_DIR = r"D:\ns\asl_alphabet_train"
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
SEED = 67
AUTOTUNE = tf.data.AUTOTUNE # automatically choose the best number of parallel calls/prefetch size

def load_preprocessed_datasets(data_dir, img_size=(64, 64), batch_size=32, seed=67):

    data_dir = Path(data_dir)

    train_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir,  # folder that contains class subfolders
        labels="inferred", # labels are taken automatically from subfolder names
        label_mode="int", # labels will be integers (e.g. 0, 1, 2, ...)
        color_mode="grayscale",
        image_size=img_size,
        batch_size=None,
        shuffle=False,
        seed=seed,
        validation_split=0.30,
        subset="training",
    )

    temp_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        image_size=img_size,
        batch_size=None,
        shuffle=False,
        seed=seed,
        validation_split=0.30,
        subset="validation",
    )

    class_names = train_raw.class_names
    num_classes = len(class_names)
    print(f"\nFound {num_classes} classes:")
    print(class_names)
    if num_classes != 29:
        print(f"WARNING: Expected 29 classes, but found {num_classes}.")

    temp_count = tf.data.experimental.cardinality(temp_raw).numpy()
    val_count = temp_count // 3

    val_raw = temp_raw.take(val_count)
    test_raw = temp_raw.skip(val_count)

    def normalize(image, label):
        image = tf.cast(image, tf.float32) / 255.0 # converts pixel values to decimal numbers
        # neural networks usually train better when inputs are scaled to small values
        return image, label

    train_ds = train_raw.map(normalize, num_parallel_calls=AUTOTUNE)
    val_ds = val_raw.map(normalize, num_parallel_calls=AUTOTUNE)
    test_ds = test_raw.map(normalize, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.shuffle(10000, seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

def count_samples(dataset):
    total = 0
    for images, labels in dataset:
        total += images.shape[0]
    return total

def show_sample_images(dataset, class_names, num_images=9):
    images, labels = next(iter(dataset))

    plt.figure(figsize=(8, 8))
    for i in range(min(num_images, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(tf.squeeze(images[i]), cmap="gray")
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    train_ds, val_ds, test_ds, class_names = load_preprocessed_datasets(
        DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    train_count = count_samples(train_ds)
    val_count = count_samples(val_ds)
    test_count = count_samples(test_ds)
    total_count = train_count + val_count + test_count

    print("\nDataset sizes:")
    print(f"Train: {train_count}")
    print(f"Validation: {val_count}")
    print(f"Test: {test_count}")
    print(f"Total: {total_count}")

    images, labels = next(iter(train_ds))
    print("\nOne training batch:")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Min pixel value:", tf.reduce_min(images).numpy())
    print("Max pixel value:", tf.reduce_max(images).numpy())

    show_sample_images(train_ds, class_names)