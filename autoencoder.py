from tensorflow.keras import layers, Model

def build_encoder(input_shape=(64, 64, 1), latent_dim=64): #the encoder will reduce the whole image into a vector of length 64
    encoder_input = layers.Input(shape=input_shape, name="encoder_input")

    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu")(encoder_input)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x) # turns that into one long vector, because Dense layers need a 1D vector as input
    x = layers.Dense(128, activation="relu")(x) # learn a smaller, more useful representation of the flattened features.

    latent_vector = layers.Dense(latent_dim, name="latent_vector")(x)

    encoder = Model(encoder_input, latent_vector, name="encoder")
    return encoder


def build_decoder(latent_dim=64):
    decoder_input = layers.Input(shape=(latent_dim,), name="decoder_input")

    x = layers.Dense(4 * 4 * 256, activation="relu")(decoder_input)
    x = layers.Reshape((4, 4, 256))(x)

    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding="same", activation="relu")(x)     # 4 -> 8
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu")(x)      # 8 -> 16
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)      # 16 -> 32
    x = layers.BatchNormalization()(x)

    decoder_output = layers.Conv2DTranspose(
        1, (3, 3), strides=2, padding="same", activation="sigmoid", name="decoder_output"
    )(x)                                                                                           # 32 -> 64

    decoder = Model(decoder_input, decoder_output, name="decoder")
    return decoder


def build_autoencoder(input_shape=(64, 64, 1), latent_dim=64):
    encoder = build_encoder(input_shape=input_shape, latent_dim=latent_dim)
    decoder = build_decoder(latent_dim=latent_dim)

    autoencoder_input = layers.Input(shape=input_shape, name="autoencoder_input")
    latent = encoder(autoencoder_input)
    reconstructed = decoder(latent)

    autoencoder = Model(autoencoder_input, reconstructed, name="autoencoder")
    return encoder, decoder, autoencoder


if __name__ == "__main__":
    encoder, decoder, autoencoder = build_autoencoder()

    print("\nENCODER:")
    encoder.summary()

    print("\nDECODER:")
    decoder.summary()

    print("\nAUTOENCODER:")
    autoencoder.summary()

