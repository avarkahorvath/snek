from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, Dense, Input, MaxPooling2D, Dropout, Rescaling
from tensorflow.keras import Model

from re import X

def build_cnn(num_species: int, IMAGE_RESOLUTION=28):
    input = Input(shape=(IMAGE_RESOLUTION, IMAGE_RESOLUTION, 3))
    #input = Input(shape=(None, None, 3))
    x = Rescaling(1./255)(input)

    x = Conv2D(32, (3,3), padding='same', activation="relu")(input)
    x = MaxPooling2D((2,2), padding='valid')(x)

    x = Conv2D(64, (3,3), padding='same', activation="relu")(x)
    x = MaxPooling2D((2,2), padding='valid')(x)

    x = Conv2D(128, (3,3), padding='same', activation="relu")(x)
    x = MaxPooling2D((2,2), padding='valid')(x)

    x = Conv2D(32, (3,3), padding='same', activation="relu")(x)
    x = MaxPooling2D((2,2), padding='valid')(x)

    x = GlobalAveragePooling2D()(x)
    #x = Flatten()(x)

    x = Dense(128, activation='relu')(x)

    x = Dropout(0.3)(x)

    species_output = Dense(num_species, activation='softmax', name='species')(x)
    venom_output = Dense(1, activation='sigmoid', name='venom')(x)

    model = Model(inputs=input, outputs=[species_output, venom_output])

    return model