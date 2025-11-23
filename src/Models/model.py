from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input

from tensorflow.keras.applications.efficientnet import preprocess_input

import tensorflow as tf

def build_multitask_model(num_species, image_resolution=224):
    # EfficientNetB0 backbone, ImageNet weights
    inputs = Input(shape=(image_resolution, image_resolution, 3))

    #TODO experiment with different preprocessings
    x = preprocess_input(inputs)

    base = EfficientNetB0(include_top=False, weights="imagenet")
    x = base(x)
    base.trainable = False
    
    # Head
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)

    species_output = Dense(num_species, activation="softmax", name="species")(x)
    venom_output   = Dense(1, activation="sigmoid", name="venom")(x)

    #Final model
    model = Model(inputs=inputs, outputs=[species_output, venom_output])

    return model
