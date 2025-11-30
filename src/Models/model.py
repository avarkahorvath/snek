from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout,
    RandomFlip, RandomRotation, RandomZoom, RandomContrast
)
import tensorflow as tf

def build_multitask_model(num_species, image_resolution=224):
    # EfficientNetB0 backbone, ImageNet weights
    inputs = Input(shape=(image_resolution, image_resolution, 3))

    
    #Random rotates (+-9°), Flips,Zooms
    data_augmentation = tf.keras.Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(factor=0.05,     ),
            RandomZoom(height_factor=0.1, width_factor=0.1),
            RandomContrast(0.1),
        ],
        name="data_augmentation"
    )
    x = data_augmentation(inputs)

    #TODO experiment with different preprocessings
    x = preprocess_input(x)


    base = EfficientNetB0(include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x)
    
    
    # Head
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)

    species_output = Dense(num_species, activation="softmax", name="species")(x)
    venom_output   = Dense(1, activation="sigmoid", name="venom")(x)

    #Final model
    model = Model(inputs=inputs, outputs=[species_output, venom_output])

    return model
