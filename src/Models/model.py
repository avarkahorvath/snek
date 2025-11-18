from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf

def build_multitask_model(num_species, image_resolution=224):
    # 1) EfficientNetB0 backbone, ImageNet súlyokkal, 3 csatornás inputtal
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_resolution, image_resolution, 3)
    )
    base.trainable = False

    # 2) Head
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)

    species_output = Dense(num_species, activation="softmax", name="species")(x)
    venom_output   = Dense(1, activation="sigmoid", name="venom")(x)

    # 3) A model inputja a base.input lesz
    model = Model(inputs=base.input, outputs=[species_output, venom_output])

    return model
