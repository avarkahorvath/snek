# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Library imports, setup

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from data import load_metadata, visualize_data, make_dataset
from model import build_cnn
from score_metrics import get_scores

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7956, "status": "ok", "timestamp": 1760103668680, "user": {"displayName": "Avarka", "userId": "01376155912533068519"}, "user_tz": -120} id="cb7d91df" outputId="632fa06d-510c-411c-bd31-1a43c8eb8343"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# check tf version
print(tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
for device in gpus:
    tf.config.experimental.set_memory_growth(device, True)
    print(f"Found GPU {device.name}, and set memory growth to True.")


# %% [markdown]
# # Loading data

# %%
image_metadata, species_metadata = load_metadata()
num_classes = len(species_metadata)

# %% [markdown]
# # Visualizing data

# %%
#in data.py
visualize_data(image_metadata)

# %% [markdown] id="d98e31d7"
# Loading python images from folder

# %% [markdown]
# # Building model

# %%
IMAGE_RESOLUTION=28

train_dataset, val_dataset, test_dataset = make_dataset(image_metadata, IMAGE_RESOLUTION)

# %%
model=build_cnn(num_classes, IMAGE_RESOLUTION)
model.summary()

# %%
model.compile(
    optimizer='adam',
    
    loss={'species': 'sparse_categorical_crossentropy',
          'venom': 'binary_crossentropy'},

    metrics={'species': 'accuracy',
             'venom': 'accuracy'}
    )

# %%
n_epochs = 10 

# checkpointing based on the validation loss
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('model.keras', monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)

# model training
model_history = model.fit(
                            x= train_dataset,
                            epochs= n_epochs,
                            validation_data= val_dataset,
                            callbacks=[model_checkpoint_callback])

model.load_weights('model.keras')  # load weights back

# %%
test_history = model.evaluate(test_dataset)
print("Test Loss: ", test_history[0])
print("Test Accuracy: ", test_history[1])

# %% [markdown]
# # Example results

# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def example_results_from_dataset(model, ds, species_names, n_examples=5, venom_threshold=0.5):
    """
    ds must yield: (image, {'species': int, 'venom': int})
    species_names: list where index == encoded_id
    """
    # unbatch to individual samples and take a few
    samples = list(ds.unbatch().take(n_examples))
    imgs = [x[0] for x in samples]
    lbls = [x[1] for x in samples]

    # stack to a batch for one predict()
    x_batch = tf.stack(imgs, axis=0)
    pred_species_logits, pred_venom_prob = model.predict(x_batch, verbose=0)

    plt.figure(figsize=(3.3 * len(imgs), 3.3))
    for i, (img, y) in enumerate(zip(imgs, lbls), start=1):
        true_species = int(y["species"].numpy())
        true_venom   = int(y["venom"].numpy())

        pred_species = int(np.argmax(pred_species_logits[i-1]))
        pred_venom   = bool(float(pred_venom_prob[i-1][0]) > venom_threshold)

        true_name = species_names[true_species] if 0 <= true_species < len(species_names) else str(true_species)
        pred_name = species_names[pred_species] if 0 <= pred_species < len(species_names) else str(pred_species)

        plt.subplot(1, len(imgs), i)
        plt.imshow(img.numpy())
        plt.axis("off")
        plt.title(
            f"True: {true_name} ({'Venom' if true_venom else 'Safe'})\n"
            f"Pred: {pred_name} ({'Venom' if pred_venom else 'Safe'})",
            fontsize=9
        )
    plt.tight_layout(); plt.show()



# %%
example_results_from_dataset(model, test_dataset, species_metadata, n_examples=5)


# %% [markdown]
# # Calculating scoring metrics

# %% [markdown]
# Function to tell if the species is venomous or not, based on encoded_id

# %%
get_scores(model, image_metadata, test_dataset, venom_threshold=0.5)
