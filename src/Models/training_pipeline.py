#!/usr/bin/env python
# coding: utf-8

# # Library imports, setup

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from data import load_metadata, visualize_data, make_dataset
from model import build_multitask_model
from score_metrics import get_scores
from loss import SoftF1Loss


# In[ ]:


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


# # Loading data

# In[ ]:


image_metadata, species_metadata = load_metadata()
num_classes = len(species_metadata)


# # Visualizing data

# In[ ]:


#in data.py
visualize_data(image_metadata)


# Loading python images from folder

# # Building model

# In[ ]:


import tensorflow as tf
import keras


# In[ ]:


IMAGE_RESOLUTION=224
from data import make_batches, split_dataset

#szükség van külön a train infora is
train_info, val_info, test_info = split_dataset(image_metadata)
train_dataset = make_batches(train_info, IMAGE_RESOLUTION)
val_dataset   = make_batches(val_info, IMAGE_RESOLUTION)
test_dataset  = make_batches(test_info, IMAGE_RESOLUTION)


#train_dataset, val_dataset, test_dataset = make_dataset(image_metadata, IMAGE_RESOLUTION)


# In[ ]:


model = build_multitask_model(num_species=num_classes, image_resolution=224)
#model.summary()


# In[ ]:


lr = 1e-4

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    
    loss={'species': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          'venom': 'binary_crossentropy',},
    loss_weights={"species": 1.0, "venom": 0.5},
    metrics={'species': 'accuracy',
             'venom': 'accuracy'}
    )


# In[ ]:


checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

early_stop_cb = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,           # 6 epoch javulás nélkül → megáll
    restore_best_weights=True,
    verbose=1,
)

reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-6,
    verbose=1,
)


# In[ ]:


from sklearn.utils.class_weight import compute_class_weight

# species class_weight
species_classes = np.unique(train_info["encoded_id"])
species_cw = compute_class_weight(
    class_weight="balanced",
    classes=species_classes,
    y=train_info["encoded_id"]
)
species_cw_dict = {int(c): w for c, w in zip(species_classes, species_cw)}

# venom class_weight 
venom_classes = np.unique(train_info["MIVS"]) 
venom_cw = compute_class_weight(
    class_weight="balanced",
    classes=venom_classes,
    y=train_info["MIVS"]
)
venom_cw_dict = {int(c): w for c, w in zip(venom_classes, venom_cw)}



# In[ ]:


num_species = len(species_cw_dict)


# In[ ]:


n_epochs = 10

# checkpointing based on the validation loss
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    'best_weights.weights.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)


early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

#reduce loss rate
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-6
)


model_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=n_epochs,
    callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
)

model.load_weights( 'best_weights.weights.h5')  # load weights back


# In[ ]:


test_history = model.evaluate(test_dataset)
print("Test Loss: ", test_history[0])
print("Test Accuracy: ", test_history[1])


# # Example results

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def example_results_from_dataset(model, ds, species_names, n_examples=5, venom_threshold=0.5):
    """
    ds must yield: (image, {'species': int, 'venom': int})
    species_names: list where index == encoded_id
    """
    ds = ds.unbatch().shuffle(1000)
    # unbatch to individual samples and take a few
    samples = list(ds.take(n_examples))
    #samples = list(ds.unbatch().take(n_examples))
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





# In[ ]:


example_results_from_dataset(model, test_dataset, species_metadata, n_examples=5)


# # Calculating scoring metrics

# Function to tell if the species is venomous or not, based on encoded_id

# In[ ]:


get_scores(model, image_metadata, test_dataset, venom_threshold=0.5)

