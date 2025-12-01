#!/usr/bin/env python
# coding: utf-8

# # Library imports, setup

# In[1]:


#if you change a file, you dont have to restart the kernel
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from data import load_metadata, visualize_data, make_dataset
from model import build_multitask_model
from score_metrics import get_scores
from loss import SoftF1Loss #custom loss function, currently not used


# In[3]:


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

# In[4]:


image_metadata, species_metadata = load_metadata()
NUM_SPECIES = len(species_metadata)


# # Visualizing data

# In[5]:


#in data.py
visualize_data(image_metadata)


# Loading python images from folder

# # Building model

# In[6]:


import tensorflow as tf
import keras


# In[7]:

from model import build_multitask_model
from score_metrics import get_scores
from loss import SoftF1Loss 

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
IMAGE_RESOLUTION=544
NUM_FOLDS = 3


from data import make_batches, split_dataset



X_paths = image_metadata["image_path"].values
y_species = image_metadata["encoded_id"].values

skf = StratifiedKFold(
    n_splits=NUM_FOLDS,
    shuffle=True,
    random_state=42,
)

fold_metrics = []


all_y_species_true = []
all_y_species_pred = []
all_y_venom_true = []
all_y_venom_pred = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_paths, y_species), start=1):
    print(f"\n===== FOLD {fold_idx}/{NUM_FOLDS} =====")

    train_info = image_metadata.iloc[train_idx].copy()
    val_info   = image_metadata.iloc[val_idx].copy()

    # --- class weight csak a trainre számolva ---
    species_classes = np.unique(train_info["encoded_id"])
    species_cw = compute_class_weight(
        class_weight="balanced",
        classes=species_classes,
        y=train_info["encoded_id"],
    )
    species_cw_dict = {int(c): w for c, w in zip(species_classes, species_cw)}

    species_weight_vec = tf.constant(
        [species_cw_dict[i] for i in range(len(species_cw_dict))],
        dtype=tf.float32,
    )

    # --- tf.data datasetek ---
    train_dataset = make_batches(
        train_info,
        IMAGE_RESOLUTION,
        species_weight_vec=species_weight_vec,
    )

    val_dataset = make_batches(
        val_info,
        IMAGE_RESOLUTION,
        species_weight_vec=None,
    )

    # --- modell: minden foldban újraépítjük ---
    model = build_multitask_model(
        num_species=NUM_SPECIES,
        image_resolution=IMAGE_RESOLUTION,
    )

    lr = 1e-4  # EfficientNetB0-hoz alacsony LR

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=[
            tf.keras.losses.SparseCategoricalCrossentropy(),
            "binary_crossentropy",
        ],
        loss_weights=[1.0, 0.5],
        metrics=["accuracy", "accuracy"],
    )

    # --- callbackek fold-specifikus checkpointtal ---
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        f"best_model_fold{fold_idx}.keras",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    early_stop_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
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

    n_epochs = 5

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=n_epochs,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
    )

    # --- saját metrikák foldonként, get_scores-szal ---
    metrics_fold = get_scores(
        model,
        image_metadata=image_metadata,
        test_dataset=val_dataset,
        venom_threshold=0.5,
    )

    fold_metrics.append(metrics_fold)

    all_y_species_true.append(metrics_fold["y_species_true"])
    all_y_species_pred.append(metrics_fold["y_species_pred"])
    all_y_venom_true.append(metrics_fold["y_venom_true"])
    all_y_venom_pred.append(metrics_fold["y_venom_pred"])

# --- aggregált CV-eredmények (átlag metrikák + összesített predikciók) ---
results_own_metrics = {
    "species_accuracy": np.mean([m["species_accuracy"] for m in fold_metrics]),
    "macro_f1": np.mean([m["macro_f1"] for m in fold_metrics]),
    "venom_accuracy": np.mean([m["venom_accuracy"] for m in fold_metrics]),
    "venom_weighted_species_accuracy": np.mean(
        [m["venom_weighted_species_accuracy"] for m in fold_metrics]
    ),
    "y_species_true": np.concatenate(all_y_species_true, axis=0),
    "y_species_pred": np.concatenate(all_y_species_pred, axis=0),
    "y_venom_true": np.concatenate(all_y_venom_true, axis=0),
    "y_venom_pred": np.concatenate(all_y_venom_pred, axis=0),
}

print("\n=== Átlagolt keresztvalidációs eredmények ===")
print(f"Species accuracy (val): {results_own_metrics['species_accuracy']:.4f}")
print(f"Macro-F1 (species, val): {results_own_metrics['macro_f1']:.4f}")
print(f"Venom accuracy (val): {results_own_metrics['venom_accuracy']:.4f}")
print(
    "Venom-weighted species accuracy (val): "
    f"{results_own_metrics['venom_weighted_species_accuracy']:.4f}"
)




# # Example results

# In[26]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

def example_results_from_dataset(model, ds, species_names, n_examples=5, venom_threshold=0.5):
    """
    Jelenlegi pipeline:
    ds yield: (img, (species, venom), (species_w, venom_w))
    """

    # csak (img, labels) kell, súlyokat eldobjuk
    ds_vis = ds.map(lambda img, labels, *rest: (img, labels))

    ds_vis = ds_vis.unbatch().shuffle(1000)
    samples = list(ds_vis.take(n_examples))

    imgs = [x[0] for x in samples]   # RAW képek (0–255)
    lbls = [x[1] for x in samples]

    # Modell bemenet előkészítése
    x_raw = tf.stack([tf.cast(img, tf.float32) for img in imgs], axis=0)
    x_for_model = preprocess_input(x_raw)

    pred_species_logits, pred_venom_prob = model.predict(x_for_model, verbose=0)

    plt.figure(figsize=(3.3 * len(imgs), 3.3))
    for i, (img, lbl) in enumerate(zip(imgs, lbls), start=1):
        # lbl: (species, venom)
        species_lbl, venom_lbl = lbl
        true_species = int(species_lbl.numpy())
        true_venom   = int(venom_lbl.numpy())

        pred_species = int(np.argmax(pred_species_logits[i-1]))
        pred_venom   = bool(float(pred_venom_prob[i-1][0]) > venom_threshold)

        true_name = species_names[true_species] if 0 <= true_species < len(species_names) else str(true_species)
        pred_name = species_names[pred_species] if 0 <= pred_species < len(species_names) else str(pred_species)

        plt.subplot(1, len(imgs), i)

        img_np = np.clip(img.numpy(), 0, 255).astype(np.uint8)
        plt.imshow(img_np)
        plt.axis("off")

        plt.title(
            f"True: {true_name} ({'Venom' if true_venom else 'Safe'})\n"
            f"Pred: {pred_name} ({'Venom' if pred_venom else 'Safe'})",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()


# In[30]:


example_results_from_dataset(model, val_dataset, species_metadata, n_examples=5)


# # Calculating scoring metrics

# Function to tell if the species is venomous or not, based on encoded_id

# In[37]:




# # Plotting mistakes

# In[38]:


from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

def plot_per_class_recall_f1(results):
    y_true = results["y_species_true"]
    y_pred = results["y_species_pred"]

    report = classification_report(y_true, y_pred, output_dict=True)
    class_ids = sorted([int(c) for c in report.keys() if c.isdigit()])

    recalls = [report[str(c)]["recall"] for c in class_ids]
    f1s     = [report[str(c)]["f1-score"] for c in class_ids]

    plt.figure(figsize=(20,6))
    plt.bar(class_ids, recalls)
    plt.title("Per-Class Recall")
    plt.xlabel("Class ID")
    plt.ylabel("Recall")
    plt.show()

    plt.figure(figsize=(20,6))
    plt.bar(class_ids, f1s)
    plt.title("Per-Class F1-Score")
    plt.xlabel("Class ID")
    plt.ylabel("F1")
    plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(results):
    y_true = results["y_species_true"]
    y_pred = results["y_species_pred"]

    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, cmap='Blues')
    plt.title("Normalized Confusion Matrix")
    plt.show()



# In[40]:


plot_per_class_recall_f1(results_own_metrics)


# In[41]:


plot_confusion_matrix(results_own_metrics)

