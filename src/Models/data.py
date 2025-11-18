import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

#loads the metadata (train_images_metadata.csv , venomous_status_metadata.csv)
def load_metadata(
    base_path: str  = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
    species_csv_path: str = os.path.join("Data", "train_images_metadata.csv"),
    venom_csv_path: str = os.path.join("Data", "venomous_status_metadata.csv"),
    train_image_path: str = os.path.join("Data", "train_images_small")
):

    #read the csv files
    species_csv = pd.read_csv(os.path.join(base_path, species_csv_path), index_col=0)
    venom_csv = pd.read_csv(os.path.join(base_path, venom_csv_path), index_col=0)


    #merging the 2 files, now every row from species contains a column with venomous status
    species_venom_merged_csv = species_csv.merge(venom_csv[["class_id", "MIVS"]], on="class_id", how="left")
    image_metadata = species_venom_merged_csv[["binomial_name", "image_path", "class_id", "MIVS"]]

    #encoded_id is same as class_id, but starts from 0
    encoder = LabelEncoder()
    image_metadata["encoded_id"] = encoder.fit_transform(image_metadata["class_id"])

    # Find all of the image_paths 
    image_metadata["image_path"] = image_metadata["image_path"].apply(
        lambda img_path: os.path.join(base_path, train_image_path, img_path)
    )

    NUM_CLASSES = len(encoder.classes_)
    print("Number of classes: {}".format(NUM_CLASSES))

    # get one name per encoded_id, sorted by id so indices line up
    species_metadata = (
        image_metadata[['encoded_id', 'class_id', 'binomial_name', 'MIVS']]
        .drop_duplicates(subset=['encoded_id'])
        .sort_values('encoded_id')['binomial_name']
        .tolist()
    )

    return image_metadata, species_metadata

#-----------------------------------------------

def visualize_data(image_metadata):
    visualize_species(image_metadata)
    visualize_venom(image_metadata)

def visualize_species(image_metadata):
    # Count number of images per species
    class_counts = image_metadata['encoded_id'].value_counts().sort_index()
    sorted_counts = class_counts.sort_values(ascending=False)

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    #  Left: original order
    class_counts.plot(kind='bar', ax=axes[0])
    axes[0].set_title("Number of images per species (unsorted)")
    axes[0].set_ylabel("Image count")
    axes[0].set_xticks([])  # remove labels if they cause crowding

    # Right: sorted descending
    sorted_counts.plot(kind='bar', ax=axes[1], color='black')
    axes[1].set_title("Number of images per species (sorted)")
    axes[1].set_ylabel("Image count")
    axes[1].set_xticks([])

    plt.tight_layout()
    plt.show()

    print(f"Maximum number per species class is {max(class_counts)}")
    print(f"Minimum number per species class is {min(class_counts)}")

def visualize_venom(image_metadata):
    venom_counts = image_metadata['MIVS'].value_counts()
    plt.figure(figsize=(4,4))
    venom_counts.plot(kind='bar', color=['green', 'red'])
    plt.title("Non-venomous vs Venomous images")
    plt.ylabel("Count")
    plt.show()

    print(venom_counts)


#-----------------------------------------------
def make_dataset(image_metadata, image_resolution=224):
    train_info, val_info, test_info = split_dataset(image_metadata)

    train_dataset = make_batches(train_info, image_resolution, shuffle=True)
    val_dataset   = make_batches(val_info, image_resolution, shuffle=False)
    test_dataset  = make_batches(test_info, image_resolution, shuffle=False)

    return train_dataset, val_dataset, test_dataset


def load_img(path, image_resolution):
    img = tf.io.read_file(path)

    #expand_animations = False needed, otherwise gif format isnt proper
    img = tf.image.decode_image(img, channels=3, expand_animations = False)
    img = tf.image.resize(img, [image_resolution, image_resolution])
    img = preprocess_input(img) 
    return img



def make_labels(image_metadata):
    # Labels are going to be either venomous, or non-venomous
    species_labels = image_metadata['encoded_id']
    venom_labels = image_metadata['MIVS']
    return species_labels, venom_labels

def make_batches(info_df, image_resolution, batch_size=32, shuffle=True):
    AUTOTUNE = tf.data.AUTOTUNE
    image_paths  = info_df["image_path"].values
    sp     = info_df["encoded_id"].values.astype(np.int32)
    ve     = info_df["MIVS"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, sp, ve))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(info_df), 10_000),
                        seed=42, reshuffle_each_iteration=True)

    def _load(path, species, venom):
        img = load_img(path, image_resolution)  # [0,1] float32
        labels = {"species": species, "venom": venom}
        return img, labels

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

def split_dataset(image_metadata):
    train_info = image_metadata.copy()
    #1. split train: 80% train, 20% validation
    train_info, temp_info = train_test_split(
        image_metadata, test_size=0.2, random_state=42, stratify=image_metadata["encoded_id"]
    )

    # 2: split validation: 10% validation, 10% test
    val_info, test_info = train_test_split(
        temp_info, test_size=0.5, random_state=42, stratify=temp_info["encoded_id"]
    )
    return train_info, val_info, test_info
