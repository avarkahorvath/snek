import numpy as np
from sklearn.metrics import accuracy_score, f1_score



# encoded_id -> venom (0/1)
def make_enc2venom(image_metadata):
    enc2venom = (
        image_metadata[["encoded_id", "MIVS"]]
        .drop_duplicates("encoded_id")
        .sort_values("encoded_id")["MIVS"]
        .to_numpy(dtype=int)
    )
    return enc2venom

def is_species_venomous(encoded_id: int) -> int:
    return int(enc2venom[encoded_id])




def venom_weighted_accuracy(y_pred_species, y_true_species, y_pred_venom_spec, y_true_venom_spec):
    # max possible loss for this sample (2 if safe, 5 if venomous)
    denom_weights = np.where(y_true_venom_spec == 1, 5.0, 2.0)


    # matrix, correct species pred is 1, incorrect is 0
    correct_mask = (y_pred_species == y_true_species)

    # initialize per-sample loss
    loss = np.zeros_like(denom_weights, dtype=float)

    # only compute loss for misclassified speciesq
    m = ~correct_mask #only incorrect species predictions
    tv = y_true_venom_spec[m]
    pv = y_pred_venom_spec[m]

    # apply table
    l = np.zeros_like(tv, dtype=float)
    l[(tv == 0) & (pv == 0)] = 1.0
    l[(tv == 0) & (pv == 1)] = 2.0
    l[(tv == 1) & (pv == 1)] = 2.0
    l[(tv == 1) & (pv == 0)] = 5.0

    loss[m] = l

    # final score
    denom = denom_weights.sum()
    return 1.0 - (loss.sum() / denom if denom else 0.0)


def get_scores(model, image_metadata, test_dataset, venom_threshold=0.5):

    # prediction from model
    species_pred_probs, venom_pred_probs = model.predict(test_dataset, verbose=1)

    # true information
    y_species_true_batches = []
    y_venom_true_batches = []

    enc2venom=make_enc2venom(image_metadata)

    for images, labels in test_dataset:
        y_species_true_batches.append(labels["species"].numpy())
        y_venom_true_batches.append(labels["venom"].numpy())

    y_species_true = np.concatenate(y_species_true_batches, axis=0)
    y_venom_true   = np.concatenate(y_venom_true_batches, axis=0)

    # hard predictions
    y_species_pred = np.argmax(species_pred_probs, axis=1)
    y_venom_pred   = (venom_pred_probs.reshape(-1) >= venom_threshold).astype(int)

    # 4) metrics
    species_accuracy = accuracy_score(y_species_true, y_species_pred)
    macro_f1         = f1_score(y_species_true, y_species_pred, average="macro")
    venom_accuracy   = accuracy_score(y_venom_true, y_venom_pred)


    weighted_species_accuracy = venom_weighted_accuracy(y_species_pred, y_species_true, enc2venom[y_species_pred], y_venom_true)

    print("=== Evaluation Metrics ===")
    print(f"1) Species accuracy: {species_accuracy:.4f}")
    print(f"2) Macro-averaged F1 (species): {macro_f1:.4f}")
    print(f"3) Venom decision accuracy: {venom_accuracy:.4f}")
    print(f"4) Venom-weighted species accuracy: {weighted_species_accuracy:.4f}")

    return {
        "species_accuracy": species_accuracy,
        "macro_f1": macro_f1,
        "venom_accuracy": venom_accuracy,
        "venom_weighted_species_accuracy": weighted_species_accuracy,
        "y_species_true": y_species_true,
        "y_species_pred": y_species_pred,
        "y_venom_true": y_venom_true,
        "y_venom_pred": y_venom_pred,
    }


