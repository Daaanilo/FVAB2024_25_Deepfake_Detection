import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, roc_auc_score, confusion_matrix, log_loss, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import random

# Percorsi dei nuovi dataset
FAKE_DATASET = "/kaggle/input/wav2lip-lips-frames-of/Wav2Lip_of/fake"
REAL_DATASET = "/kaggle/input/m2fred-lips-frames-of/M2FRED_lips_frames_of"

# Parametri
SEQUENCE_LENGTH = 60
IMG_SIZE = (64, 64)
LOG_DIR = "/kaggle/working/logs_new_datasets/"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def load_file_paths(base_path, label, sequence_length=SEQUENCE_LENGTH):
    file_paths = []
    labels = []
    total_processed_folders = 0
    for root, dirs, files in os.walk(base_path):
        frames = sorted([os.path.join(root, f) for f in files if f.endswith('.jpg')])
        if len(frames) >= sequence_length:
            file_paths.append(frames[:sequence_length])
            labels.append(label)
        total_processed_folders += 1
        if total_processed_folders % 100 == 0:
            print(f"Caricate {total_processed_folders} cartelle finora...")
    print(f"Totale cartelle elaborate ({base_path}): {total_processed_folders}")
    print(f"Totale sequenze valide caricate ({base_path}): {len(file_paths)}")
    return file_paths, labels

print("Caricamento dei percorsi delle sequenze dai nuovi dataset...")
fake_file_paths, fake_labels = load_file_paths(FAKE_DATASET, label=1)
real_file_paths, real_labels = load_file_paths(REAL_DATASET, label=0)

# (Opzionale) Bilancia il dataset se molto sbilanciato!
min_len = min(len(fake_file_paths), len(real_file_paths))
if abs(len(fake_file_paths) - len(real_file_paths)) > 0:
    print(f"Bilanciamento dataset: prendo {min_len} fake e {min_len} real")
    fake_idxs = random.sample(range(len(fake_file_paths)), min_len)
    real_idxs = random.sample(range(len(real_file_paths)), min_len)
    fake_file_paths = [fake_file_paths[i] for i in fake_idxs]
    fake_labels = [fake_labels[i] for i in fake_idxs]
    real_file_paths = [real_file_paths[i] for i in real_idxs]
    real_labels = [real_labels[i] for i in real_idxs]

all_file_paths = fake_file_paths + real_file_paths
all_labels = fake_labels + real_labels

# Shuffle per evitare bias di ordine
combined = list(zip(all_file_paths, all_labels))
random.shuffle(combined)
all_file_paths, all_labels = zip(*combined)
all_file_paths, all_labels = list(all_file_paths), list(all_labels)

# Stampa distribuzione classi
from collections import Counter
print("Distribuzione delle etichette:", Counter(all_labels))

# Visualizza un batch di immagini e label per controllo
def plot_sample_sequences(file_paths, labels, num_samples=4):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(num_samples, SEQUENCE_LENGTH // 10, figsize=(15, 3*num_samples))
    for i in range(num_samples):
        for j in range(SEQUENCE_LENGTH // 10):
            img = tf.io.decode_jpeg(tf.io.read_file(file_paths[i][j*10]))
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
        axs[i, 0].set_ylabel(f"Label: {labels[i]}", fontsize=12)
    plt.show()

plot_sample_sequences(all_file_paths, all_labels, num_samples=2)

# Funzioni di preelaborazione
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def preprocess_sequence(file_paths, label):
    sequence = tf.map_fn(preprocess_image, file_paths, dtype=tf.float32)
    return sequence, label

def create_dataset(file_paths, labels, batch_size):
    file_paths_tensor = tf.ragged.constant(file_paths, dtype=tf.string)
    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((file_paths_tensor, labels_tensor))
    dataset = dataset.map(preprocess_sequence, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

BATCH_SIZE = 4
dataset = create_dataset(all_file_paths, all_labels, BATCH_SIZE)

# Caricamento del modello salvato
MODEL_PATH = "/kaggle/working/celebdf_model/model/best_model_fold1.keras"
print("Caricamento del modello salvato...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modello caricato.")

def get_predictions(dataset):
    predictions = []
    true_labels = []
    total_batches = len(dataset)
    print(f"Calcolo delle previsioni... Totale batch: {total_batches}")
    for i, (batch_x, batch_y) in enumerate(dataset):
        batch_preds = model.predict(batch_x, verbose=0)
        predictions.extend(batch_preds.flatten())
        true_labels.extend(batch_y.numpy())
    return np.array(predictions), np.array(true_labels)

# Ottenere le previsioni
pred_probs, true_labels = get_predictions(dataset)

# TROVA LA SOGLIA OTTIMALE
fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Soglia ottimale trovata tramite ROC: {optimal_threshold:.3f}")

pred_classes = (pred_probs > optimal_threshold).astype(int)

# Valutazione
conf_matrix = confusion_matrix(true_labels, pred_classes)
metrics = {
    "accuracy": float(accuracy_score(true_labels, pred_classes)),
    "precision": float(precision_score(true_labels, pred_classes, zero_division=1)),
    "recall": float(recall_score(true_labels, pred_classes, zero_division=1)),
    "f1_score": float(f1_score(true_labels, pred_classes, zero_division=1)),
    "mse": float(mean_squared_error(true_labels, pred_classes)),
    "confusion_matrix": conf_matrix.tolist(),
    "true_positive": int(conf_matrix[1][1]) if conf_matrix.shape == (2, 2) else None,
    "true_negative": int(conf_matrix[0][0]) if conf_matrix.shape[0] > 0 else None,
    "false_positive": int(conf_matrix[0][1]) if conf_matrix.shape == (2, 2) else None,
    "false_negative": int(conf_matrix[1][0]) if conf_matrix.shape == (2, 2) else None,
    "roc_auc": float(roc_auc_score(true_labels, pred_probs)),
    "log_loss": float(log_loss(true_labels, pred_probs)),
    "optimal_threshold": float(optimal_threshold)
}

print("Metriche calcolate:", metrics)

# Salvataggio delle metriche
print("Salvataggio delle metriche...")
with open(os.path.join(LOG_DIR, "metrics_new_datasets.json"), "w") as metrics_file:
    json.dump(metrics, metrics_file, indent=4)
print("Metriche salvate.")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.savefig(os.path.join(LOG_DIR, "confusion_matrix_new_datasets.png"))
plt.close()
print("Test completato e risultati salvati.")
