import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, roc_auc_score, confusion_matrix, log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Percorsi dei nuovi dataset
NEW_DATASETS = [
    "/kaggle/input/xm2vts-lips-frames-of/xm2vts-lips-frames-of",
    "/kaggle/input/m2fred-lips-frames-of/m2fred-lips-frames-of"
]

# Parametri
SEQUENCE_LENGTH = 60
IMG_SIZE = (64, 64)
LOG_DIR = "/kaggle/working/logs_new_datasets/"

# Creazione directory di log, se non esiste
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Funzione per caricare i percorsi delle sequenze per i nuovi dataset
def load_real_file_paths(base_paths):
    print("Caricamento dei percorsi delle sequenze dai nuovi dataset...")
    file_paths = []
    for base_path in base_paths:
        for video_folder in os.listdir(base_path):
            video_path = os.path.join(base_path, video_folder)
            frames = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg')])
            if len(frames) >= SEQUENCE_LENGTH:
                file_paths.append(frames[:SEQUENCE_LENGTH])  # Prende solo SEQUENCE_LENGTH frame
            else:
                print(f"Salto video {video_folder}: Previsti almeno {SEQUENCE_LENGTH} frame, trovati {len(frames)} frame.")
    print(f"Totale sequenze caricate: {len(file_paths)}")
    return file_paths

file_paths = load_real_file_paths(NEW_DATASETS)

# Funzioni di preelaborazione
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def preprocess_sequence(file_paths):
    sequence = tf.map_fn(preprocess_image, file_paths, dtype=tf.float32)
    return sequence

def create_dataset(file_paths, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(preprocess_sequence, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Creazione del dataset per i nuovi dati
BATCH_SIZE = 4
real_dataset = create_dataset(file_paths, BATCH_SIZE)

# Caricamento del modello salvato
MODEL_PATH = "/kaggle/working/model/optimized_cnn_lstm_model.h5"
print("Caricamento del modello salvato...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modello caricato.")

# Funzione per ottenere le previsioni
def get_predictions(dataset):
    predictions = []
    for batch_x in dataset:
        batch_preds = model.predict(batch_x, verbose=0)
        predictions.extend(batch_preds.flatten())
    return np.array(predictions)

# Ottenere le previsioni
print("Calcolo delle previsioni...")
pred_probs = get_predictions(real_dataset)
pred_classes = (pred_probs > 0.5).astype(int)

# Valutazione
true_labels = np.zeros(len(pred_classes))  # Tutti i video sono real (etichetta 0)
conf_matrix = confusion_matrix(true_labels, pred_classes)
metrics = {
    "accuracy": float(accuracy_score(true_labels, pred_classes)),
    "precision": float(precision_score(true_labels, pred_classes, zero_division=1)),
    "recall": float(recall_score(true_labels, pred_classes, zero_division=1)),
    "f1_score": float(f1_score(true_labels, pred_classes, zero_division=1)),
    "mse": float(mean_squared_error(true_labels, pred_classes)),
    "roc_auc": float(roc_auc_score(true_labels, pred_probs)),
    "log_loss": float(log_loss(true_labels, pred_probs)),
    "confusion_matrix": conf_matrix.tolist(),
    "true_positive": int(conf_matrix[1][1]),
    "true_negative": int(conf_matrix[0][0]),
    "false_positive": int(conf_matrix[0][1]),
    "false_negative": int(conf_matrix[1][0])
}

# Stampa delle metriche
print("Metriche calcolate:", metrics)

# Salvataggio delle metriche
print("Salvataggio delle metriche...")
with open(os.path.join(LOG_DIR, "metrics_new_datasets.json"), "w") as metrics_file:
    json.dump(metrics, metrics_file, indent=4)
print("Metriche salvate.")

# Plot della matrice di confusione
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.savefig(os.path.join(LOG_DIR, "confusion_matrix_new_datasets.png"))
plt.close()

print("Test completato e risultati salvati.")
