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
    "/kaggle/input/xm2vts-lips-frames-of",
    "/kaggle/input/m2fred-lips-frames-of"
]

# Parametri
SEQUENCE_LENGTH = 60
IMG_SIZE = (64, 64)
LOG_DIR = "/kaggle/working/logs_new_datasets/"

# Creazione directory di log, se non esiste
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Funzione per caricare i percorsi delle sequenze dai nuovi dataset (ricorsivamente)
def load_real_file_paths(base_paths):
    print("Caricamento dei percorsi delle sequenze dai nuovi dataset...")
    file_paths = []
    total_processed_folders = 0
    for base_path in base_paths:
        print(f"Analizzando dataset: {base_path}")
        for root, dirs, files in os.walk(base_path):  # Ricorsione nelle sottocartelle
            frames = sorted([os.path.join(root, f) for f in files if f.endswith('.jpg')])
            if len(frames) >= SEQUENCE_LENGTH:
                file_paths.append(frames[:SEQUENCE_LENGTH])  # Prende solo SEQUENCE_LENGTH frame
            total_processed_folders += 1
            if total_processed_folders % 100 == 0:
                print(f"  >> Caricate {total_processed_folders} cartelle finora...")
    print(f"Totale cartelle elaborate: {total_processed_folders}")
    print(f"Totale sequenze valide caricate: {len(file_paths)}")
    return file_paths

file_paths = load_real_file_paths(NEW_DATASETS)

# Funzioni di preelaborazione
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def preprocess_sequence(file_paths, label):
    # Preelabora una sequenza di immagini usando TensorFlow
    sequence = tf.map_fn(preprocess_image, file_paths, dtype=tf.float32)
    return sequence, label

def create_dataset(file_paths, batch_size):
    # Converti i percorsi delle immagini in tensori
    file_paths_tensor = tf.ragged.constant(file_paths, dtype=tf.string)
    labels = np.zeros(len(file_paths), dtype=np.int32)  # Tutte etichette 0 perché real
    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)

    # Crea il dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_paths_tensor, labels_tensor))
    dataset = dataset.map(preprocess_sequence, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

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
    total_batches = len(dataset)
    print(f"Calcolo delle previsioni... Totale batch: {total_batches}")
    for i, (batch_x, _) in enumerate(dataset):
        batch_preds = model.predict(batch_x, verbose=0)
        predictions.extend(batch_preds.flatten())
        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            print(f"  >> Batch {i + 1}/{total_batches} completato.")
    return np.array(predictions)

# Ottenere le previsioni
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
    "confusion_matrix": conf_matrix.tolist(),
    "true_positive": int(conf_matrix[1][1]),
    "true_negative": int(conf_matrix[0][0]),
    "false_positive": int(conf_matrix[0][1]),
    "false_negative": int(conf_matrix[1][0])
}

# Calcolo condizionato della metrica ROC AUC
if len(np.unique(true_labels)) > 1:  # Verifica se ci sono almeno due classi
    metrics["roc_auc"] = float(roc_auc_score(true_labels, pred_probs))
else:
    print("Avviso: ROC AUC non calcolabile con una sola classe nei dati reali.")
    metrics["roc_auc"] = None  # Oppure imposta un valore predefinito

# Calcolo condizionato della log loss
if len(np.unique(true_labels)) > 1:  # Verifica se ci sono almeno due classi
    metrics["log_loss"] = float(log_loss(true_labels, pred_probs))
else:
    print("Avviso: Log Loss non calcolabile con una sola classe nei dati reali.")
    metrics["log_loss"] = None  # Oppure imposta un valore predefinito

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
