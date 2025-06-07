import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, roc_auc_score, confusion_matrix, log_loss, 
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Abilita la precisione mista per ottimizzare la memoria
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Imposta i semi random per la riproducibilità
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Percorsi e parametri
BASE_PATH = "/kaggle/input/lips-frames-of/lips_frames/"
BATCH_SIZE = 4  # Batch size ridotto
IMG_SIZE = (64, 64)
SEQUENCE_LENGTH = 60
EPOCHS = 20
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.5
LOG_DIR = "/kaggle/working/logs/"
MODEL_DIR = "/kaggle/working/model/"

print("Impostazione delle directory...")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
print("Directory create.")

# Funzione di supporto per caricare i percorsi dei file e le etichette
def load_file_paths_and_labels(base_path):
    print("Caricamento dei percorsi dei file e delle etichette...")
    classes = {'real': 0, 'fake': 1}
    file_paths = []
    labels = []
    for class_name, label in classes.items():
        print(f"Elaborazione classe: {class_name} (etichetta: {label})")
        class_path = os.path.join(base_path, class_name)
        for video_folder in os.listdir(class_path):
            video_path = os.path.join(class_path, video_folder)
            frames = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg')])
            if len(frames) >= SEQUENCE_LENGTH:
                file_paths.append(frames[:SEQUENCE_LENGTH])  # Prendi solo SEQUENCE_LENGTH frame
                labels.append(label)
            else:
                print(f"Salto video {video_folder}: Attesi almeno {SEQUENCE_LENGTH} frame, trovati {len(frames)} frame.")
    print(f"Totale sequenze caricate: {len(file_paths)}")
    return file_paths, labels

file_paths, labels = load_file_paths_and_labels(BASE_PATH)

# Suddividi in set di addestramento e validazione
print("Suddivisione del dataset in set di addestramento e validazione...")
train_files, val_files, train_labels, val_labels = train_test_split(
    file_paths, labels, test_size=0.3, stratify=labels, random_state=42
)
print(f"Set di addestramento: {len(train_files)} sequenze, Set di validazione: {len(val_files)} sequenze.")

# Conta il numero di video "real" e "fake" nel dataset (commentato)
# train_real_count = sum(1 for label in train_labels if label == 0)
# train_fake_count = sum(1 for label in train_labels if label == 1)
# val_real_count = sum(1 for label in val_labels if label == 0)
# val_fake_count = sum(1 for label in val_labels if label == 1)
# print(f"Set di addestramento: {len(train_files)} sequenze (Reali: {train_real_count}, Fake: {train_fake_count})")
# print(f"Set di validazione: {len(val_files)} sequenze (Reali: {val_real_count}, Fake: {val_fake_count})")

# Converti le etichette in array numpy
train_labels = np.array(train_labels, dtype=np.int32)
val_labels = np.array(val_labels, dtype=np.int32)

# Caricamento e pre-processamento dei dati
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def preprocess_sequence(file_paths, label):
    sequence = tf.map_fn(preprocess_image, file_paths, dtype=tf.float32)
    return sequence, label

def create_dataset(file_paths, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(preprocess_sequence, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_files, train_labels, BATCH_SIZE)
val_dataset = create_dataset(val_files, val_labels, BATCH_SIZE)

# Definizione del modello
def build_optimized_cnn_lstm_model():
    print("Costruzione del modello ottimizzato CNN-LSTM...")
    input_shape = (SEQUENCE_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3)

    model = Sequential([
        Input(shape=input_shape),
        TimeDistributed(Conv2D(32, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        LSTM(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(0.5),
        Dense(1, activation='sigmoid', dtype='float32')
    ])
    print("Modello ottimizzato costruito.")
    return model

model = build_optimized_cnn_lstm_model()
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(), 
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

# Addestramento
print("Inizio addestramento...")
class TrainingLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        log = f"Epoca {epoch+1}, Loss: {logs['loss']:.4f}, Accuratezza: {logs['accuracy']:.4f}, Val Loss: {logs['val_loss']:.4f}, Val Accuratezza: {logs['val_accuracy']:.4f}"
        print(log)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
)

start_time = datetime.now()
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[TrainingLogger(), early_stopping, lr_schedule]
)
end_time = datetime.now()
print(f"Addestramento completato in: {end_time - start_time}")

# Salva il modello
print("Salvataggio del modello...")
model.save(os.path.join(MODEL_DIR, "optimized_cnn_lstm_model.keras"))
print("Modello salvato.")

# Valutazione
print("Valutazione del modello...")
def get_all_predictions(dataset):
    predictions = []
    true_labels = []
    for batch_x, batch_y in dataset:
        batch_preds = model.predict(batch_x, verbose=0)
        predictions.extend(batch_preds.flatten())
        true_labels.extend(batch_y.numpy())
    return np.array(predictions), np.array(true_labels)

pred_probs, true_labels = get_all_predictions(val_dataset)
pred_classes = (pred_probs > 0.5).astype(int)

# Calcola le metriche
conf_matrix = confusion_matrix(true_labels, pred_classes)
metrics = {
    "accuracy": float(accuracy_score(true_labels, pred_classes)),
    "precision": float(precision_score(true_labels, pred_classes)),
    "recall": float(recall_score(true_labels, pred_classes)),
    "f1_score": float(f1_score(true_labels, pred_classes)),
    "mse": float(mean_squared_error(true_labels, pred_classes)),
    "roc_auc": float(roc_auc_score(true_labels, pred_probs)),
    "log_loss": float(log_loss(true_labels, pred_probs)),
    "confusion_matrix": conf_matrix.tolist(),
    "true_positive": int(conf_matrix[1][1]),
    "true_negative": int(conf_matrix[0][0]),
    "false_positive": int(conf_matrix[0][1]),
    "false_negative": int(conf_matrix[1][0])
}
 
print("Metriche calcolate:", metrics)
 
# Salva le metriche
print("Salvataggio delle metriche...")
with open(os.path.join(LOG_DIR, "memory_optimized_metrics.json"), "w") as metrics_file:
    json.dump(metrics, metrics_file, indent=4)
print("Metriche salvate.")

# Calcola curva ROC e AUC
fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
roc_auc = auc(fpr, tpr)

# Plotta e salva la curva ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random Guess")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasso di Falsi Positivi', fontsize=12)
plt.ylabel('Tasso di Veri Positivi', fontsize=12)
plt.title('Curva ROC', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
plt.savefig(os.path.join(LOG_DIR, "roc_curve.png"))
plt.close()

# Calcola la curva Precision-Recall
precision, recall, pr_thresholds = precision_recall_curve(true_labels, pred_probs)

# Plotta e salva la curva Precision-Recall
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Curva Precision-Recall')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precisione', fontsize=12)
plt.title('Curva Precision-Recall', fontsize=16)
plt.legend(loc="lower left", fontsize=12)
plt.grid(alpha=0.3)
plt.savefig(os.path.join(LOG_DIR, "precision_recall_curve.png"))
plt.close()

# Plotta e salva la matrice di confusione
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Matrice di Confusione', fontsize=16)
plt.xlabel('Etichetta Predetta', fontsize=14)
plt.ylabel('Etichetta Vera', fontsize=14)
plt.savefig(os.path.join(LOG_DIR, "confusion_matrix.png"))
plt.close()

# Plotta perdita (loss) training vs validation
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Loss Training', color='blue', lw=2)
plt.plot(history.history['val_loss'], label='Loss Validazione', color='orange', lw=2)
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss Training vs Validazione', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.savefig(os.path.join(LOG_DIR, "training_vs_validation_loss.png"))
plt.close()

# Plotta accuratezza training vs validation
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Accuratezza Training', color='blue', lw=2)
plt.plot(history.history['val_accuracy'], label='Accuratezza Validazione', color='orange', lw=2)
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('Accuratezza', fontsize=12)
plt.title('Accuratezza Training vs Validazione', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.savefig(os.path.join(LOG_DIR, "training_vs_validation_accuracy.png"))
plt.close()

print("Tutti i grafici sono stati salvati.")
