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
import json
from datetime import datetime

# Enable mixed precision for memory optimization
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Paths and parameters
BASE_PATH = "/kaggle/input/lips-frames-of/lips_frames/"
BATCH_SIZE = 4  # Reduced batch size
IMG_SIZE = (64, 64)  # Reduced image size
SEQUENCE_LENGTH = 60
EPOCHS = 20
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.5
LOG_DIR = "/kaggle/working/logs/"
MODEL_DIR = "/kaggle/working/model/"

print("Setting up directories...")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
print("Directories created.")

# Helper function to load file paths and labels
def load_file_paths_and_labels(base_path):
    print("Loading file paths and labels...")
    classes = {'real': 0, 'fake': 1}
    file_paths = []
    labels = []
    for class_name, label in classes.items():
        print(f"Processing class: {class_name} (label: {label})")
        class_path = os.path.join(base_path, class_name)
        for video_folder in os.listdir(class_path):
            video_path = os.path.join(class_path, video_folder)
            frames = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg')])
            if len(frames) >= SEQUENCE_LENGTH:
                file_paths.append(frames[:SEQUENCE_LENGTH])  # Take only SEQUENCE_LENGTH frames
                labels.append(label)
            else:
                print(f"Skipping video {video_folder}: Expected at least {SEQUENCE_LENGTH} frames, found {len(frames)} frames.")
    print(f"Total sequences loaded: {len(file_paths)}")
    return file_paths, labels

file_paths, labels = load_file_paths_and_labels(BASE_PATH)

# Split into train and validation sets
print("Splitting dataset into train and validation sets...")
train_files, val_files, train_labels, val_labels = train_test_split(
    file_paths, labels, test_size=0.3, stratify=labels, random_state=42
)
print(f"Train set: {len(train_files)} sequences, Validation set: {len(val_files)} sequences.")

# Convert labels to numpy arrays
train_labels = np.array(train_labels, dtype=np.int32)
val_labels = np.array(val_labels, dtype=np.int32)

# Data loading and preprocessing
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
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(preprocess_sequence, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_files, train_labels, BATCH_SIZE)
val_dataset = create_dataset(val_files, val_labels, BATCH_SIZE)

# Model definition
def build_cnn_lstm_model():
    print("Building CNN-LSTM model...")
    input_shape = (SEQUENCE_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3)
    model = Sequential([
        Input(shape=input_shape),
        TimeDistributed(Conv2D(16, (3, 3), activation='relu')),  # Reduced filters
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Conv2D(32, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        LSTM(32, return_sequences=False),  # Reduced LSTM units
        Dense(32, activation='relu'),  # Reduced dense layer neurons
        Dropout(DROPOUT_RATE),
        Dense(1, activation='sigmoid', dtype='float32')
    ])
    print("Model built.")
    return model

model = build_cnn_lstm_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(), 
    metrics=['accuracy']
)
model.summary()

# Training
print("Starting training...")
class TrainingLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        log = f"Epoch {epoch+1}, Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}"
        print(log)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

start_time = datetime.now()
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[TrainingLogger(), early_stopping]
)
end_time = datetime.now()
print(f"Training completed in: {end_time - start_time}")

# Save model
print("Saving model...")
model.save(os.path.join(MODEL_DIR, "memory_optimized_cnn_lstm_model.h5"))
print("Model saved.")

# Evaluation
print("Evaluating model...")
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

# Calculate metrics
conf_matrix = confusion_matrix(true_labels, pred_classes)
metrics = {
    "accuracy": float(accuracy_score(true_labels, pred_classes)),
    "precision": float(precision_score(true_labels, pred_classes)),
    "recall": float(recall_score(true_labels, pred_classes)),
    "f1_score": float(f1_score(true_labels, pred_classes)),
    "mse": float(mean_squared_error(true_labels, pred_classes)),
    "roc_auc": float(roc_auc_score(true_labels, pred_probs)),
    "log_loss": float(log_loss(true_labels, pred_probs)),
}

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
roc_auc = auc(fpr, tpr)

# Plot and save ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random Guess")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
plt.savefig(os.path.join(LOG_DIR, "roc_curve_beautiful.png"))
plt.close()

# Calculate Precision-Recall curve
precision, recall, pr_thresholds = precision_recall_curve(true_labels, pred_probs)

# Plot and save Precision-Recall curve
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=16)
plt.legend(loc="lower left", fontsize=12)
plt.grid(alpha=0.3)
plt.savefig(os.path.join(LOG_DIR, "precision_recall_curve_beautiful.png"))
plt.close()

# Plot and save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.savefig(os.path.join(LOG_DIR, "confusion_matrix_beautiful.png"))
plt.close()

# Plot training vs validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue', lw=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', lw=2)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training vs Validation Loss', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.savefig(os.path.join(LOG_DIR, "training_vs_validation_loss_beautiful.png"))
plt.close()

# Plot training vs validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', lw=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', lw=2)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Training vs Validation Accuracy', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.savefig(os.path.join(LOG_DIR, "training_vs_validation_accuracy_beautiful.png"))
plt.close()

print("All graphs have been saved beautifully.")
