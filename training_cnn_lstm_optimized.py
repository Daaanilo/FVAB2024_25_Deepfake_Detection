import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Dropout, Input
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, roc_auc_score, confusion_matrix, log_loss
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
BASE_PATH = "/kaggle/input/celeb-df-v2-lips/output_lips/"
BATCH_SIZE = 2  # Keeping batch size small to stay under 13 GB
IMG_SIZE = (96, 96)  # Increased resolution for better feature extraction
SEQUENCE_LENGTH = 60  # Sequence length remains at 60
EPOCHS = 25
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
    classes = {'Celeb-real': 0, 'Celeb-synthesis': 1, 'YouTube-real': 0}
    file_paths = []
    labels = []
    for class_name, label in classes.items():
        print(f"Processing class: {class_name} (label: {label})")
        class_path = os.path.join(base_path, class_name)
        video_count = 0
        for video_folder in os.listdir(class_path):
            video_path = os.path.join(class_path, video_folder)
            frames = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg')])
            if len(frames) >= SEQUENCE_LENGTH:
                file_paths.append(frames[:SEQUENCE_LENGTH])  # Take only SEQUENCE_LENGTH frames
                labels.append(label)
                video_count += 1
            else:
                print(f"Skipping video {video_folder}: Expected at least {SEQUENCE_LENGTH} frames, found {len(frames)} frames.")
        print(f"Processed {video_count} videos for class '{class_name}'.")
    print(f"Total sequences loaded: {len(file_paths)}")
    return file_paths, labels

file_paths, labels = load_file_paths_and_labels(BASE_PATH)

# Split into train and validation sets
print("Splitting dataset into train and validation sets...")
train_split = int(0.8 * len(file_paths))
train_files, val_files = file_paths[:train_split], file_paths[train_split:]
train_labels, val_labels = labels[:train_split], labels[train_split:]
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
    # Use tf.map_fn to iterate over file paths
    sequence = tf.map_fn(preprocess_image, file_paths, dtype=tf.float32)
    return sequence, label

def create_dataset(file_paths, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(preprocess_sequence, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_files, train_labels, BATCH_SIZE)
val_dataset = create_dataset(val_files, val_labels, BATCH_SIZE)

# Model definition
def build_cnn_lstm_model():
    print("Building CNN-LSTM model...")
    input_shape = (SEQUENCE_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3)
    model = Sequential([
        Input(shape=input_shape),
        TimeDistributed(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1e-4))),  # Added L2 regularization
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(1e-4))),  # Increased filters
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        LSTM(128, return_sequences=False, kernel_regularizer=l2(1e-4)),  # Added L2 regularization
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),  # Added L2 regularization
        Dropout(0.5),
        Dense(1, activation='sigmoid', dtype='float32')  # Output in float32 due to mixed precision
    ])
    print("Model built.")
    return model

model = build_cnn_lstm_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),  # Reduced learning rate for better convergence
    loss=tf.keras.losses.BinaryCrossentropy(), 
    metrics=['accuracy']
)
model.summary()

# Training
print("Starting training...")
class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n--- Starting Epoch {epoch + 1}/{EPOCHS} ---")
    
    def on_epoch_end(self, epoch, logs=None):
        log = (f"Epoch {epoch + 1}/{EPOCHS} Summary: "
               f"Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, "
               f"Validation Loss: {logs['val_loss']:.4f}, Validation Accuracy: {logs['val_accuracy']:.4f}")
        print(log)

# Early stopping to prevent overfitting
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
    callbacks=[EpochLogger(), early_stopping]
)
end_time = datetime.now()
print(f"Training completed in: {end_time - start_time}")

# Save model
print("Saving model...")
model.save(os.path.join(MODEL_DIR, "cnn_lstm_model.h5"))
print("Model saved.")