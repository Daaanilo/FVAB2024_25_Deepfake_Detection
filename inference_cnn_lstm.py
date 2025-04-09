import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Paths and parameters
MODEL_PATH = "/kaggle/working/model/cnn_lstm_model.h5"  # Path to the trained model
FRAME_PATH = "/path/to/new/lip/frames/"  # Path to new frames for inference
SEQUENCE_LENGTH = 60  # Number of frames per sequence
IMG_SIZE = (96, 96)  # Image size (should match training)
BATCH_SIZE = 1  # Batch size for inference

# Load the trained model
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Preprocessing function for a single image
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Function to load and preprocess a sequence of frames
def preprocess_sequence(frame_paths):
    sequence = [preprocess_image(fp) for fp in frame_paths]
    return np.array(sequence)

# Function to generate sequences from frames in a directory
def generate_sequences(frame_dir, sequence_length):
    all_sequences = []
    frame_folders = sorted(os.listdir(frame_dir))  # Each folder represents a video
    for folder in frame_folders:
        folder_path = os.path.join(frame_dir, folder)
        if os.path.isdir(folder_path):
            frames = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')])
            if len(frames) >= sequence_length:
                sequence = preprocess_sequence(frames[:sequence_length])  # Take only the first 'sequence_length' frames
                all_sequences.append(sequence)
            else:
                print(f"Skipping folder {folder}: Not enough frames (found {len(frames)}, required {sequence_length}).")
    return np.array(all_sequences)

# Load and preprocess new sequences
print("Preprocessing new sequences...")
sequences = generate_sequences(FRAME_PATH, SEQUENCE_LENGTH)
print(f"Total sequences prepared: {len(sequences)}")

# Predict using the model
print("Running inference...")
predictions = model.predict(sequences, batch_size=BATCH_SIZE)
predicted_classes = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions
print("Inference completed.")

# Display the results
for i, prediction in enumerate(predicted_classes):
    print(f"Sequence {i + 1}: {'Fake' if prediction == 1 else 'Real'} (Confidence: {predictions[i][0]:.4f})")