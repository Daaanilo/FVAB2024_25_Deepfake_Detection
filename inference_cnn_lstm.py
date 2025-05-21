import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_squared_error, roc_auc_score, confusion_matrix, log_loss,
                             roc_curve, precision_recall_curve)
import json

# ==== CONFIG ====
IMG_SIZE = (64, 64)
SEQUENCE_LENGTH = 60
BATCH_SIZE = 4
MODEL_PATH = "celebdf_model/model/best_model_global.keras"

# Paths to datasets (update if necessary)
REAL_DATASET_PATH = "/kaggle/input/m2fred-lips-frames-of/M2FRED_lips_frames_of"
FAKE_DATASET_PATH = "/kaggle/input/wav2lip-lips-frames-of/Wav2Lip_of/fake"

def load_file_paths_and_labels_singleclass(base_path, label, verbose=True):
    file_paths, labels = [], []
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    print(f"Found {len(folders)} folders in {base_path} (label={label})")
    for idx, video_folder in enumerate(folders):
        video_path = os.path.join(base_path, video_folder)
        frames = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg')])
        if len(frames) >= SEQUENCE_LENGTH:
            if len(frames) > SEQUENCE_LENGTH:
                start_idx = np.random.randint(0, len(frames) - SEQUENCE_LENGTH)
                frames = frames[start_idx:start_idx+SEQUENCE_LENGTH]
            file_paths.append(frames)
            labels.append(label)
        if verbose and (idx+1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(folders)} folders in {base_path}")
    print(f"Loaded {len(file_paths)} samples for label {label} from {base_path}\n")
    return file_paths, labels

def preprocess_image(image_path, label, training=False):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def preprocess_sequence(file_paths, label, training=False):
    sequence = tf.map_fn(lambda x: preprocess_image(x, label, training), file_paths, dtype=tf.float32)
    return sequence, label

def create_dataset(file_paths, labels, batch_size):
    print(f"Creating tf.data.Dataset with {len(file_paths)} samples...")
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(lambda x, y: preprocess_sequence(x, y, False), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print("tf.data.Dataset created!\n")
    return dataset

def get_all_predictions(model, dataset):
    predictions, true_labels = [], []
    batch_num = 0
    print("Starting model predictions on test set...")
    for batch_x, batch_y in dataset:
        batch_preds = model.predict(batch_x, verbose=0)
        predictions.extend(batch_preds.flatten())
        true_labels.extend(batch_y.numpy())
        batch_num += 1
        if batch_num % 20 == 0:
            print(f"  Processed {batch_num} batches")
    print("Model predictions completed!\n")
    return np.array(predictions), np.array(true_labels)

# ==== Load external dataset paths ====
print("Loading REAL samples...")
real_file_paths, real_labels = load_file_paths_and_labels_singleclass(REAL_DATASET_PATH, 0)
print("Loading FAKE samples...")
fake_file_paths, fake_labels = load_file_paths_and_labels_singleclass(FAKE_DATASET_PATH, 1)

test_file_paths = real_file_paths + fake_file_paths
test_labels = real_labels + fake_labels

print(f"Test set: {len(real_labels)} real, {len(fake_labels)} fake, total={len(test_labels)}\n")

test_labels = np.array(test_labels, dtype=np.int32)
test_dataset = create_dataset(test_file_paths, test_labels, BATCH_SIZE)

# ==== Load model ====
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded!\n")
# (If custom loss was used: compile with it, e.g., model.compile(..., loss=focal_loss()))

# ==== Evaluate on test set ====
print("Evaluating model on external test set...")
pred_probs, true_labels = get_all_predictions(model, test_dataset)
pred_classes = (pred_probs > 0.5).astype(int)
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
print("\nExternal Test Metrics:", metrics)
with open("external_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("\nMetrics saved to external_metrics.json")
