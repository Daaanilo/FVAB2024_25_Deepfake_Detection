import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, LSTM, Dense,
                                     TimeDistributed, Dropout, Input, BatchNormalization, GaussianNoise)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, mean_squared_error, roc_auc_score, confusion_matrix,
                             log_loss, roc_curve, auc, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ========== CONFIG ==========
BASE_PATH = "/kaggle/input/lips-frames-of/lips_frames/"
SAVE_DIR = "celebdf_model2/"
LOG_DIR = os.path.join(SAVE_DIR, "logs")
MODEL_DIR = os.path.join(SAVE_DIR, "model")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 4
IMG_SIZE = (64, 64)
SEQUENCE_LENGTH = 60
EPOCHS = 10
LEARNING_RATE = 2e-4

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ===== FOCAL LOSS =====
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_exp = tf.exp(-bce)
        focal_loss = alpha * (1 - bce_exp) ** gamma * bce
        return focal_loss
    return focal_loss_fixed

# ======= DATA LOADING & BALANCING =======
def load_file_paths_and_labels(base_path):
    classes = {'real': 0, 'fake': 1}
    file_paths, labels = [], []
    for class_name, label in classes.items():
        class_path = os.path.join(base_path, class_name)
        for video_folder in os.listdir(class_path):
            video_path = os.path.join(class_path, video_folder)
            frames = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg')])
            if len(frames) >= SEQUENCE_LENGTH:
                if len(frames) > SEQUENCE_LENGTH:
                    start_idx = random.randint(0, len(frames) - SEQUENCE_LENGTH)
                    frames = frames[start_idx:start_idx+SEQUENCE_LENGTH]
                file_paths.append(frames)
                labels.append(label)
    return file_paths, labels

def oversample_minority(file_paths, labels, target_ratio=0.4):
    from collections import Counter
    counter = Counter(labels)
    n_real, n_fake = counter[0], counter[1]
    total = n_real + n_fake
    desired_real = int(target_ratio * total)
    if n_real >= desired_real:
        return file_paths, labels
    real_samples = [(fp, lbl) for fp, lbl in zip(file_paths, labels) if lbl == 0]
    n_to_add = desired_real - n_real
    new_samples = [random.choice(real_samples) for _ in range(n_to_add)]
    new_paths, new_labels = zip(*new_samples)
    file_paths += list(new_paths)
    labels += list(new_labels)
    combined = list(zip(file_paths, labels))
    random.shuffle(combined)
    file_paths[:], labels[:] = zip(*combined)
    return list(file_paths), list(labels)

file_paths, labels = load_file_paths_and_labels(BASE_PATH)
file_paths, labels = oversample_minority(file_paths, labels, target_ratio=0.4)
print(f"Final for training: {sum(1 for l in labels if l==0)} real, {sum(1 for l in labels if l==1)} fake")

train_labels = np.array(labels, dtype=np.int32)
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print(f"Class weights: {class_weight_dict}")

# ========== DATASET SPLIT ==========
# Split into train and validation (stratified), keep 10% for validation
train_paths, val_paths, train_lbls, val_lbls = train_test_split(
    file_paths, train_labels, test_size=0.1, stratify=train_labels, random_state=42
)

# ========== DATA AUGMENTATION ==========
def augment_image(image, is_real):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, k=random.randint(0, 3))
    if is_real:
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_jpeg_quality(image, 60, 100)
        image = tf.image.stateless_random_crop(image, size=[IMG_SIZE[0], IMG_SIZE[1], 3],
                                               seed=(random.randint(0, 9999), random.randint(0, 9999)))
    else:
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def preprocess_image(image_path, label, training=True):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    if training:
        image = augment_image(image, tf.equal(label, 0))
    return image

def preprocess_sequence(file_paths, label, training=True):
    sequence = tf.map_fn(lambda x: preprocess_image(x, label, training), file_paths, dtype=tf.float32)
    return sequence, label

def create_dataset(file_paths, labels, batch_size, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=2000)
    dataset = dataset.map(lambda x, y: preprocess_sequence(x, y, training), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_paths, train_lbls, BATCH_SIZE, training=True)
val_dataset   = create_dataset(val_paths,   val_lbls,   BATCH_SIZE, training=False)

# ========== MODEL ==========
def build_cnn_lstm():
    input_shape = (SEQUENCE_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3)
    model = Sequential([
        Input(shape=input_shape),
        TimeDistributed(GaussianNoise(0.2)),  # aiuta la generalizzazione
        TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-2))),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Dropout(0.3)),
        TimeDistributed(GaussianNoise(0.1)),
        TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-2))),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Dropout(0.35)),
        TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-2))),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Dropout(0.45)),
        TimeDistributed(Flatten()),
        LSTM(32, return_sequences=False, dropout=0.6, recurrent_dropout=0.5),
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-2)),
        BatchNormalization(),
        Dropout(0.55),
        Dense(1, activation='sigmoid', dtype='float32')
    ])
    return model

model = build_cnn_lstm()
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-2),
    loss=focal_loss(),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
model_ckpt = ModelCheckpoint(os.path.join(MODEL_DIR, "best_model_global.keras"),
                            monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, lr_schedule, model_ckpt],
    class_weight=class_weight_dict,
    verbose=1
)
model.save(os.path.join(MODEL_DIR, "last_model_global.keras"))

# ========== METRICS & PLOTS (ON TRAINING DATA, for reference) ==========
def get_all_predictions(dataset):
    predictions, true_labels = [], []
    for batch_x, batch_y in dataset:
        batch_preds = model.predict(batch_x, verbose=0)
        predictions.extend(batch_preds.flatten())
        true_labels.extend(batch_y.numpy())
    return np.array(predictions), np.array(true_labels)

pred_probs, true_labels = get_all_predictions(train_dataset)
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
print("Final Training Metrics:", metrics)
with open(os.path.join(LOG_DIR, "metrics_global_trainset.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# Plot ROC curve
fpr, tpr, _ = roc_curve(true_labels, pred_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Train set)')
plt.legend()
plt.savefig(os.path.join(LOG_DIR, "roc_curve_trainset.png"))
plt.close()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Train set)')
plt.savefig(os.path.join(LOG_DIR, "precision_recall_curve_trainset.png"))
plt.close()

# Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix (Train set)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(LOG_DIR, "confusion_matrix_trainset.png"))
plt.close()

# Training curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue', lw=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', lw=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.savefig(os.path.join(LOG_DIR, "training_loss.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', lw=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', lw=2)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.savefig(os.path.join(LOG_DIR, "training_accuracy.png"))
plt.close()

print("Training complete, model and metrics saved. Use best_model_global.keras for external testing.")
