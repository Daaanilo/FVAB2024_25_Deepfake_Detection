import cv2
import os
import numpy as np

# Impostazioni
num_frames = 60  # Numero di frame da estrarre per ogni video
dataset_path = os.path.expanduser('~/Desktop')  # Percorso base per le cartelle di input
output_path = os.path.expanduser('~/Desktop/lips_frames')  # Cartella di output
flow_threshold = 1.0  # Soglia per il flusso ottico

# Funzione per calcolare l'intensità del flusso ottico
def calculate_motion_intensity(prev_frame, next_frame):
    flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(magnitude)  # Media delle intensità del movimento

# Funzione per estrarre i frame più significativi basati sul flusso ottico
def extract_significant_frames(video_path, output_folder, num_frames, flow_threshold):
    print(f"[INFO] Inizio estrazione frame da: {video_path}")
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 2:
        print(f"[ERROR] Il video {video_path} non contiene frame sufficienti per calcolare il flusso ottico.")
        cap.release()
        return

    motion_scores = []
    frames = []

    ret, prev_frame = cap.read()
    if not ret:
        print(f"[ERROR] Impossibile leggere il primo frame da {video_path}.")
        cap.release()
        return

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    for i in range(1, total_frames):
        ret, next_frame = cap.read()
        if not ret:
            print(f"[WARNING] Frame {i} non disponibile. Interruzione.")
            break

        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        motion_intensity = calculate_motion_intensity(prev_frame_gray, next_frame_gray)
        
        # Salva il frame solo se supera la soglia del flusso ottico
        if motion_intensity > flow_threshold:
            motion_scores.append((motion_intensity, i, next_frame.copy()))  # Salva intensità, indice e frame
        
        prev_frame_gray = next_frame_gray

    cap.release()

    # Ordina i frame per intensità di movimento decrescente
    motion_scores.sort(reverse=True, key=lambda x: x[0])
    selected_frames = motion_scores[:num_frames]

    # Salva i frame selezionati
    for rank, (motion, frame_index, frame) in enumerate(selected_frames, start=1):
        frame_filename = os.path.join(output_folder, f'frame_{rank:03d}.jpg')
        cv2.imwrite(frame_filename, frame)
        print(f"[INFO] Salvato frame {rank} (indice: {frame_index}) in: {frame_filename}")

    print(f"[INFO] Completata estrazione frame significativi per: {video_path}")

# Percorsi delle cartelle principali
main_folders = ['celeb_real', 'celeb_fake']  # Cartelle di input

# Loop attraverso le cartelle principali
for folder in main_folders:
    folder_path = os.path.join(dataset_path, folder)
    print(f"[INFO] Elaborazione cartella: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"[WARNING] La cartella {folder_path} non esiste. Saltata.")
        continue
    for video_name in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_name)
        output_subfolder = 'real' if folder == 'celeb_real' else 'fake'  # Suddivisione in real e fake
        output_folder = os.path.join(output_path, output_subfolder, os.path.splitext(video_name)[0])
        print(f"[INFO] Elaborazione video: {video_name}")
        extract_significant_frames(video_path, output_folder, num_frames, flow_threshold)

print("[INFO] Estrazione dei frame completata!")
