import cv2
import os

# Impostazioni
num_frames = 60  # Numero di frame da estrarre per ogni video
dataset_path = os.path.expanduser('~/Desktop/Celeb-DF-v2')
output_path = os.path.expanduser('~/Desktop/output_frames')

# Funzione per estrarre i frame
def extract_frames(video_path, output_folder, num_frames):
    print(f"[INFO] Inizio estrazione frame da: {video_path}")
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"[ERROR] Il video {video_path} non contiene frame.")
        return

    step = max(1, total_frames // num_frames)  # Calcola lo step per estrarre i frame uniformemente

    for i in range(num_frames):
        frame_index = i * step
        if frame_index >= total_frames:  # Evita di superare il numero totale di frame
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(output_folder, f'frame_{i+1:03d}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f"[INFO] Salvato frame {i+1} in: {frame_filename}")
        else:
            print(f"[WARNING] Frame {i+1} non disponibile.")
            break

    cap.release()
    print(f"[INFO] Completata estrazione frame per: {video_path}")

# Percorsi delle cartelle principali
main_folders = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']

# Loop attraverso le cartelle principali
for folder in main_folders:
    folder_path = os.path.join(dataset_path, folder)
    print(f"[INFO] Elaborazione cartella: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"[WARNING] La cartella {folder_path} non esiste. Saltata.")
        continue
    for video_name in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_name)
        output_folder = os.path.join(output_path, folder, os.path.splitext(video_name)[0])
        print(f"[INFO] Elaborazione video: {video_name}")
        extract_frames(video_path, output_folder, num_frames)

print("[INFO] Estrazione dei frame completata!")