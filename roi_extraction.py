import cv2
import os
import mediapipe as mp

# Percorsi principali
input_frames_path = os.path.expanduser('~/Desktop/output_frames')
output_faces_path = os.path.expanduser('~/Desktop/output_faces')
output_lips_path = os.path.expanduser('~/Desktop/output_lips')

# Inizializzazione di Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

def save_image(image, path, roi_type):
    """Salva un'immagine e gestisce eventuali errori."""
    try:
        if image.size == 0:
            print(f"[WARNING] ROI {roi_type} vuota, immagine non salvata: {path}")
            return False
        cv2.imwrite(path, image)
        print(f"[INFO] Salvata ROI {roi_type} in: {path}")
        return True
    except Exception as e:
        print(f"[ERROR] Impossibile salvare ROI {roi_type} in {path}: {e}")
        return False

def extract_face(image, detection, iw, ih):
    """Estrae la ROI del volto dall'immagine."""
    bboxC = detection.location_data.relative_bounding_box
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
    return image[y:y+h, x:x+w]

def extract_lips(face_roi, face_landmarks, iw, ih):
    """Estrae la ROI delle labbra dalla ROI del volto."""
    lips_points = [61, 291, 0, 17, 13, 14, 78, 308]  # Indici dei landmark delle labbra
    x_coords = [int(face_landmarks.landmark[i].x * iw) for i in lips_points]
    y_coords = [int(face_landmarks.landmark[i].y * ih) for i in lips_points]
    x_min, x_max = max(0, min(x_coords)), min(iw, max(x_coords))
    y_min, y_max = max(0, min(y_coords)), min(ih, max(y_coords))
    return face_roi[y_min:y_max, x_min:x_max]

def extract_roi(image_path, output_face_folder, output_lips_folder):
    """Estrae le ROI di volto e labbra da un'immagine."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Immagine non trovata: {image_path}")
            return

        ih, iw, _ = image.shape
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection, \
             mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

            # Rilevamento volto
            try:
                results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not results.detections:
                    print(f"[WARNING] Nessun volto rilevato in: {image_path}")
                    return

                for detection in results.detections:
                    try:
                        face_roi = extract_face(image, detection, iw, ih)
                        face_filename = os.path.join(output_face_folder, os.path.basename(image_path).replace('frame', 'face'))
                        if not save_image(face_roi, face_filename, "volto"):
                            continue

                        # Rilevamento labbra
                        try:
                            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                            face_results = face_mesh.process(face_rgb)
                            if not face_results.multi_face_landmarks:
                                print(f"[WARNING] Nessun landmark rilevato per le labbra in: {image_path}")
                                continue

                            for face_landmarks in face_results.multi_face_landmarks:
                                lips_roi = extract_lips(face_roi, face_landmarks, face_roi.shape[1], face_roi.shape[0])
                                if lips_roi.size == 0:
                                    print(f"[WARNING] ROI labbra vuota per immagine: {image_path}")
                                    continue

                                lips_filename = os.path.join(output_lips_folder, os.path.basename(image_path).replace('frame', 'lips'))
                                if not save_image(lips_roi, lips_filename, "labbra"):
                                    continue
                                break  # Salva solo la prima rilevazione delle labbra
                        except Exception as e:
                            print(f"[ERROR] Errore durante il rilevamento o il salvataggio delle labbra in: {image_path} - {e}")
                    except Exception as e:
                        print(f"[ERROR] Errore durante il rilevamento o il salvataggio del volto in: {image_path} - {e}")
            except Exception as e:
                print(f"[ERROR] Errore durante il rilevamento del volto in: {image_path} - {e}")
    except Exception as e:
        print(f"[ERROR] Errore durante l'elaborazione di {image_path}: {e}")

def process_frames(input_folder, output_faces_folder, output_lips_folder):
    """Elabora tutti i frame in una cartella."""
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.jpg'):
                input_image_path = os.path.join(root, file)

                # Creazione delle cartelle di output
                relative_path = os.path.relpath(root, input_folder)
                face_output_folder = os.path.join(output_faces_folder, relative_path)
                lips_output_folder = os.path.join(output_lips_folder, relative_path)
                os.makedirs(face_output_folder, exist_ok=True)
                os.makedirs(lips_output_folder, exist_ok=True)

                # Estrazione delle ROI
                try:
                    extract_roi(input_image_path, face_output_folder, lips_output_folder)
                except Exception as e:
                    print(f"[ERROR] Errore durante l'elaborazione del file {input_image_path}: {e}")

# Avvio del processo
print("[INFO] Inizio estrazione delle ROI...")
process_frames(input_frames_path, output_faces_path, output_lips_path)
print("[INFO] Estrazione delle ROI completata!")