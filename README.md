# 🎭 FVAB: Rilevamento di Video Deepfake tramite Dinamiche Facciali  
**Fondamenti di Visione Artificiale e Biometria - A.A. 2024/2025**  
**Danilo Gisolfi, Vincenzo Maiellaro, Tommaso Nardi**

## 🧠 Descrizione del Progetto / Project Description

**ITA 🇮🇹**  
Il progetto **FVAB** si concentra sul rilevamento di video deepfake analizzando le dinamiche facciali, in particolare i movimenti temporali del volto, difficili da replicare nei contenuti sintetici. L'approccio combina **CNN** (Convolutional Neural Networks) per l'estrazione delle caratteristiche spaziali e **LSTM** (Long Short-Term Memory) per catturare le dinamiche temporali. Il modello è addestrato sul dataset **Celeb-DFv2** e valutato tramite accuratezza, **F1-score** e **MSE**.

**ENG 🇬🇧**  
The **FVAB** project focuses on detecting deepfake videos by analyzing facial dynamics, specifically temporal facial movements that are difficult to replicate in synthetic content. The approach combines **CNNs** for spatial feature extraction and **LSTMs** for temporal sequence learning. The model is trained on the **Celeb-DFv2** dataset and evaluated using accuracy, **F1-score**, and **MSE**.

---

## 👨‍💻 Componenti del Team / Team Members

- Danilo Gisolfi  
- Vincenzo Maiellaro  
- Tommaso Nardi

---

## 🎯 Obiettivi del Progetto / Project Goals

**ITA 🇮🇹**
- Rilevare deepfake analizzando i movimenti facciali temporali.
- Sviluppare un modello CNN-LSTM robusto.
- Addestrare il modello con Celeb-DFv2.
- Valutare le prestazioni usando accuratezza, F1-score, MSE.

**ENG 🇬🇧**
- Detect deepfakes by analyzing temporal facial movements.
- Develop a robust CNN-LSTM model.
- Train the model with Celeb-DFv2.
- Evaluate performance with accuracy, F1-score, and MSE.

---

## 🛠️ Caratteristiche Principali / Key Features

| Funzionalità                          | Descrizione 🇮🇹                                                                  | Description 🇬🇧                                                                |
|---------------------------------------|----------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| 📊 Preprocessing dei Dati             | Estrazione dei punti chiave del volto e tratti facciali.                        | Extracts facial landmarks and key facial points.                               |
| 🧠 Modello CNN-LSTM                   | CNN per caratteristiche spaziali + LSTM per dinamiche temporali.               | CNN for spatial features + LSTM for temporal dynamics.                         |
| ⚙️ Addestramento del Modello         | Deep learning su video reali/sintetici.                                         | Deep learning on real/synthetic videos.                                       |
| 📏 Valutazione delle Prestazioni      | Accuratezza, F1-score, MSE.                                                     | Accuracy, F1-score, MSE.                                                      |

---

## 🗂️ Dataset Utilizzati / Datasets Used

- **📦 Celeb-DFv2**  
  Dataset avanzato per deepfake detection con video ad alta qualità.

- **📦 M2FRED / WAV2LIP**  
  Utilizzato per la sincronizzazione audio-labbra e l’analisi del movimento labiale sintetico.

- **📦 XM2VTS**  
  Dataset biometricamente bilanciato per dati facciali reali.

---

## 🚀 Avvio del Progetto / How to Run

Segui questi passaggi per eseguire il progetto in locale:

### 📦 1. Installazione delle dipendenze / Install dependencies

Assicurati di avere Python 3.8+ installato. Poi esegui:

pip install -r requirements.txt
python 1_frame_extraction_optical_flow.py
python 2_training_cnn_lstm_optimized.py
python 3_inference_cnn_lstm.py

## 🎓 Contesto / Context

Questo progetto è stato sviluppato come parte di un’attività accademica nel campo della Bioinformatica e Data Integration.  
This project was developed as part of an academic activity in the field of Bioinformatics and Data Integration.
