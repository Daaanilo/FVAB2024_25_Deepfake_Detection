<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Status-Research-blueviolet?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/License-Academic-green?style=for-the-badge" alt="License">
</p>

<h1 align="center">🎭 FVAB: Deepfake Video Detection</h1>

<p align="center">
  <strong>Rilevamento di Video Deepfake tramite Dinamiche Facciali</strong><br>
  <em>Detecting Deepfake Videos through Facial Dynamics Analysis</em>
</p>

<p align="center">
  <a href="#-descrizione-del-progetto-italiano">🇮🇹 Italiano</a> •
  <a href="#-project-description-english">🇬🇧 English</a> •
  <a href="#-quick-start">🚀 Quick Start</a> •
  <a href="#-architettura--architecture">🏗️ Architecture</a>
</p>

---

## 👨‍💻 Team Members / Componenti del Team

<table align="center">
  <tr>
    <td align="center">
      <strong>Danilo Gisolfi</strong>
    </td>
    <td align="center">
      <strong>Vincenzo Maiellaro</strong>
    </td>
    <td align="center">
      <strong>Tommaso Nardi</strong>
    </td>
  </tr>
</table>

<p align="center">
  <sub>Fondamenti di Visione Artificiale e Biometria • A.A. 2024/2025</sub>
</p>

---

## 📖 Descrizione del Progetto (Italiano)

Il progetto **FVAB** si concentra sul rilevamento di video deepfake analizzando le **dinamiche facciali**, in particolare i movimenti temporali del volto, difficili da replicare nei contenuti sintetici.

### 🧠 Approccio Tecnico

L'architettura combina due tecnologie di deep learning:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FVAB Detection Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📹 Video Input                                                  │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────┐                                             │
│  │  Frame Extraction │  ──► Optical Flow Computation            │
│  └────────┬────────┘                                             │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐    Spatial Features                        │
│  │      CNN        │ ──────────────────┐                        │
│  │  (Convolutional)│                   │                        │
│  └─────────────────┘                   ▼                        │
│                              ┌─────────────────┐                │
│                              │      LSTM       │                │
│                              │ (Temporal Seq.) │                │
│                              └────────┬────────┘                │
│                                       │                         │
│                                       ▼                         │
│                              ┌─────────────────┐                │
│                              │  Classification │                │
│                              │  REAL / FAKE    │                │
│                              └─────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

| Componente | Funzione |
|------------|----------|
| **🔍 CNN** | Estrazione caratteristiche spaziali da ogni frame |
| **🔄 LSTM** | Apprendimento delle dinamiche temporali |
| **📊 Classifier** | Classificazione binaria Real/Fake |

### 🎯 Obiettivi

- [x] Rilevare deepfake analizzando i movimenti facciali temporali
- [x] Sviluppare un modello CNN-LSTM robusto
- [x] Addestrare il modello con Celeb-DFv2
- [x] Valutare le prestazioni usando accuratezza, F1-score, MSE

---

## 📖 Project Description (English)

The **FVAB** project focuses on detecting deepfake videos by analyzing **facial dynamics**, specifically temporal facial movements that are difficult to replicate in synthetic content.

### 🧠 Technical Approach

The architecture combines two deep learning technologies:

| Component | Function |
|-----------|----------|
| **🔍 CNN** | Spatial feature extraction from each frame |
| **🔄 LSTM** | Learning temporal dynamics |
| **📊 Classifier** | Binary classification Real/Fake |

### 🎯 Goals

- [x] Detect deepfakes by analyzing temporal facial movements
- [x] Develop a robust CNN-LSTM model
- [x] Train the model with Celeb-DFv2
- [x] Evaluate performance with accuracy, F1-score, and MSE

---

## 🛠️ Key Features / Caratteristiche Principali

| Feature | Descrizione 🇮🇹 | Description 🇬🇧 |
|---------|-----------------|-----------------|
| 📊 **Data Preprocessing** | Estrazione dei punti chiave del volto e tratti facciali | Extracts facial landmarks and key facial points |
| 🧠 **CNN-LSTM Model** | CNN per caratteristiche spaziali + LSTM per dinamiche temporali | CNN for spatial features + LSTM for temporal dynamics |
| ⚙️ **Model Training** | Deep learning su video reali/sintetici | Deep learning on real/synthetic videos |
| 📏 **Performance Metrics** | Accuratezza, F1-score, MSE | Accuracy, F1-score, MSE |

---

## 🗂️ Datasets Used / Dataset Utilizzati

| Dataset | Descrizione | Purpose |
|---------|-------------|---------|
| 📦 **Celeb-DFv2** | Dataset avanzato per deepfake detection con video ad alta qualità | Advanced high-quality deepfake detection |
| 📦 **M2FRED / WAV2LIP** | Sincronizzazione audio-labbra e movimento labiale sintetico | Lip-sync and synthetic lip movement analysis |
| 📦 **XM2VTS** | Dataset biometricamente bilanciato per dati facciali reali | Biometrically balanced real facial data |

---

## 🚀 Quick Start

### ✅ Requisiti / Requirements

- Python 3.8+
- pip installed

### 📦 Installation

```bash
# Clone the repository / Clona la repository
git clone https://github.com/your-username/FVAB-deepfake-detection.git
cd FVAB-deepfake-detection

# Install dependencies / Installa le dipendenze
pip install -r requirements.txt
```

### 🧪 Execution / Esecuzione

```bash
# 1️⃣ Frame extraction & optical flow
python 1_frame_extraction_optical_flow.py

# 2️⃣ Train CNN-LSTM model
python 2_training_cnn_lstm_optimized.py

# 3️⃣ Run inference
python 3_inference_cnn_lstm.py
```

---

## 📊 Evaluation Metrics / Metriche di Valutazione

| Metric | Description 🇬🇧 | Descrizione 🇮🇹 |
|--------|-----------------|-----------------|
| **Accuracy** | Overall classification correctness | Correttezza complessiva della classificazione |
| **F1-Score** | Harmonic mean of precision and recall | Media armonica di precisione e recall |
| **MSE** | Mean Squared Error for regression tasks | Errore quadratico medio |

---

## 📜 License / Licenza

This project was developed as part of an academic activity for the **Fondamenti di Visione Artificiale e Biometria** course (A.A. 2024/2025).

Questo progetto è stato sviluppato come parte di un'attività accademica per il corso di **Fondamenti di Visione Artificiale e Biometria** (A.A. 2024/2025).

---

<p align="center">
  <strong>Made with ❤️ for Computer Vision Research</strong><br>
  <sub>Fondamenti di Visione Artificiale e Biometria • A.A. 2024/2025</sub>
</p>

<p align="center">
  <a href="#-fvab-deepfake-video-detection">⬆️ Back to Top / Torna su</a>
</p>
