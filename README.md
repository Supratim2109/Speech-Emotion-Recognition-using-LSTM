# Speech-Emotion-Recognition-using-LSTM

A deep learning–based Speech Emotion Recognition (SER) system built using **LSTM networks** to classify human emotions from speech signals. This project combines multiple benchmark datasets and provides a complete pipeline for training and inference.

---

## Project Description

Speech Emotion Recognition aims to identify the emotional state of a speaker from audio signals. This project implements an **LSTM-based neural network** capable of learning temporal dependencies in speech features to classify emotions such as *happy, sad, angry, neutral*, etc.

The system is trained on a **combined multi-dataset corpus** to improve generalization and robustness across different speakers, accents, and recording conditions.

---

## Repository Structure
Speech-Emotion-Recognition-using-LSTM/
│
├── app.py # Inference / demo script
├── main.ipynb # Model training, evaluation & experimentation
├── ser_combined_lstm.h5 # Trained LSTM model
├── label_encoder.pkl # Encoded emotion labels
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## Datasets Used

This project is trained using a **combined dataset** created from the following widely used emotion speech datasets:

- **TESS (Toronto Emotional Speech Set)**
- **CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)**
- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
- **IEMOCAP (Interactive Emotional Dyadic Motion Capture Database)**

Combining multiple datasets improves:
- Speaker diversity
- Emotional variation
- Real-world generalization performance

> Due to licensing and size constraints, datasets are **not included** in this repository.  
> You must download them separately and preprocess them before training.

---

## Feature Extraction

The model relies on extracted audio features such as:

- MFCCs (Mel-Frequency Cepstral Coefficients)
- Spectral features
- Temporal speech patterns

Feature extraction is performed using libraries like `librosa` and fed into the LSTM network for sequential learning.

---

## Model Architecture

- Long Short-Term Memory (LSTM) network
- Designed to capture temporal dependencies in speech
- Trained on combined dataset features
- Output layer mapped to encoded emotion labels

The trained model is saved as:
**ser_combined_lstm.h5**


---

## Installation

### Prerequisites

- Python 3.7+
- pip

### Setup

```bash
git clone https://github.com/Supratim2109/Speech-Emotion-Recognition-using-LSTM.git
cd Speech-Emotion-Recognition-using-LSTM
pip install -r requirements.txt



