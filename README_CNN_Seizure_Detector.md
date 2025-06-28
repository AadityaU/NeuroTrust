
# EEG Seizure Detection System

This repository provides a complete solution to detect epileptic seizures from EEG signals using a **1D Convolutional Neural Network (CNN)**. The system includes model training, inference, and a professional web application interface for real-time visualization and decision support.

---

##  Objective

Develop a clinically relevant, automated seizure detection system using deep learning, aimed at assisting neurologists and researchers in identifying seizure-prone EEG segments.

---

##  Project Structure

```
eeg-seizure-app/
├── Untitled.ipynb              # Jupyter notebook used to train the CNN model
├── detection.py                # Streamlit-based dashboard for predictions and analytics
├── model/
│   ├── eeg_cnn_model.h5        # Trained CNN model
│   ├── scaler.joblib           # Optional: preprocessing scaler
│   └── label_encoder.joblib    # Optional: label encoder for outputs
├── assets/
│   └── logo.png                # App branding/logo
├── requirements.txt            # Dependency file (CNN-based)
└── README.md                   # Project documentation
```

---

##  Features

- **Deep Learning Model**: 1D CNN trained on 4-second windowed EEG segments
- **Risk Classification**: Classifies EEG data into High Risk (seizure) or Low Risk (normal)
- **Interactive Web App**: Streamlit frontend for uploading EEG files and visualizing output
- **Advanced Charts**: Risk timeline, confidence distribution, 3D scatter plot, and heatmap
- **Downloadable Results**: Export prediction results in CSV format

---

##  Visual Outputs

-  Risk Timeline Chart
-  Seizure Risk Heatmap
-  3D Risk Landscape
-  Confidence Histogram
-  Prediction Report (CSV)

---

##  Data Format

- Input File: `.csv`
- Each row represents a **4-second EEG window** of features
- No headers required
- Features must match model training format

---

##  Model Details

- Model Type: **1D Convolutional Neural Network (CNN)**
- Framework: **TensorFlow / Keras**
- Trained on pre-segmented, labeled EEG data
- Output Classes: High Risk (Seizure) / Low Risk (Normal)

---

##  How to Use

###  Local Setup

```bash
cd eeg-seizure-app
pip install -r requirements.txt
streamlit run detection.py
```


##  Requirements

> For full environment: see [`requirements_cnn_based.txt`](requirements_cnn_based.txt)

Key packages:

```txt
tensorflow==2.15.0
streamlit
pandas
numpy
plotly
joblib
scikit-learn
matplotlib
```

---

##  Author

**Aaditya Upadhyay**  
B.Tech, Information Technology  
Madan Mohan Malaviya University of Technology, Gorakhpur

---

##  License

For research and educational use only. Not certified for clinical use.

---

##  Acknowledgements

- Siena & CHB-MIT EEG Dataset Providers
- Keras, TensorFlow, and Streamlit
- Research papers on CNN-based seizure detection

---
