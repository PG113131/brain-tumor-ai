# 🧠 Brain Tumor Multi-Modal AI Diagnostic Suite

> **An AI-powered medical imaging system for automated brain tumor classification using Deep Learning, Explainable AI (Grad-CAM), and Large Language Models (LLMs).**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)

====================================================================================================

# 📚 Table of Contents

- Overview
- Features
- System Architecture
- Technology Stack
- Project Structure
- Installation
- Configuration
- Quick Start
- Utility Scripts
- Model Performance
- REST API
- Screenshots
- Future Improvements
- References
- Author
- License

====================================================================================================

# 📖 Overview

The **Brain Tumor Multi-Modal AI Diagnostic Suite** is an end-to-end AI application designed to assist in the diagnosis of brain tumors from MRI images.

The system combines Deep Learning, Explainable AI, and Large Language Models to provide:

- Brain tumor classification
- Confidence scores
- Class probability distribution
- Grad-CAM visual explanations
- AI-generated radiology reports
- Patient history management

The application consists of a **FastAPI backend**, a **Streamlit frontend**, an **EfficientNet-B0** classification model, **Grad-CAM** visualization, and **ChatGroq** for structured medical report generation.

====================================================================================================

# ✨ Features

- 🧠 Brain MRI Classification
- 🔥 Explainable AI with Grad-CAM
- 🤖 AI-generated Radiology Reports
- 📊 Confidence Scores & Probability Distribution
- 👤 Patient Information Management
- 📜 Prediction History
- 💾 SQLite Database
- ⚡ RESTful API using FastAPI
- 🎨 Interactive Streamlit Dashboard

====================================================================================================

# 🏗️ System Architecture

```text
                   Brain MRI Image
                           │
                           ▼
                  Image Preprocessing
                           │
                           ▼
             EfficientNet-B0 Classifier
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
 Tumor Prediction                  Grad-CAM Heatmap
           │                               │
           └───────────────┬───────────────┘
                           ▼
          ChatGroq Radiology Report Generator
                           │
                           ▼
                 FastAPI REST Backend
                           │
                           ▼
                 Streamlit Web Dashboard
                           │
                           ▼
                     SQLite Database
```

====================================================================================================

# 🛠 Technology Stack

| Category | Technologies |
|----------|--------------|
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Deep Learning | PyTorch, Torchvision |
| CNN Model | EfficientNet-B0 |
| Explainable AI | Grad-CAM |
| LLM | ChatGroq (Llama-3.3-70B-Versatile) |
| Prompt Framework | LangChain |
| ORM | SQLAlchemy |
| Database | SQLite |
| Image Processing | Pillow, OpenCV |
| Visualization | Matplotlib |

====================================================================================================

# 📂 Project Structure

```text
brain-tumor-ai/
│
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   └── components/
│
├── src/
│   ├── api/
│   ├── database/
│   ├── llm/
│   ├── schemas/
│   ├── vision/
│   ├── utils/
│   └── config.py
│
├── scripts/
│   ├── download_dataset.py
│   ├── download_weights.py
│   ├── verify_weights.py
│   ├── train_model.py
│   ├── evaluate_dataset.py
│   ├── predict_image.py
│   └── predict_directory.py
│
├── models/
│   └── README.md
│
├── data/
│   ├── uploads/
│   ├── heatmaps/
│   └── brain_tumor.db
│
├── assets/
├── tests/
├── requirements.txt
├── run.py
├── main.py
├── README.md
├── LICENSE
├── .env.example
└── .gitignore
```

====================================================================================================

# 🚀 Installation

Clone the repository.

```bash
git clone https://github.com/PG113131/brain-tumor-ai.git

cd brain-tumor-ai
```

Create a virtual environment.

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Install the required packages.

```bash
pip install -r requirements.txt
```

====================================================================================================

# ⚙ Configuration

This project requires a **Groq API Key** to generate AI-powered radiology reports.

## Step 1 – Get a Groq API Key

Visit:

https://console.groq.com/keys

Create a free API key.

## Step 2 – Create the Environment File

Copy the provided template.

### Windows

```powershell
copy .env.example .env
```

### Linux / macOS

```bash
cp .env.example .env
```

Open the `.env` file and replace the placeholder.

```env
GROQ_API_KEY=your_groq_api_key_here

DATABASE_URL=sqlite:///./data/brain_tumor.db
```

====================================================================================================

# 🚀 Quick Start

Start the application with a single command.

```bash
python run.py
```

The launcher will automatically:

- Check whether the trained model exists.
- Download the model weights if they are missing.
- Start the FastAPI backend.
- Start the Streamlit frontend.

Once started:

### FastAPI API

```
http://localhost:8000
```

### Swagger Documentation

```
http://localhost:8000/docs
```

====================================================================================================

# 📌 First Run

When you run the project for the first time:

- The pretrained model is downloaded automatically from the GitHub Release.
- No manual download is required.
- The download happens only once.
- Future runs reuse the downloaded model.

====================================================================================================

### Streamlit Dashboard

```
http://localhost:8501
```

====================================================================================================

# 🛠 Utility Scripts

| Script | Description |
|---------|-------------|
| `download_dataset.py` | Download the Brain Tumor MRI Dataset |
| `download_weights.py` | Download pretrained model weights |
| `verify_weights.py` | Verify model weights |
| `train_model.py` | Train the EfficientNet-B0 model |
| `evaluate_dataset.py` | Evaluate the trained model |
| `predict_image.py` | Predict a single MRI image |
| `predict_directory.py` | Predict all MRI images in a folder |

====================================================================================================

# 📈 Model Performance

| Metric | Score |
|---------|-------:|
| Accuracy | **93.69%** |
| Precision | **94.27%** |
| Recall | **93.69%** |
| F1 Score | **93.56%** |

====================================================================================================

# 🌐 REST API

| Method | Endpoint | Description |
|----------|----------|-------------|
| POST | `/api/v1/predict` | Predict a brain tumor from an MRI image |
| GET | `/api/v1/history` | Retrieve prediction history |
| GET | `/docs` | Interactive Swagger documentation |

====================================================================================================

# 📸 Screenshots

## Home Page

<img width="1880" height="890" alt="Screenshot 2026-07-27 195701" src="https://github.com/user-attachments/assets/508071cb-f8f6-497f-b8cc-d1bf68be7704" />


====================================================================================================

## Prediction Result

<img width="1835" height="854" alt="Screenshot 2026-07-27 195818" src="https://github.com/user-attachments/assets/0474ec25-c7ba-4733-81a9-13796912c562" />


====================================================================================================

## Prediction_History

<img width="1885" height="894" alt="Screenshot 2026-07-27 195727" src="https://github.com/user-attachments/assets/f5159433-befc-4326-8a7d-23d2d15f2963" />


====================================================================================================

# 🔮 Future Improvements

- Support for DICOM images
- Multi-modal MRI (T1, T2, FLAIR)
- User authentication
- Cloud deployment
- PACS integration
- Continuous model retraining

====================================================================================================

# 📚 References

- PyTorch Documentation
- FastAPI Documentation
- Streamlit Documentation
- LangChain Documentation
- Groq API Documentation
- EfficientNet Research Paper
- Grad-CAM Research Paper

====================================================================================================

# 👨‍💻 Author

**Pawan Ganesh Bokka**

B.Tech – Artificial Intelligence & Machine Learning

GitHub: https://github.com/PG113131
