# 🌿 AgriVision – Automated Plant Phenotyping with Deep Learning

[![CI](https://github.com/mukaram163/AgriVision/actions/workflows/ci.yml/badge.svg)](https://github.com/mukaram163/AgriVision/actions)
[![Hugging Face Spaces](https://img.shields.io/badge/🚀 Live%20Demo-HuggingFace-blue)](https://huggingface.co/spaces/mukaram163/AgriVision)

AgriVision is a **deep-learning–powered system** that automatically detects plant diseases and extracts phenotypic traits from leaf images.  
Built using **PyTorch**, **Streamlit**, and **OpenCV**, it demonstrates full-stack ML engineering —  
from **data wrangling** and **model training** to **containerization**, **CI/CD**, and **deployment**.

---

## 🚀 Live Demo
👉 **Try the app here:** [https://huggingface.co/spaces/mukaram163/AgriVision](https://huggingface.co/spaces/mukaram163/AgriVision)

The interactive dashboard supports real-time image uploads, visualized model predictions, and persistent prediction history —  
showcasing **end-to-end MLOps and data engineering readiness**.

---

## 🧠 Problem Statement
Agricultural scientists spend extensive time manually inspecting plant images to identify diseases and assess plant health.  
**AgriVision** automates this process using a fine-tuned **ResNet18** model trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease),  
reducing manual workload and enabling data-driven insights for crop management.

---

## 🎯 Objectives
- 🧠 Detect and classify plant leaf diseases (healthy vs diseased).  
- ⚙️ Automate preprocessing, inference, and feature extraction.  
- 📊 Visualize metrics and predictions in an interactive dashboard.  
- 🧱 Implement CI/CD + containerized deployment via **GitHub Actions**, **Docker**, and **Hugging Face Spaces**.  
- 💾 Store prediction results using **SQLite** for historical analysis.

---

## 🧩 Tech Stack
| Category | Tools & Libraries |
|-----------|------------------|
| Deep Learning | PyTorch · Torchvision |
| Data Wrangling | Pandas · PySpark (Optional) |
| Visualization | Matplotlib · Seaborn |
| App Framework | Streamlit |
| Database | SQLite (Persistent prediction storage) |
| MLOps & Deployment | Docker · GitHub Actions · Hugging Face Spaces |
| Model | ResNet18 (Transfer Learning) |

---

## 📸 Dashboard Previews

| 🏠 Home | 🧠 Prediction |
|:-------:|:-------------:|
| ![Home](assets/screenshot_home.png) | ![Prediction](assets/screenshot_prediction.png) |

---

## 💾 Database Integration & History Tab

AgriVision now includes a **data persistence layer** powered by **SQLite**, showcasing real-world data engineering practices.

### 🔹 Key Features:
- Automatically logs every inference (image name, predicted class, confidence, timestamp).  
- Stores predictions locally in a lightweight `results.db` file.  
- Provides a new **📜 History Tab** in the dashboard for quick review and reproducibility.  
- Demonstrates **ETL principles** (Extract → Transform → Load) within an ML app context.

### 📂 Database Schema:
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| filename | TEXT | Uploaded image name |
| prediction | TEXT | Predicted disease class |
| confidence | REAL | Model confidence score |
| timestamp | TEXT | Date and time of prediction |

---

## 📈 Results
| Metric | Score |
|---------|-------|
| Test Accuracy | > 95 % |
| Test Loss | Low (< 0.1) |
| Model | ResNet18 (Transfer Learning) |

> Evaluation metrics are logged and visualized through the Streamlit dashboard.  
> Prediction history is automatically stored and retrievable in real time.

---

## 📂 Repository Structure
```bash
AgriVision/
├── app.py                   # Streamlit dashboard (inference + history + metrics)
├── src/
│   ├── model.py             # Model creation & checkpoint loading
│   ├── database.py          # SQLite integration for persistent storage
│   ├── preprocessing.py     # Image transforms and loaders
│   ├── train.py             # Training logic
│   ├── evaluate.py          # Evaluation metrics and confusion matrix
│   └── inference.py         # Helper for predictions
├── models/                  # Trained model weights (best_model.pth)
├── results/                 # Evaluation metrics + SQLite database
├── notebooks/               # EDA and data wrangling notebooks
├── assets/                  # App screenshots for documentation
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci.yml

🧰 DevOps Integration

CI/CD: Automated code quality checks (flake8) via GitHub Actions.

Docker: Fully containerized for cloud or on-prem deployment.

Hugging Face Spaces: Streamlit-based live demo deployment with persistent UI.

👨‍💻 Author

Mukaram Ali
🌐 LinkedIn
 · GitHub

📍 Machine Learning Engineer · Specializing in Deep Learning, Computer Vision & MLOps

🧱 License

MIT License © 2025 Mukaram Ali


---

### ✅ What This Update Does for You:
- Makes your project *portfolio-grade* — recruiters see MLOps + DataOps in one project.
- Highlights **real database integration** (shows engineering maturity).
- Your Hugging Face link is now clearly showcased as the **live deployment**.
- The README now has full documentation flow: problem → solution → engineering → deployment.

---

Would you like me to also generate a **diagram (system architecture)** to include in your README — showing  
“Data Input → Preprocessing → Model → SQLite → Streamlit Dashboard”?  
It would make the README *stand out visually* to employers.
