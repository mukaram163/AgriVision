# 🌿 AgriVision – Automated Plant Phenotyping with Deep Learning

[![CI](https://github.com/mukaram163/AgriVision/actions/workflows/ci.yml/badge.svg)](https://github.com/mukaram163/AgriVision/actions)
[![Hugging Face Spaces](https://img.shields.io/badge/🚀 Live%20Demo-HuggingFace-blue)](https://mukaram163-agriVision.hf.space)

AgriVision is a **deep-learning–powered system** that automatically detects plant diseases and extracts phenotypic traits from leaf images.  
Built using **PyTorch**, **Streamlit**, and **OpenCV**, it demonstrates full-stack ML engineering —  
from **data wrangling** and **model training** to **containerization**, **CI/CD**, and **deployment**.

---

## 🚀 Live Demo
👉 **Try the app:** [https://mukaram163-agriVision.hf.space](https://mukaram163-agriVision.hf.space)

The dashboard supports real-time image upload, inference visualization, and metric tracking —  
a showcase of end-to-end **MLOps readiness**.

---

## 🧠 Problem Statement
Agricultural scientists spend extensive time manually inspecting plant images to identify diseases.  
**AgriVision** automates this process using a fine-tuned ResNet18 model trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease),  
reducing manual workload and improving accuracy.

---

## 🎯 Objectives
- Detect and classify plant leaf diseases (healthy vs diseased).  
- Automate image preprocessing and model inference.  
- Display predictions and performance metrics in an interactive Streamlit dashboard.  
- Enable CI/CD + containerized cloud deployment for reproducibility.

---

## 🧩 Tech Stack
| Category | Tools & Libraries |
|-----------|------------------|
| Deep Learning | PyTorch · Torchvision |
| Data Wrangling | Pandas · PySpark (Mock) |
| Visualization | Matplotlib · Seaborn |
| App Framework | Streamlit |
| MLOps & Deployment | Docker · GitHub Actions · Hugging Face Spaces |
| Model | ResNet18 fine-tuned on PlantVillage |

---

## 📸 Dashboard Previews

| Home | Prediction |
|------|-------------|
| ![Home](assets/screenshot_home.png) | ![Prediction](assets/screenshot_prediction.png) |

---

## 📈 Results
| Metric | Score |
|---------|-------|
| Test Accuracy | > 95 % |
| Test Loss | Low (< 0.1) |
| Model | ResNet18 (Transfer Learning) |

---

## 💾 Repository Structure

AgriVision/
├── app.py # Streamlit dashboard
├── src/ # Model, training, preprocessing code
├── models/ # Trained model weights
├── results/ # Evaluation metrics
├── notebooks/ # EDA + wrangling notebooks
├── assets/ # Screenshots & visuals
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci.yml


---

## 👨‍💻 Author
**Mukaram Ali**  
🌐 [LinkedIn](https://linkedin.com/in/mukaram-ali-a05061279) · [GitHub](https://github.com/mukaram163)

---

## 🧱 License
MIT License © 2025 Mukaram Ali
