# Vision Transformer for Powerline Damage Detection

## 🔍 Overview
This project uses a Vision Transformer (ViT) model to classify satellite imagery and detect powerline infrastructure damage. It leverages the XView2 dataset and integrates with HuggingFace Transformers and Albumentations for preprocessing.

## 🧠 Objective
To automate the classification of grid infrastructure damage (e.g., from storms or fallen lines) using a Vision Transformer model.

## 📦 Tech Stack
- Vision Transformer (ViT)
- HuggingFace Transformers
- Albumentations
- TensorBoard
- PyTorch

## 🗂️ Project Structure
- `notebooks/` – Jupyter notebooks for model development
- `src/` – Python scripts for training, inference, preprocessing
- `images/` – Sample output visualizations
- `data/` – Placeholder for dataset

## 📝 Dataset
XView2: Satellite Building Damage Assessment
[Kaggle Link](https://www.kaggle.com/competitions/xview2-building-damage-assessment)

## 🚀 Steps
1. Preprocess satellite imagery using Albumentations.
2. Fine-tune Vision Transformer for damage classification.
3. Visualize outputs and metrics in TensorBoard.
4. Optionally generate bounding boxes or masks.

## 📈 Results
Model performance is tracked using accuracy, precision, and confusion matrix.

## 📜 License
This project is licensed under the MIT License.

## 👤 Author
Pujah Balasubramaniam
