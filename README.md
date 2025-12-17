# Breast_Cancer_Detection
An end-to-end AI-powered breast cancer detection system built using a hybrid deep learning architecture (CNN + Transformer) for accurate medical image classification. The project includes a Streamlit-based interactive web application for real-time predictions, making the model accessible and easy to use for non-technical users.

🚀 Project Highlights

✅ Hybrid CNN + Transformer architecture

✅ Classifies 8 types of breast cancer (Malignant & Benign)

✅ Achieved ~95% validation accuracy

✅ Robust data preprocessing & augmentation

✅ Interactive Streamlit web app with probability visualization

✅ Clean, modular, and deployment-ready codebase

🧠 Model Architecture
🔹 CNN Backbone

ResNet50 (pretrained)

Extracts rich spatial features from histopathology images

🔹 Transformer Encoder

Multi-Head Self Attention

Positional embeddings + CLS token

Captures global contextual dependencies

🔹 Classification Head

Fully connected layers

Softmax probabilities for confidence estimation

🧬 Cancer Classes
🔴 Malignant (Cancerous)

Papillary Carcinoma

Mucinous Carcinoma

Lobular Carcinoma

Ductal Carcinoma

🟢 Benign (Non-Cancerous)

Tubular Adenoma

Phyllodes Tumor

Fibroadenoma

Adenosis

📊 Model Performance

Validation Accuracy: ~95%

Metrics generated:

Accuracy

Precision / Recall / F1-score

Confusion Matrix

Per-class performance plots

All metrics and plots are automatically saved during training.

🖥️ Streamlit Web Application
Features

Upload histopathology images (JPG / PNG)

One-click cancer analysis

Displays:

Predicted cancer type

Malignant / Benign status

Confidence score

Full probability distribution (Plotly chart)

UI Highlights

Medical-style dashboard

Interactive probability visualization

Clear medical disclaimer & usage guidance

📁 Project Structure
├── app.py                     # Streamlit web application
├── breast_cancer_detection.py # Model training & evaluation script
├── requirements.txt           # Project dependencies
├── best_model.pth             # Trained model (hosted externally)
└── README.md

📦 Dataset

Due to GitHub size limits, the dataset is hosted externally.

📂 Dataset Source: Kaggle (Histopathology Breast Cancer Dataset)

🔗 Add dataset path inside training script:

data_dir = "www.kaggle.com/datasets/vanshagarwal12/breast-cancer-data"

🧠 Trained Model

The trained model (best_model.pth) is not included directly in this repository.

🔗 Model Download (External Hosting):
(Add Google Drive / Hugging Face link here)

Place the model file in the same directory as app.py before running the app.

⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/your-username/breast-cancer-detection.git
cd breast-cancer-detection

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run app.py

🛠️ Tech Stack

Python

PyTorch

Torchvision

Transformer Architecture

Streamlit

Plotly

NumPy, Pandas

OpenCV, PIL

Matplotlib, Seaborn

🔮 Future Improvements

🔹 FastAPI-based inference service

🔹 Dockerized deployment

🔹 Model monitoring & logging

🔹 Explainable AI (Grad-CAM)

🔹 Multi-dataset training support

👨‍💻 Author

Vansh Agarwal
B.Tech CSE | AI/ML Engineer
📧 Email: agarwalvansh0001@gmail.com

🔗 GitHub: https://github.com/vansh-ag

⭐ If You Found This Useful

Give this repository a ⭐ — it helps a lot!
