# 🚗 Driver Drowsiness Detection System

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Detection-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

</p>

## 📖 Overview

Driver fatigue is one of the leading causes of road accidents worldwide. This project presents a **real-time Driver Drowsiness Detection System** that monitors a driver's face through a webcam and detects signs of drowsiness using **Deep Learning**.

The system combines **MediaPipe Face Detection** for accurate face localization with **EfficientNet-B0** for binary image classification. To improve prediction stability and reduce frame-by-frame fluctuations, a **prediction smoothing algorithm** is applied before displaying the final result.

Whether you're learning Computer Vision, Deep Learning, or building an AI-powered safety application, this project serves as a practical implementation.

---

## ✨ Features

- 🚘 Real-time webcam monitoring
- 🧠 EfficientNet-B0 based deep learning classifier
- 😀 MediaPipe face detection
- 📊 Prediction confidence score
- 🔄 Prediction smoothing for stable results
- ⚡ Live FPS display
- 🎯 Binary classification:
  - **Drowsy**
  - **Non Drowsy**

---

## 🖼️ Project Workflow

```text
Webcam
   │
   ▼
MediaPipe Face Detection
   │
   ▼
Face Crop & Preprocessing
   │
   ▼
EfficientNet-B0 Model
   │
   ▼
Prediction Smoothing
   │
   ▼
Drowsy / Non Drowsy Detection
```

---

# 📂 Dataset

This project uses the **Driver Drowsiness Dataset (DDD)** from Kaggle.

🔗 **Dataset Link**

https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd

### Dataset Structure

```
raw_dataset/
│
├── Drowsy/
└── Non Drowsy/
```

---

# 📁 Project Structure

```
Driver-Drowsiness-Detection/
│
├── app.py
├── preprocess.py
├── smooth_prediction.py
├── requirements.txt
├── README.md
│
├── model/
│   └── drowsiness_detector_model.pth
│
├── processed_dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
└── raw_dataset/
    ├── Drowsy/
    └── Non Drowsy/
```

---

# 🧠 Model Architecture

This project uses **EfficientNet-B0**, a lightweight yet highly accurate convolutional neural network.

### Model Pipeline

- Input Image
- Image Preprocessing
- EfficientNet-B0 Feature Extraction
- Fully Connected Classification Layer
- Sigmoid Activation
- Binary Prediction

Output:

- 😴 Drowsy
- 😊 Non Drowsy

---

# 🔧 Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Driver-Drowsiness-Detection.git

cd Driver-Drowsiness-Detection
```

---

## 2️⃣ Create Virtual Environment (Optional)

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Project

Launch the real-time drowsiness detector:

```bash
python app.py
```

The application will:

- Open your webcam
- Detect your face
- Predict drowsiness in real time
- Display confidence score
- Show FPS

Press **Q** to exit.

---

# 📜 Script Description

## 🔹 preprocess.py

Responsible for preparing the dataset.

✔ Splits dataset into:

- Train
- Validation
- Test

✔ Performs preprocessing

✔ Saves processed images

---

## 🔹 smooth_prediction.py

Reduces prediction flickering by applying a moving average across recent predictions.

Benefits:

- Stable predictions
- Less false alarms
- Better user experience

---

## 🔹 app.py

Main application.

Functions:

- Starts webcam
- Detects faces using MediaPipe
- Preprocesses face images
- Loads trained model
- Predicts drowsiness
- Applies smoothing
- Displays prediction and FPS

---

# 📦 Requirements

Major libraries used in this project:

- Python 3.10+
- PyTorch
- TorchVision
- OpenCV
- MediaPipe
- NumPy
- Pillow
- tqdm
- scikit-learn

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# 💡 Future Improvements

- 🔊 Audio alarm for drowsiness detection
- 📱 Mobile application integration
- 🌙 Night vision support
- 🚗 Multiple face detection
- 📈 Driver attention analytics
- ☁️ Cloud-based monitoring dashboard

---

# 📷 Demo

You can add screenshots or GIFs here.

Example:

```
demo/
├── output.gif
├── screenshot1.png
└── screenshot2.png
```

Then include them:

```markdown
![Demo](demo/output.gif)
```

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 👩‍💻 Author

**Shikha Shukla**

B.Tech Computer Science Engineering

Passionate about Artificial Intelligence, Deep Learning, Computer Vision, and Open Source.

---

## ⭐ Show Your Support

If you found this project helpful, please consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates future improvements!

---

## 📄 License

This project is intended for educational and research purposes.


















