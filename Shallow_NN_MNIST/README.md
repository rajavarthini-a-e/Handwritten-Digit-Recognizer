# ✍️ Handwritten Digit Recognition using Shallow Neural Network

## 📌 Project Overview

This project is a Handwritten Digit Recognition System built using a **Shallow Neural Network (SNN)** with TensorFlow and Keras.

The model is trained on the **MNIST dataset** and predicts handwritten digits (0–9). A **Streamlit web application** allows users to upload their own handwritten digit image and receive the predicted digit along with the confidence score.

---

## 🚀 Features

- Train a Shallow Neural Network on the MNIST dataset
- Predict handwritten digits (0–9)
- Upload custom handwritten images
- Image preprocessing using OpenCV
- Display prediction confidence
- Interactive Streamlit web application

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Streamlit

---

## 📂 Project Structure

```
Handwritten-Digit-Recognition-ShallowNN/
│
├── app.py
├── predict.py
├── train_model.py
├── explore_dataset.py
├── mnist_shallow_model.keras
├── uploads/
├── screenshots/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

- Dataset: MNIST
- Training Images: 60,000
- Testing Images: 10,000
- Image Size: 28 × 28 pixels
- Classes: 10 (Digits 0–9)

---

## 🧠 Model Architecture

```
Input Layer
784 Neurons
      │
      ▼
Dense Layer
10 Neurons
Softmax Activation
```

Since this project uses a **Shallow Neural Network**, it contains only a single Dense output layer.

---

## ⚙️ Image Preprocessing

Before prediction, the uploaded image undergoes the following preprocessing steps:

- Convert image to grayscale
- Resize to 28 × 28 pixels
- Invert colors
- Normalize pixel values
- Flatten image to a 784-dimensional vector

---

## 📈 Results

- Training Accuracy: ~93%
- Test Accuracy: ~92.6%

The application predicts the handwritten digit and displays the confidence score.

---

## ▶️ How to Run

### Clone the repository

```bash
git clone https://github.com/yourusername/Handwritten-Digit-Recognition-ShallowNN.git
```

### Navigate to the project

```bash
cd Handwritten-Digit-Recognition-ShallowNN
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python train_model.py
```

### Launch the web application

```bash
streamlit run app.py
```

---

## 📷 Application Preview

### Upload Image

*(Add a screenshot here)*

### Prediction Result

*(Add another screenshot here)*

---

## 🎯 Future Improvements

- Improve preprocessing for real-world handwritten images
- Build a Deep Neural Network (DNN)
- Replace the Shallow Neural Network with a Convolutional Neural Network (CNN)
- Deploy the application online using Streamlit Community Cloud

---

## 👨‍💻 Author

**Rajavarthini**

This project was developed as part of my Deep Learning learning journey to understand the fundamentals of neural networks and image classification.