
# 🧠 Deep Learning Classification on PathMNIST Dataset

This project performs image classification on the **PathMNIST** dataset using a Convolutional Neural Network (CNN) implemented in TensorFlow/Keras. The goal is to classify histopathology images into 9 different tissue types.

---

## 📁 Dataset

- **Source:** PathMNIST (part of [MedMNIST v2](https://medmnist.com/))
- **Input Shape:** 28x28 RGB images
- **Classes:** 9 tissue types
- **Size:**
  - Train: 89,996 images
  - Test: 7,180 images

---

## 🔧 Setup

```bash
pip install tensorflow==2.10.0 medmnist matplotlib scikit-learn
```

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from medmnist import INFO, PathMNIST
```

---

## 📊 Data Preprocessing

- Normalize images to [0, 1]
- Resize input images to **128x128**
- Split into train/test sets

---

## 🏗️ Model Architecture

```python
Sequential([
    Rescaling(1./255),
    Resizing(128, 128),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(9, activation='softmax')
])
```

- **Loss Function:** `SparseCategoricalCrossentropy`
- **Optimizer:** `Adam`
- **Metrics:** `Accuracy`

---

## 🏃 Training

- Trained on 15,000 samples (subset of full training set)
- **Epochs:** 10
- **Validation Accuracy:** ~75%

---

## 📈 Evaluation

```python
Test Accuracy: 66.4%
```

- Indicates decent generalization
- Could benefit from further hyperparameter tuning or data augmentation

---

## 💾 Save & Predict

- Model saved as `.h5` file
- Single image prediction tested visually using `matplotlib`

---

## 📌 Conclusion

This notebook demonstrates a full workflow for applying a CNN to a medical imaging classification problem using the PathMNIST dataset. The results suggest the potential for better accuracy through more extensive training or architecture optimization.
