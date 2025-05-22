
# Identification of Plant Diseases using CNN

> **Paper Reference**: *Prabhat Srivastava et al. (2024)* – Educational Administration: Theory and Practice, [DOI: 10.53555/kuey.v30i5.5252](https://kuey.net/)

This repository contains a complete, modular, and reproducible implementation of Convolutional Neural Network (CNN) models for detecting plant leaf diseases using image classification techniques.

Originally based on the research paper by Prabhat Srivastava et al. for **potato plant** diseases, we extended the project to work with **pepper bell** plants and later attempted classification on **tomato plants**. During the transition to tomatoes, we discovered that the previous CNN architecture underperformed, which led us to redesign the network architecture to better suit the tomato dataset.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation & Results](#evaluation--results)
- [Predictions](#predictions)
- [Visualizations](#visualizations)
- [Citation](#citation)

---

## 🧠 Overview

This project leverages deep learning to automate the detection of plant diseases via image classification. Initially designed for potatoes (early blight, late blight, healthy), it was later adapted to **pepper bell** with strong results and finally evaluated on **tomato plants**. The tomato dataset did not perform well using the original CNN, so we completely revamped the architecture for this case.

---

## 🏗️ Architecture

### 🥔 Previous CNN Architecture (for Potato & Pepper Bell):

```python
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * IMAGE_SIZE//32 * IMAGE_SIZE//32, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )
```

### 🍅 Updated CNN Architecture (for Tomato):

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

---

## 📂 Dataset

- Total Images: ~2,150
- Categories:
  - Early Blight
  - Late Blight
  - Healthy
- Source: Provided in the paper; for this repo, use structured image folders.

```
/pepper-bell-dataset/
  ├── Early_Blight/
  ├── Late_Blight/
  └── Healthy/
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/potato-disease-cnn.git
cd potato-disease-cnn
pip install -r requirements.txt
```

---

## ▶️ Usage

1. Place your dataset inside the root directory under `pepper-bell-dataset/`.
2. Run the notebook step-by-step:
   - `01_data_preprocessing.ipynb`
   - `02_model_training.ipynb`
   - `03_evaluation_and_visualization.ipynb`

---

## 🏋️ Model Training

- Image size: `256x256`
- Batch size: `32`
- Epochs: `15–50`
- Optimizer: `Adam`
- Loss: `SparseCategoricalCrossentropy`
- Data split: `Train (80%)`, `Validation (10%)`, `Test (10%)`

---

## 📊 Evaluation & Results

- **Validation Accuracy**: ~97%
- **Test Accuracy**: ~99%
- **Model Comparison (from paper)**:

| Algorithm | Accuracy  |
|-----------|-----------|
| ANN       | 85%       |
| SVM       | 88.89%    |
| **CNN**   | **99.07%**|

---

## 🔍 Predictions

Use this utility to predict a single image or batch:

```python
def classify_image(model, image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_label = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_label, confidence
```

---

## 📈 Visualizations

- Epoch-wise training and validation accuracy/loss
- Prediction visualization grid
- Confusion matrix (optional extension)

---

## 📜 Citation

```bibtex
@article{srivastava2024cnn,
  title={Identification of potato plant diseases using CNN model},
  author={Ram Kinkar Pandey, Gaurav Kumar Srivastava, Prabhat Kr. Srivastava, Chandani Sharma, Neha Chauhan},
  journal={Educational Administration: Theory and Practice},
  volume={30},
  number={5},
  pages={12656-12662},
  year={2024},
  doi={10.53555/kuey.v30i5.5252}
}
```

---

## 🙌 Acknowledgements

Thanks to the authors of the original paper and the open-source community for tools like TensorFlow, Keras, and Matplotlib.
