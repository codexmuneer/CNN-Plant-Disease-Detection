
# 🥔 Identification of Potato Plant Diseases using CNN

> **Paper Reference**: *Prabhat Srivastava et al. (2024)* – Educational Administration: Theory and Practice, [DOI: 10.53555/kuey.v30i5.5252](https://kuey.net/)

This repository contains a complete, modular, and reproducible implementation of the Convolutional Neural Network (CNN) model proposed for detecting potato leaf diseases (early blight, late blight, and healthy leaves) using image classification techniques.

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

In this project, we apply a deep learning approach to automate the identification of potato plant diseases through image-based classification using CNNs. This can help farmers by reducing the dependency on manual inspection and enabling early disease detection.

---

## 🏗️ Architecture

The model architecture includes:

- Input Preprocessing (Resize + Rescale)
- Data Augmentation (Flip + Rotate)
- CNN Layers: 6× Conv2D → MaxPool → Flatten → Dense
- Softmax classifier for 3 categories

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

---
