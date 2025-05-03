# 🧠 CNN Research Paper Reproduction & Extension – Concrete Crack Detection

This repository contains our university semester project based on the reproduction and enhancement of a recent research paper on CNN-based concrete crack detection.

## 📄 Project Description

The project is divided into two phases:

### Phase 1: Paper Reproduction
- 📚 **Selected Paper**: [Application of Mask R-CNN and YOLOv8 Algorithms for Concrete Crack Detection (2023)](https://doi.org/10.3390/s23135933)
- 🎯 **Goal**: Reproduce the methodology and results from the paper.
- 🛠️ **Approach**:
  - Implemented both **Mask R-CNN** (via Detectron2) and **YOLOv8** (via Ultralytics).
  - Used a publicly available crack detection dataset (or substitute, if exact dataset was unavailable).
  - Performed inference, segmentation, and visualization on test images.
  - Planned comparison of key metrics: mAP, IoU, inference time.

### Phase 2: Contribution & Improvement *(Upcoming)*
- 🔬 Propose a meaningful extension or improvement to the original research.
- 🧪 Ideas in consideration:
  - Model performance enhancement using attention modules.
  - Hybrid architecture combining YOLOv8 speed with Mask R-CNN accuracy.
  - Robustness evaluation on varied or noisy datasets.
- 📊 Final report will include:
  - Summary of original research
  - Implementation details
  - Proposed contribution and experimental evaluation
  - Comparison with baseline results

---

## 🧠 Research Paper Summary

- **Title**: Application of Mask R-CNN and YOLOv8 Algorithms for Concrete Crack Detection  
- **Authors**: Syed Bilal Shah, et al.  
- **Year**: 2023  
- **Focus**: Evaluates Mask R-CNN vs YOLOv8 on concrete surface crack detection.  
- **Conclusion**: YOLOv8 shows faster inference; Mask R-CNN performs better on segmentation precision.

---

## 📁 Repository Structure

```bash
📦 Concrete-Crack-Detection-CNN
│
├── 📓 ANN_Project.ipynb           # Jupyter notebook with full implementation (YOLOv8 and Mask R-CNN)
├── 📄 Application_of_Mask_R-CNN...pdf   # Original research paper
├── 📁 models/                     # (Optional) Custom model weights or configurations
├── 📁 datasets/                   # (Optional) Sample dataset images or links
└── README.md                     # Project documentation
🛠️ Tech Stack
Python 3.x

Jupyter Notebook

Ultralytics YOLOv8

Detectron2 (Mask R-CNN)

OpenCV, NumPy, Matplotlib, PyTorch

🚀 Getting Started
Clone the repository


git clone https://github.com/your-username/concrete-crack-cnn.git
cd concrete-crack-cnn
Install dependencies


pip install -r requirements.txt
Run the notebook
Open ANN_Project.ipynb in Jupyter Lab or Jupyter Notebook.

Download dataset
Use any public dataset (e.g., SDNET2018) for testing if original is not available.

📊 Results Snapshot (Phase 1)
Model	mAP@0.5	IoU	Inference Time
YOLOv8	TBD	TBD	TBD
Mask R-CNN	TBD	TBD	TBD

Note: Quantitative results will be updated after full metric evaluation.

📌 Future Work (Phase 2 Goals)
Add attention-based enhancements or transformer backbones.

Improve model efficiency via quantization or pruning.

Experiment on noisy or challenging images.

Finalize contribution by May 8, 2025.

👥 Team Members
Student 1 –  Muhammad Muneer

Student 2 – Shahzaib Ali


📄 License
This project is for academic use only.

💬 Acknowledgements
Research paper authors for their contribution.

Ultralytics and Detectron2 communities for powerful frameworks.