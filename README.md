# 🧠 Face Recognition using Deep Neural Network (FaceNet + CelebA)

A deep learning-based face recognition system that detects faces using **MTCNN**, extracts **512-dimensional face embeddings** using a pretrained **FaceNet (InceptionResnetV1)** model, and performs identity matching using Euclidean distance.

This project demonstrates a complete modern face recognition pipeline using PyTorch.

---

## 📌 Project Overview

This system performs:

* Face Detection (MTCNN)
* Deep Feature Extraction (FaceNet)
* Embedding-based Identity Matching
* Unknown Face Rejection
* Randomized Testing Demo

The model does not classify faces directly. Instead, it generates **face embeddings** (numerical identity representations) and compares them using similarity metrics.

---

## 🏗️ System Architecture

```
Input Image
     ↓
Face Detection (MTCNN)
     ↓
FaceNet (InceptionResnetV1)
     ↓
512-D Face Embedding
     ↓
Euclidean Distance Matching
     ↓
Match / Unknown
```

---

## 📂 Dataset

This project uses the **CelebA dataset**, which contains over 200,000 face images.

The dataset is automatically downloaded using `torchvision` when you run the script.

No manual download is required.

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/dnn-face-recognition.git
cd face_recognition_dnn
```

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Required Dependencies

```bash
pip install torch torchvision facenet-pytorch matplotlib numpy
```

If you have a GPU, install the CUDA-enabled version of PyTorch from:
https://pytorch.org/

---

## ▶️ How to Run the Project

Run:

```bash
python facial_recognition_facenet.py
```

On first run:

* CelebA dataset will download automatically
* Database embeddings will be generated
* 10 randomized test images will be shown
* Up to 5 images will match known identities
* Remaining images will be classified as unknown

Close each image window to proceed to the next.

---

## 🧠 How Recognition Works

1. Faces are detected using **MTCNN**
2. FaceNet generates a **512-dimensional embedding**
3. Embeddings are stored in a database
4. Test embeddings are compared using Euclidean distance
5. If distance < threshold (1.0), face is recognized

---

## 📁 Project Structure

```
.
├── celeba_facenet_demo_random.py
├── README.md
└── data/
    └── celeba/
```

The `data/` folder is created automatically when the dataset downloads.

---

## 🔍 Key Technologies Used

* **PyTorch** – Deep learning framework
* **Torchvision** – Dataset and image transforms
* **facenet-pytorch** – FaceNet & MTCNN implementation
* **NumPy** – Numerical operations
* **Matplotlib** – Visualization

---

## 🎯 Features

* Pretrained deep neural network
* Embedding-based recognition
* Randomized known/unknown testing
* GPU support (if available)
* Fully automated dataset download

---

## 📊 Configuration Parameters

You can modify these values inside the script:

```python
DATABASE_SIZE = 20
TOTAL_TEST_IMAGES = 10
MAX_KNOWN = 5
THRESHOLD = 1.0
```

* `DATABASE_SIZE` → number of enrolled identities
* `TOTAL_TEST_IMAGES` → number of test images shown
* `MAX_KNOWN` → maximum matching images in demo
* `THRESHOLD` → recognition sensitivity

---

## 🚀 Future Improvements

* Real-time webcam recognition
* Save/load embedding database
* Add GUI interface
* Implement classifier (SVM/KNN)
* Evaluate accuracy metrics

---

## 🎓 Educational Purpose

This project demonstrates how deep neural networks:

* Learn hierarchical visual features
* Encode facial identity into embeddings
* Enable scalable biometric recognition systems

It serves as an academic mini-project illustrating modern face recognition pipelines.

---

## 📜 License

This project is for educational purposes only.
CelebA dataset usage follows its original research license.

---

## 👩‍💻 Author

Amruta Panda
Deep Learning Mini Project

---