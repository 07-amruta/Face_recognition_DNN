# Face Recognition using Deep Neural Network (CelebA + FaceNet)

## 📌 Project Overview

This project implements a **face recognition system using Deep Neural Networks (DNNs)** to identify individuals from facial images. The system processes images from the CelebA dataset, detects faces, extracts deep features using a neural network, and performs identity matching based on similarity measurements.

The primary goal is to demonstrate how deep learning models can learn rich facial representations and enable automated recognition.

This project covers:

* Face detection
* Deep neural feature extraction
* Identity matching
* Unknown identity rejection

---

## 🧠 System Architecture

The recognition pipeline consists of the following stages:

1. Dataset Loading
2. Face Detection
3. Deep Neural Network Feature Extraction
4. Database Creation
5. Recognition Decision
6. Visualization

```
Input Image
     ↓
Face Detection (MTCNN)
     ↓
Deep Neural Network (FaceNet)
     ↓
512-D Embedding Vector
     ↓
Distance Comparison
     ↓
Identity Match / Unknown
```

---

## ⭐ Role of the Deep Neural Network (Core Component)

The **Deep Neural Network is the most important part of the system**.

Traditional face recognition relied on handcrafted features (edges, textures, distances between landmarks). These methods struggle under changes in lighting, pose, or expression.

### What the DNN does instead

The FaceNet DNN automatically learns hierarchical representations:

#### Early Layers

* Detect simple patterns
* Edges
* Shapes
* Textures

#### Middle Layers

* Detect facial components
* Eyes
* Nose
* Mouth
* Contours

#### Deep Layers

* Encode identity-specific features
* Complex geometry
* Subtle variations
* Global structure

---

### Output of the DNN

The network converts each face into a:

## ➤ 512-Dimensional Embedding Vector

This vector:

* Represents identity information
* Is invariant to lighting/pose
* Enables comparison between faces
* Acts as a numerical fingerprint

Example:

```
Face Image → DNN → [0.21, -1.44, 0.78, ... 512 values]
```

---

### Why This Matters

Because of the DNN:

* Same person → embeddings close together
* Different people → embeddings far apart

This property makes recognition possible using simple distance metrics.

Without the DNN, recognition accuracy would be extremely poor.

---

## 🤖 Deep Learning Models Used

### 1️⃣ MTCNN — Face Detection

Detects and crops facial regions from images.

Purpose:

* Isolate face area
* Align orientation
* Remove background noise

---

### 2️⃣ FaceNet (InceptionResnetV1) — Deep Neural Network

This is the primary DNN used in the project.

#### Architecture Characteristics

* Deep convolutional neural network
* Residual connections
* Trained on millions of faces
* Learns discriminative embeddings

#### Function in this Project

* Converts face image → feature embedding
* Enables identity comparison
* Drives recognition accuracy

This model replaces the need to train a large network from scratch.

---

## 📊 Dataset Used — CelebA

* 200,000+ images
* 10,000+ identities
* Wide variations in appearance

Used for:

* Database creation
* Recognition testing

Dataset is downloaded automatically through PyTorch utilities.

---

## 🗂️ Database Construction

Steps:

1. Detect face
2. Run through DNN
3. Store embedding

This simulates enrolling known individuals in a biometric system.

---

## 🎲 Randomized Testing Strategy

To demonstrate realistic behavior:

* 10 images shown per run
* Up to 5 selected from database
* Remaining chosen externally
* Order shuffled

This demonstrates:

* Recognition capability
* Unknown rejection

---

## 📏 Recognition Method

Similarity measured using Euclidean distance:

```
distance = || embedding_test − embedding_database ||
```

Decision rule:

* distance < threshold → Known
* distance ≥ threshold → Unknown

This shows how DNN embeddings enable simple yet effective matching.

---

## ▶️ How to Run

Install:

```
pip install torch torchvision facenet-pytorch matplotlib
```

Execute:

```
python celeba_facenet_demo_random.py
```

Close each plot window to move to next image.

---

## 🎓 Learning Outcomes

This project demonstrates:

* Practical application of deep neural networks
* Feature representation learning
* Biometric system design
* Integration of multiple AI components
* Real-world dataset usage

---

## ⚠️ Limitations

* Small identity database
* No classifier training
* No persistent storage
* Limited quantitative evaluation

---

## 🚀 Future Improvements

* Real-time webcam recognition
* Larger enrollment database
* SVM/KNN classifier
* Accuracy metrics
* GUI interface
* Model fine-tuning

---

## 🏁 Conclusion

The project showcases how Deep Neural Networks transform facial images into meaningful numerical representations that enable identity recognition. By leveraging pretrained deep models, the system achieves reliable feature extraction and demonstrates the effectiveness of deep learning in biometric applications.

The DNN serves as the foundation of the recognition system, providing robust, invariant features that allow accurate comparison and decision-making.

---
