

# 🍎 Goodness Grader
This project uses **Convolutional Neural Networks (CNN)** to classify fruit images into **Good**, **Bad**, or **Mixed** quality. It also compares results with a classical **Random Forest** model to demonstrate the effectiveness of deep learning for image-based quality assessment.

---

## 📌 Highlights

* ✅ **95% test accuracy** using a custom-built **CNN**
* 🆚 Compared against a **Random Forest** classifier (only \~87% accuracy, 17% recall for "mixed")
* 🔍 Feature extraction using image histograms for classical ML
* 🖼️ Web app interface using Flask for real-time predictions

---

## 🧠 Models Overview

### 🥇 **Main Model: CNN (Keras Sequential)**

* Input: 64x64 RGB images
* Layers:

  * Conv2D + ReLU
  * MaxPooling
  * Dropout
  * Fully connected Dense layers
* Output: 3-class softmax (good, bad, mixed)
* Trained on preprocessed fruit images
* Final Accuracy: **\~95%**

### 🧪 **Baseline Model: Random Forest**

Used for comparison only.

* Feature extraction via color histograms
* Trained on flattened histogram vectors
* Final Accuracy: **\~87%**
* Key finding: **Only 17% recall for "mixed" class** → Not suitable for complex visual tasks

---

## 📊 Performance Comparison

| Model          | Accuracy | Recall (Mixed)             |
| -------------- | -------- | -------------------------- |
| **CNN (Main)** | 95%      | High (details in notebook) |
| Random Forest  | 87%      | **17%** ❌                  |

> 📌 Conclusion: Traditional models like Random Forest struggle with nuanced visual classes like "mixed" fruit quality. CNNs learn better spatial features.

---

## 📂 Project Structure

```
chan9an/
├── app.py                             # Flask app for image upload and prediction
├── index.html                         # Frontend for web UI
├── custom_layer.py                    # Custom Keras layers (if used)
├── fruit-quality-prediction-95...ipynb  # ✅ Main CNN model notebook
├── RandomForest.ipynb                 # ❌ Classical baseline notebook
├── fruit_quality_model4.h5           # Trained Keras model
├── Dataset/
│   └── Processed Images_Fruits/      # Image folders (Good, Bad, Mixed)
├── README.md
└── .gitignore
```

---

## 🚀 Getting Started

### 🧰 Requirements

```bash
pip install tensorflow opencv-python pandas numpy scikit-learn flask
```

### 🧪 Run the CNN Notebook

```bash
jupyter notebook fruit-quality-prediction-95-accuracy.ipynb
```

### 🌐 Launch Web App

```bash
python app.py
```

Then go to `http://localhost:5000` to upload images and get predictions.

---

## 🖼️ Dataset Structure

Organize your dataset like this:

```
Dataset/
└── Processed Images_Fruits/
    ├── Good Quality_Fruits/
    ├── Bad Quality_Fruits/
    └── Mixed Qualit_Fruits/
```

Each folder should contain subfolders for each fruit type, which in turn hold the images.

---

## ✅ Future Improvements

* Better handling of borderline/mixed cases
* Expand dataset size and diversity
* Mobile or cloud deployment


---

Would you like a visual badge (like accuracy/shield.io) or a demo video section in this README too?
