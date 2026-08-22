# 🩻 MURA Abnormality Detector

An end-to-end Deep Learning application and web interface designed to detect musculoskeletal abnormalities in upper extremity X-rays using Stanford's MURA (Musculoskeletal Radiographs) dataset.

---

## 📌 Project Overview
* **Model Architecture:** Custom Convolutional Neural Network (CNN) / Transfer Learning.
* **Frameworks:** TensorFlow, Keras, NumPy, Pandas.
* **Frontend:** Streamlit web app for real-time radiograph inference.
* **Dataset:** Stanford MURA Dataset (Upper limb radiographs covering wrist, elbow, shoulder, forearm, hand, humerus, and finger).

---

## 🚀 Key Features
- **Instant Abnormality Classification:** Predicts whether an input radiograph is normal or abnormal with confidence scores.
- **Interactive Web Interface:** Upload custom X-ray images (`.png`, `.jpg`, `.jpeg`) directly via Streamlit.
- **Reproducible Pipeline:** Clean notebook containing preprocessing, data augmentation, training curves, and metric evaluation.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Image Processing:** Pillow, OpenCV, NumPy
* **Deployment:** Streamlit Community Cloud

---

## 💻 Local Setup & Installation

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/Yoshi1710/mura-abnormality-detector.git
   cd mura-abnormality-detector
   \`\`\`

2. **Create and activate a virtual environment:**
   \`\`\`bash
   python -m venv myenv
   # Windows:
   myenv\Scripts\activate
   \`\`\`

3. **Install dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. **Run the Streamlit app:**
   \`\`\`bash
   streamlit run app.py
   \`\`\`

## 🔮 Future Enhancements

This project establishes a strong baseline for AI-assisted radiology. The roadmap for future iterations includes:

* **Automated Diagnostic Reporting (NLP):** Integrating Large Language Models (LLMs) to automatically generate comprehensive medical reports based on visual findings, assisting doctors with instant documentation.
* **Multi-Class Pathology Detection:** Expanding the architecture from binary screening to multi-class detection, specifically identifying various conditions like arthritis, bone tumors, and specific fracture types.
* **Advanced Vision Models:** Upgrading the underlying architecture to Vision Transformers (ViTs) to achieve even more precise spatial localization and higher diagnostic accuracy across complex bone structures.
---

## 👤 Author
* **Deepak Singh Bisht** - [GitHub Profile](https://github.com/Yoshi1710)