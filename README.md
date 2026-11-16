# ARTVISION – Artwork Style Classification Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%20--%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey.svg)](https://sqlite.org/)

An intelligent academic Computer Vision and Deep Learning system that classifies fine art paintings into four foundational art movements using **Transfer Learning (MobileNetV2 / EfficientNetB0)**, and provides multi-disciplinary recommendations across **cultural heritage, spatial interior design, fine art learning, and commercial advertising**.

---

## 1. Project Introduction

Art history curation and stylistic attribution traditionally demand extensive scholarly expertise to discern subtle brushwork rhythms, chromatic palettes, and compositional structures. With the global digitization of art collections across institutions such as WikiArt and Google Arts & Culture, machine learning provides an automated, scalable approach to cataloging fine art.

**ARTVISION** implements a deep convolutional neural network pipeline coupled with a relational knowledge base and color clustering engine to:
1. Accurately categorize fine art paintings into **Impressionism, Cubism, Abstract, and Surrealism**.
2. Preserve and disseminate cultural history, techniques, and landmark works.
3. Analyze room and interior photographs using **OpenCV K-Means color clustering** to recommend harmonious artwork styles, framing profiles, and lighting.
4. Deliver tiered studio art pedagogy (**Beginner, Intermediate, Advanced**) for practicing artists.
5. Map art movements to **commercial brand archetypes, color psychology, and marketing campaign themes**.

---

## 2. Project Objectives

- **Deep Learning Image Classification:** Implement transfer learning with MobileNetV2 (pretrained on ImageNet) to extract deep visual features from artwork images.
- **Robust Preprocessing Pipeline:** Standardize inputs to $224 \times 224 \times 3$, apply Lanczos resampling, normalize pixel intensities to $[-1, 1]$, and employ real-time data augmentation (rotation, zoom, horizontal flips).
- **Relational Cultural Database:** Store verified academic records for all four art movements in an SQLite database (`database/artwork.db`).
- **Spatial Interior Pairing:** Use OpenCV K-Means to extract 5 dominant colors from room photos, assess warm/cool chromatic distribution and luminosity, and pair the space with matching art styles.
- **Studio Art Pedagogy:** Formulate structured curriculum spanning materials, mechanical brushwork drills, and master copy exercises.
- **Commercial Brand Intelligence:** Translate artistic philosophy into consumer psychology, market demographics, and advertising slogans.
- **Interactive Web Interface:** Build an academic dashboard using Streamlit with custom art-inspired styling.

---

## 3. Technology Stack

| Category | Technologies | Purpose |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.10 – 3.12 (Recommended) | Core system development |
| **Deep Learning** | TensorFlow / Keras | CNN Transfer Learning (MobileNetV2 / EfficientNetB0) |
| **Computer Vision** | OpenCV, Pillow | Image validation, resizing, color space transformations, K-Means |
| **Data Processing** | NumPy, Pandas | Tensor manipulation and array operations |
| **Evaluation & Metrics** | Scikit-learn, Matplotlib, Seaborn | Confusion matrix, precision, recall, F1-score, loss/accuracy curves |
| **Database** | SQLite3 | Local relational storage for cultural, interior, learning, and ad data |
| **User Interface** | Streamlit | Modern interactive multi-page web dashboard |

---

## 4. Project Folder Structure

```text
artwork_style_classification/
│
├── app.py                         # Streamlit primary dashboard application
├── requirements.txt               # Project Python package dependencies
├── README.md                      # Academic documentation and user manual
│
├── dataset/                       # Dataset directory partitioned for ML
│   ├── train/                     # 70% Training set (Impressionism, Cubism, Abstract, Surrealism)
│   ├── validation/                # 15% Validation set
│   ├── test/                      # 15% Test set
│   └── sample_data/               # Curated test artwork and room BMP images
│
├── models/                        # Serialized Keras Deep Learning models
│   └── artwork_classifier.keras   # Trained MobileNetV2 transfer learning model
│
├── src/                           # Backend computational & analytical modules
│   ├── __init__.py
│   ├── database.py                # SQLite schema definition, seeding, and queries
│   ├── preprocessing.py           # Resizing (224x224), [-1, 1] normalization, validation
│   ├── train_model.py             # Transfer learning training pipeline with callbacks
│   ├── predict.py                 # Neural inference, softmax probability calculation
│   ├── evaluation.py              # Confusion matrix and classification report generator
│   ├── interior_recommendation.py # OpenCV K-Means color clustering and aesthetic scoring
│   ├── dataset_setup.py           # Dataset verification, WikiArt acquisition guide, splitter
│   └── sample_images.py           # Standard library procedural sample image generator
│
├── database/                      # Relational database storage
│   └── artwork.db                 # Seeded SQLite database file
│
├── modules/                       # Streamlit UI page components
│   ├── __init__.py
│   ├── home.py                    # Dashboard, objectives, style showcase, system telemetry
│   ├── classification.py          # Painting uploader, sample selector, neural predictions
│   ├── cultural_heritage.py       # Art history, techniques, master artists, and works
│   ├── interior_design.py         # Room photo uploader, palette swatches, framing advice
│   ├── skill_guidance.py          # Beginner, Intermediate, and Advanced art curriculum
│   ├── advertising.py             # Brand archetypes, color psychology, and marketing themes
│   └── about.py                   # Academic documentation, architecture, and manual
│
├── assets/                        # Static styling and image assets
│   ├── css/
│   │   └── style.css              # Custom art-inspired design system
│   └── images/                    # Local artwork and room preview assets
│
├── reports/                       # Empirical evaluation artifacts
│   ├── accuracy.png               # Training and validation accuracy curves
│   ├── loss.png                   # Training and validation loss curves
│   ├── confusion_matrix.png       # Test set confusion matrix heatmap
│   └── classification_report.json # Per-class precision, recall, and F1 metrics
│
└── tests/                         # Automated unit testing suite
    ├── __init__.py
    └── test_pipeline.py           # 13 Automated test cases for database, vision, and logic
```

---

## 5. Dataset Management & Acquisition

The system categorizes paintings into four styles:
1. **Impressionism:** Monet, Renoir, Degas, Pissarro (visible brushstrokes, natural light).
2. **Cubism:** Picasso, Braque, Gris (multi-perspective geometric planar deconstruction).
3. **Abstract:** Kandinsky, Mondrian, Pollock (non-objective form, pure color fields).
4. **Surrealism:** Dalí, Magritte, Ernst (dreamscapes, irrational juxtaposition).

### Acquisition Guide (WikiArt / Kaggle)
1. **Using Kaggle CLI:**
   ```bash
   pip install kaggle
   kaggle datasets download -d steubk/wikiart
   ```
2. **Manual Download:**
   - [Kaggle WikiArt Dataset](https://www.kaggle.com/datasets/steubk/wikiart)
   - [Impressionist Classifier Dataset](https://www.kaggle.com/datasets/delayedkarma/impressionist-classifier-data)

3. **Automatic Dataset Partitioning:**
   Place the extracted style folders into `dataset/raw/` and execute:
   ```bash
   python src/dataset_setup.py --split-raw dataset/raw
   ```
   This splits images into 70% Train, 15% Validation, and 15% Test.

4. **Instant Offline Sample Testing:**
   ARTVISION includes a standalone sample image generator that requires zero internet access:
   ```bash
   python src/sample_images.py
   python src/dataset_setup.py --populate-demo
   ```

---

## 6. Installation & Execution Guide

### Prerequisites
- Python 3.10 – 3.12 is recommended for pre-built TensorFlow wheels on Windows.
- Git (optional).

### Step 1: Navigate to Project Directory
```powershell
cd C:\Users\vh566.LAPTOP-U33RR19L\.gemini\antigravity\scratch\artwork_style_classification
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Initialize Database
```powershell
python src/database.py
```
*Creates `database/artwork.db` and populates detailed cultural, interior, learning, and advertising records.*

### Step 4: Run Automated Verification Tests
```powershell
python -m unittest tests/test_pipeline.py
```
*Verifies database tables, OpenCV/Pillow color clustering, preprocessing, and error handling (13 tests).*

### Step 5: Train the Deep Learning Model
```powershell
python src/train_model.py --epochs 12 --batch-size 16 --model-type mobilenet_v2
```
*Loads ImageNet weights, trains classification layers, generates `reports/accuracy.png`, `reports/loss.png`, `reports/confusion_matrix.png`, and saves `models/artwork_classifier.keras`.*

### Step 6: Launch the Streamlit Application
```powershell
streamlit run app.py
```
The browser interface opens automatically at `http://localhost:8501`.

---

## 7. Application Modules Overview

### Module 1: Image Input Module
- Upload painting files (JPG, JPEG, PNG, WEBP, BMP) up to 25MB.
- Upload room/interior photographs.
- 1-click curated test artwork and room selectors for instant demonstration.

### Module 2: Image Preprocessing Module
- Validates dimensions ($\ge 32 \times 32$), formats, and file integrity.
- Resizes images via high-fidelity Lanczos interpolation to $224 \times 224 \times 3$.
- Maps RGB pixel values to $[-1.0, 1.0]$ matching MobileNetV2 standards.

### Module 3: Artwork Classification Module
- Neural forward pass produces raw logits and softmax probabilities across all 4 styles.
- Displays predicted movement, confidence score, and distribution progress bars.
- Automatically logs inference history into the SQLite database.
- Provides immediate shortcut buttons to inspect cultural, interior, learning, and advertising insights for the classified style.

### Module 4: Cultural and Heritage Module
- Retrieves academic historical context, time periods, and geographic origins.
- Displays revolutionary techniques (e.g. wet-on-wet *alla prima*, planar faceting, psychic automatism).
- Lists prominent artists, iconic masterworks, and chromatic philosophies.

### Module 5: Interior Design Recommendation Module
- Extracts 5 dominant colors from room photos via OpenCV K-Means clustering.
- Generates visual color swatches with hex codes, percentages, and temperature labels.
- Calculates spatial chromatic balance (warm vs. cool vs. neutral, saturation, and luminosity).
- Recommends primary and secondary art styles, framing profiles, and lighting conditions.

### Module 6: Skill-Based Learning Guidance Module
- Three structured difficulty tiers: **Beginner**, **Intermediate**, and **Advanced**.
- Specifies learning objectives, required materials (brushes, paints, grounds), mechanical exercises, daily improvement drills, and common pitfalls.

### Module 7: Advertising Suggestion Module
- Aligns art styles with Carl Jung brand archetypes (e.g., *The Creator*, *The Visionary*, *The Innocent*).
- Identifies prime industry verticals (Fintech, Luxury Travel, Architecture, Gaming).
- Outlines consumer emotional impact, color psychology, and creative campaign themes.

### Module 8: User Interface Module
- Responsive, modern art-inspired UI built in Streamlit.
- Persistent session state across pages.
- Dynamic telemetry sidebar monitoring model readiness, dataset size, and database health.

---

## 8. Error Handling & Robustness

ARTVISION incorporates defensive engineering:
- **Missing Dataset:** Automatically falls back to setup status with step-by-step download guidance (`python src/dataset_setup.py --guide`).
- **Missing Model:** Displays an informative warning banner in the UI and permits setup mode exploration rather than throwing unhandled exceptions.
- **Corrupted / Invalid Images:** Catches file format errors, undersized images, and empty streams via `validate_image_file()`.
- **Hardware Agnostic:** Automatically utilizes GPU acceleration when CUDA is available, or defaults to optimized CPU threads.

---

## 9. Academic Project Verification Checklist

- [x] Full source code written and organized in structured project directory.
- [x] SQLite database schema initialized, seeded with 4 styles, and verified.
- [x] MobileNetV2 transfer learning training pipeline implemented with callbacks.
- [x] Real-time image validation and preprocessing ($224 \times 224$, $[-1, 1]$).
- [x] OpenCV K-Means color clustering for room interior aesthetics.
- [x] Multi-tier skill learning guidance and advertising suggestion engines.
- [x] 7-page interactive Streamlit dashboard designed and connected.
- [x] 13 automated unit tests executed and passed (`OK`).
- [x] Detailed documentation and step-by-step execution commands.
