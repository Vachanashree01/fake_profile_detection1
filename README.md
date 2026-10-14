# 🛡️ AI Social Network Fake Profile Detection System

This is an advanced Artificial Intelligence and Machine Learning academic team project designed to analyze structural metadata anomalies and identify fraudulent or automated bot profiles across social media platforms.

---

## 🚀 Core Project Architecture & Pipeline

Our classification framework uses a modular data science architecture built entirely within **VS Code**:

1. **Data Ingestion Engine:** Parses raw tabular profiles directly from our curated repository (`data/social_media_profiles.csv`).
2. **Advanced Feature Engineering:** Rather than relying on simple metadata checks, the engine dynamically computes custom operational variables:
   * **Network Dispersion Imbalance Ratio:** Followers ÷ (Following + 1) to catch hyper-aggressive industrial follow-bots.
   * **Syntactic Scripting Ratio:** Detects algorithmic random text streams inside usernames.
3. **Machine Learning Processing Layer:** Evaluates the engineered matrices utilizing a tuned **Random Forest Ensemble Classifier** with balanced class distributions.
4. **Interactive Dashboard Middleware:** A clean local development server powered by **Flask** that handles input queries and renders authenticity badges in real time.

---

## 📂 Repository File Blueprint

```text
fake_profile_detection1/
├── data/
│   └── social_media_profiles.csv   # Unified Kaggle training profile data
├── static/
│   └── style.css                   # Cyber-security visual theme styles
├── templates/
│   └── index.html                  # Front-end query input dashboard
├── app.py                          # Live Flask backend web router script
├── train.py                        # Automated ML model compiling pipeline
├── requirements.txt                # Enforced dependency libraries manifest
└── fake_profile_model.pkl          # Serialized classification intelligence engine
```

---

## 🛠️ Step-by-Step Workspace Setup & Execution

Follow these operations inside your terminal to initialize the architecture:

### 1. Initialize System Library Dependencies
Ensure your environment satisfies the package manifest variables:
```bash
pip install -r requirements.txt
```

### 2. Compile and Serialize the ML Model Engine
Execute the standalone pipeline script to process the training database, balance sample distributions, and export the binary classifier file:
```bash
python train.py
```

### 3. Launch the Interface Server
Fire up the Flask microservice controller to bridge your front-end viewports with the saved machine learning model:
```bash
python app.py
```
*Open your web browser and navigate to `http://127.0.0.1:5000` to interact with the operational diagnostic engine layout live.*
