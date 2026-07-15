# 🩺 Breast Cancer Diagnosis Prediction

A Machine Learning web application that predicts whether a breast tumor is **Benign** or **Malignant** using the Breast Cancer Wisconsin Dataset.

Built using **Python**, **Scikit-learn**, and **Streamlit**.

---

## 📌 Project Overview

Early diagnosis of breast cancer significantly increases the chances of successful treatment. This project uses Machine Learning to classify tumors into:

- 🟢 Benign
- 🔴 Malignant

The application accepts medical measurements of a tumor and instantly predicts the diagnosis.

---

## Features

- Data preprocessing
- Feature Selection (Top 10 important features)
- Logistic Regression Model
- StandardScaler
- Interactive Streamlit UI
- Real-time Predictions
- Model Serialization using Joblib

---

## Dataset

**Breast Cancer Wisconsin Diagnostic Dataset**

Dataset contains:

- 569 samples
- 30 numerical features
- Target:
  - M = Malignant
  - B = Benign

---

## Selected Features

The model uses the following top 10 features:

- perimeter_worst
- area_worst
- radius_worst
- concave_points_worst
- concave_points_mean
- perimeter_mean
- area_mean
- radius_mean
- area_se
- concavity_mean

---

## Machine Learning Pipeline

Dataset
↓

Data Cleaning
↓

Label Encoding
↓

Feature Selection
↓

Train-Test Split
↓

Standard Scaling
↓

Logistic Regression
↓

Model Evaluation
↓

Save Model (Joblib)
↓

Streamlit Deployment

---

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Breast-Cancer-Prediction.git

cd Breast-Cancer-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Project Structure

```
Breast-Cancer-Prediction/
│
├── app.py
├── model_training.py
├── model.pkl
├── scalar.pkl
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
└── dataset.csv
```

---

## Model Performance

| Model | Accuracy |
|---------|----------|
| Logistic Regression | ~97–99% |

---

## Future Improvements

- Random Forest
- XGBoost
- Hyperparameter Tuning
- Explainable AI (SHAP)
- Docker Deployment
- AWS EC2 Deployment
- CI/CD using GitHub Actions

---

## Screenshots

### Home Page

<img width="1913" height="997" alt="image" src="https://github.com/user-attachments/assets/210c0261-f40c-4a01-b9fd-80a864b93317" />


### Prediction Result

<img width="1817" height="887" alt="image" src="https://github.com/user-attachments/assets/ac7ef6dd-9a8f-4761-a48c-a348280f4ea9" />

---

## Author

**Prakash Dora**

GitHub: https://github.com/Prakash563

LinkedIn: https://linkedin.com/in/prakashdora

---

## License

This project is licensed under the MIT License.
