# ❤️ Heart Disease Prediction Using Machine Learning

This project is focused on predicting the presence of heart disease using machine learning algorithms, specifically **Support Vector Machine (SVM)** and **Logistic Regression**. The system is designed to assist in early diagnosis and support decision-making in the healthcare sector.

## 📌 Project Overview

Heart disease is one of the major causes of death globally. Early detection is essential to reduce the mortality rate. Traditional diagnostic methods are often time-consuming, expensive, and inaccessible in rural areas. This project aims to automate heart disease prediction using machine learning, making it faster, affordable, and accessible.

## 🎯 Objectives

- Build a machine learning model to predict heart disease from medical data.
- Evaluate and compare the performance of SVM and Logistic Regression algorithms.
- Identify important risk factors contributing to heart disease.
- Provide visual analysis and insights from the model’s predictions.
- Make healthcare prediction accessible via a software system.

## 🧠 Algorithms Used

### 1. Logistic Regression
- Supervised learning algorithm used for binary classification.
- Calculates the probability that an instance belongs to a specific class (disease/no disease).
- Simple and interpretable model with fast training time.

### 2. Support Vector Machine (SVM)
- Constructs hyperplanes to separate data into classes.
- Effective for high-dimensional spaces.
- Works well when data is clean and features are normalized.
- Suitable for both linear and non-linear classification.

## 🔍 Dataset

- **Source**: UCI Machine Learning Repository – Heart Disease Dataset
- **Features include**:
  - Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol
  - Fasting Blood Sugar, ECG, Max Heart Rate
  - Exercise-induced Angina, Oldpeak, Slope, Vessels, Thal
  - Target (1 = Heart Disease, 0 = No Heart Disease)

## ⚙️ Methodology

1. **Data Collection** – Patient data is loaded from a CSV dataset.
2. **Data Preprocessing** – Clean data, handle missing values, and normalize features.
3. **Train-Test Split** – Dataset is split into training and testing subsets.
4. **Model Training** – Train SVM and Logistic Regression using the training set.
5. **Model Evaluation** – Evaluate using accuracy, confusion matrix, and visualizations.
6. **Visualization** – Graphs such as pie charts and bar graphs are used to show data and model performance.

## 💻 Implementation Details

- **Language**: Python / MATLAB (2023a)
- **Libraries**: scikit-learn, pandas, matplotlib, seaborn
- **System Specs**:
  - Intel Core i5
  - 16 GB RAM
  - Super GPU, 3.60 GHz

## 📈 Results

- Models tested on the dataset showed high accuracy.
- **SVM Accuracy**: ~90.4%
- Logistic Regression also provided competitive results with high precision and recall.

## 🖼️ System Flow

