# Heart Disease Prediction using Python & Machine Learning

## 📌 Project Overview

This project uses Python and Machine Learning to predict the presence of heart disease using the Cleveland Heart Disease dataset.

The project covers the complete machine learning workflow, including data loading, data preprocessing, missing-value handling, feature-target separation, train-test splitting, model training, model comparison, hyperparameter experiments, cross-validation, feature importance visualization, and final model evaluation.

---

## 🎯 Objective

The main objectives of this project are:

- Clean and preprocess the heart disease dataset
- Convert the target variable into a binary classification problem
- Split the data into training and testing datasets
- Train multiple classification models
- Compare model performance
- Experiment with Random Forest and XGBoost parameters
- Apply 5-Fold Cross-Validation
- Analyze feature importance
- Evaluate the final Random Forest model using multiple classification metrics

---

## 📊 Dataset

The project uses the *Cleveland Heart Disease dataset* from the UCI Machine Learning Repository.

The dataset contains medical and demographic features related to heart disease.

### Features Used

| Feature | Description |
|---|---|
| age | Age of the patient |
| sex | Sex |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels |
| thal | Thalassemia |
| target | Heart disease target |

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Assigned meaningful column names.
3. Replaced "?" values with missing values.
4. Converted ca and thal columns to numeric format.
5. Filled missing values in ca and thal using their median values.
6. Converted the target variable to integer format.
7. Converted the original target into a binary classification:
   - 0 → No heart disease
   - 1 → Heart disease
8. Separated features (X) and target (y).

---

## 🔀 Train-Test Split

The dataset was divided into:

- *80% Training Data*
- *20% Testing Data*

The split uses:

- random_state = 42
- stratify = y

This helps maintain the target-class distribution between training and testing data.

---

## 🤖 Machine Learning Models

The following classification models were implemented:

### 1. Decision Tree

A Decision Tree classifier was trained and evaluated using training and testing accuracy.

### 2. Shallow Decision Tree

A second Decision Tree was created with:

```text
max_depth = 4
