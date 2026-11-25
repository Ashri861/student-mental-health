# Student Mental Wellness Risk Predictor

A machine learning web app that predicts a student's risk of depression from survey responses and points them to nearby psychologists and counsellors.

## Overview

The project cleans the Kaggle *Student Depression Dataset*, trains and compares three classification models, and serves the best-suited model through a Flask web form. A student enters their details, receives a risk prediction, and can search for support in their city.

> **Disclaimer:** This is an academic project and a screening aid only. It is not a medical diagnosis. Anyone struggling should speak to a qualified mental health professional.

## Dataset

- **Source:** Kaggle - Student Depression Dataset
- **Target:** `Depression` (1 = at risk, 0 = not at risk)
- **Features used:** Gender, Age, Academic Pressure, Work Pressure, CGPA, Study Satisfaction, Job Satisfaction, Sleep Duration, Dietary Habits, Degree, Suicidal Thoughts, Work/Study Hours, Financial Stress, Family History of Mental Illness
- **Dropped:** `id`, `City`, `Profession` (no predictive value for this task)

## Project Structure

```
student_mental_health/
├── Student Depression Dataset.csv   # raw data
├── preprocessing.py                 # cleaning, encoding, scaling, train/test split
├── model.py                         # model training, comparison, evaluation charts
├── app.py                           # Flask backend
├── templates/
│   └── index.html                   # web form + result page
├── requirements.txt
└── (generated) *.pkl, *.png         # saved model, scaler, encoders, charts
```

## How It Works

1. **Preprocessing** - removes unneeded columns and missing rows, label-encodes categorical columns, standardises features with `StandardScaler`, and splits the data 80/20 into train and test sets. The scaler, encoders, and column order are saved so the web app can apply the exact same transformations to new input.
2. **Modelling** - trains Decision Tree, Gradient Boosting, and Random Forest classifiers and compares them. Evaluation includes accuracy, F1-score, a confusion matrix, an ROC curve, and a feature importance plot.
3. **Web app** - the user fills in the form, the input is encoded and scaled the same way as the training data, and the saved Random Forest model returns the prediction. A "Find Support Near You" section opens a Google Maps search for psychologists and counsellors in the user's city.

## Results

| Model | Accuracy | F1-Score |
|---|---|---|
| Decision Tree | 0.78 | 0.81 |
| Gradient Boosting | 0.85 | 0.87 |
| Random Forest | 0.84 | 0.87 |

**Why Random Forest?** Gradient Boosting scored about 1% higher, but Random Forest trains faster, is less prone to overfitting, and is easier to explain and interpret through feature importance.

**Top predictors:** suicidal thoughts, academic pressure, CGPA, and financial stress.

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn flask
```

## Usage

Run the scripts in this order from the project folder:

```bash
python preprocessing.py   # prepares data and saves scaler/encoders
python model.py           # trains models and saves rf_model.pkl
python app.py             # starts the web app
```

Then open **http://127.0.0.1:5000** in your browser.

## Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Flask, HTML/CSS

## Limitations and Future Work

- Trained on survey data from one dataset, so results may not generalise to all student populations.
- Predictions are probabilistic and should never replace professional assessment.
- Possible improvements: hyperparameter tuning, cross-validation, SHAP explanations, and integrating the Google Places API for in-page counsellor listings.
