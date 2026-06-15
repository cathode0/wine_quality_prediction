# Wine Quality Prediction System

A modular machine learning project developed for **SF35803 Computer Programming**. This project predicts wine quality based on physicochemical properties using a Logistic Regression classifier. The original machine learning notebook was decomposed into a structured software project following software engineering best practices.

---

# Project Overview

The goal of this project is to classify wine samples as either:

* **Good Quality Wine (1)** → Quality score ≥ 6
* **Low Quality Wine (0)** → Quality score < 6

using chemical measurements such as:

* Fixed Acidity
* Volatile Acidity
* Citric Acid
* Residual Sugar
* Chlorides
* Free Sulfur Dioxide
* Total Sulfur Dioxide
* Density
* pH
* Sulphates
* Alcohol

The project demonstrates:

* Data Loading
* Exploratory Data Analysis (EDA)
* Data Visualization
* Data Preprocessing
* Machine Learning Model Training
* Model Evaluation
* Software Modularization
* Unit Testing

---

# Project Structure

```text
wine_quality_prediction/
│
├── main.py
├── config.py
├── run_all_tests.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── winequality.csv
│   └── data_loader.py
│
├── preprocessing/
│   └── preprocess.py
│
├── models/
│   └── logistic_model.py
│
├── evaluation/
│   └── evaluate.py
│
├── visualization/
│   └── plots.py
│
├── outputs/
│   ├── Predictions.csv
│   └── figures/  (optional - visualizations displayed, not saved)
│
└── tests/
    ├── test_data_loader.py
    ├── test_preprocessing.py
    ├── test_model.py
    └── test_evaluation.py

Features
Data Analysis
The system performs:

Dataset inspection (1599 samples, 12 columns)

Missing value detection (0 missing values)

Statistical summaries (mean, std, min, max, quartiles)

Quality distribution analysis

Dataset Statistics
Total samples: 1599 wine samples

Features: 11 physicochemical properties

Target: Quality scores (3-8)

Data types: float64(11), int64(1)

Missing values: None

Quality Distribution
Quality Score	Count
5	681
6	638
7	199
4	53
8	18
3	10
Data Visualization
The system generates interactive visualizations (displayed, not saved):

1. Correlation Heatmap
Shows relationships between all 11 chemical properties and quality.

2. Histograms
Displays distribution of each feature (12 histograms with 20 bins each).

3. Boxplots
Compares selected features against wine quality:

Sulphates vs Quality

Alcohol vs Quality

Citric Acid vs Quality

Fixed Acidity vs Quality

Machine Learning
Algorithm Used
Logistic Regression from scikit-learn

Why Logistic Regression?
Simple and interpretable

Efficient for binary classification

Fast training (max 2000 iterations)

Suitable baseline model

Provides probability outputs

Data Split Configuration
python
TEST_SIZE = 0.35      # 35% for testing, 65% for training
RANDOM_STATE = 100    # Ensures reproducible results
MAX_ITER = 2000       # Maximum iterations for convergence
Training samples: 1039 (65% of data)

Testing samples: 560 (35% of data)

Evaluation Metrics
The model reports:

Confusion Matrix - Shows correct/incorrect classifications

Accuracy - Overall correct predictions

Precision - Accuracy of positive predictions

Recall - Ability to find positive samples

F1 Score - Harmonic mean of precision and recall

Actual Model Performance
text
Confusion Matrix:
[[187  71]
 [ 66 236]]

Accuracy: 0.7554 (75.54%)

Classification Report:
              precision    recall  f1-score   support

           0       0.74      0.72      0.73       258
           1       0.77      0.78      0.78       302

    accuracy                           0.76       560
   macro avg       0.75      0.75      0.75       560
weighted avg       0.76      0.76      0.76       560
Performance Interpretation
Overall Accuracy: 75.5% of wines correctly classified

Good Wine Detection (Class 1):

Precision: 77% - When model predicts "good wine", it's correct 77% of the time

Recall: 78% - Model successfully identifies 78% of actual good wines

Poor Wine Detection (Class 0):

Precision: 74% - When model predicts "poor wine", it's correct 74% of the time

Recall: 72% - Model identifies 72% of actual poor wines

Balanced Performance: Similar metrics across both classes (no significant bias)

Module Documentation
config.py
Purpose
Central configuration for the entire project.

Configuration Variables
python
DATA_PATH = "data/winequality.csv"    # Dataset location
TEST_SIZE = 0.35                       # Test split ratio
RANDOM_STATE = 100                     # Reproducibility seed
MAX_ITER = 2000                        # Logistic regression iterations
data/data_loader.py
Purpose
Loads wine dataset from CSV file.

Function
python
def load_data():
    """Loads dataset from DATA_PATH"""
    return pd.read_csv(DATA_PATH)
Returns
pandas.DataFrame with 1599 rows and 12 columns

Example
python
df = load_data()
print(df.shape)  # (1599, 12)
preprocessing/preprocess.py
Purpose
Prepares data for model training.

Functions
python
def create_target(df):
    """
    Convert wine quality to binary classification.
    Good (1) if quality >= 6, else Bad (0)
    """
    df = df.copy()
    df['response'] = np.where(df['quality'] >= 6, 1, 0)
    return df
python
def split_features(df):
    """
    Split dataset into features (X) and target (y).
    Drops 'quality' and 'response' columns for features.
    """
    X = df.drop(columns=['quality', 'response'])
    y = df['response']
    return X, y
models/logistic_model.py
Purpose
Train and use Logistic Regression model.

Functions
python
def train_model(X_train, y_train):
    """Trains logistic regression model with MAX_ITER iterations"""
    model = LogisticRegression(max_iter=MAX_ITER)
    model.fit(X_train, y_train)
    return model
python
def predict(model, X_test):
    """Returns probability predictions for class 1"""
    return model.predict_proba(X_test)[:, 1]
Note
Predictions return probabilities (0-1 range), not binary classes.

evaluation/evaluate.py
Purpose
Measures model performance.

Function
python
def evaluate(y_test, y_pred):
    """Prints confusion matrix, accuracy, and classification report"""
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
Output Metrics
Confusion Matrix (TN, FP, FN, TP)

Accuracy Score (0.0 to 1.0)

Precision, Recall, F1-score for each class

Support (number of samples per class)

visualization/plots.py
Purpose
Generates data visualizations for EDA.

Functions
python
def correlation_plot(df):
    """Creates heatmap showing correlations between features"""
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
python
def histograms(df):
    """Creates histograms for all features (12 plots, 20 bins each)"""
    df.hist(figsize=(12, 10), bins=20)
    plt.tight_layout()
    plt.show()
python
def boxplots(df, features):
    """Creates boxplots comparing specified features against quality"""
    for feature in features:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='quality', y=feature, data=df)
        plt.title(f"{feature} vs Quality")
        plt.show()
Visualized Features
sulphates

alcohol

citric acid

fixed acidity
