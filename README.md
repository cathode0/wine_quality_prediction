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
* Data Preprocessing
* Machine Learning Model Training
* Model Evaluation
* Data Visualization
* Software Modularization
* Unit Testing

---

# Project Structure

```text
wine_quality_prediction/
│
├── main.py
├── config.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── winequality-red.csv
│   └── data_loader.py
│
├── preprocessing/
│   └── preprocessing.py
│
├── models/
│   └── logistic_regression_model.py
│
├── evaluation/
│   └── evaluation.py
│
├── visualization/
│   └── visualization.py
│
├── utils/
│   └── file_handler.py
│
├── outputs/
│   ├── predictions.csv
│   └── figures/
│
└── tests/
    ├── test_data_loader.py
    ├── test_preprocessing.py
    ├── test_model.py
    └── test_evaluation.py
```

---

# Features

## Data Analysis

The system performs:

* Dataset inspection
* Missing value detection
* Statistical summaries
* Correlation analysis

---

## Data Visualization

Generated visualizations include:

### Correlation Heatmap

Shows relationships between variables.

### Histograms

Displays feature distributions.

### Boxplots

Compares important features against wine quality.

Features visualized:

* Sulphates
* Alcohol
* Citric Acid
* Fixed Acidity

---

## Machine Learning

### Algorithm Used

Logistic Regression

### Why Logistic Regression?

* Simple and interpretable
* Efficient for binary classification
* Fast training
* Suitable baseline model

---

## Evaluation Metrics

The model reports:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

# Installation Guide

## Clone Repository

```bash
git clone https://github.com/yourusername/wine_quality_prediction.git
```

```bash
cd wine_quality_prediction
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

### Linux / MacOS

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Packages

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
pytest
```

---

# Running the Project

Execute:

```bash
python main.py
```

The application will:

1. Load dataset
2. Display dataset information
3. Generate visualizations
4. Preprocess data
5. Train model
6. Evaluate model
7. Save predictions

---

# Example Output

```text
Training samples: 1039
Testing samples: 560

Confusion Matrix:
[[180  52]
 [ 41 287]]

Accuracy:
0.8339
```

---

# Dataset Information

Dataset:

**Wine Quality Dataset**

Source:

UCI Machine Learning Repository

Dataset contains:

* 1599 wine samples
* 11 input features
* 1 target variable (quality)

Target quality scores range from:

```text
3 – 8
```

For this project:

```python
quality >= 6
```

is converted to:

```python
1 = Good Wine
```

and

```python
0 = Poor Wine
```

---

# Module Documentation

## data_loader.py

### Purpose

Loads wine dataset from CSV file.

### Functions

```python
load_data(filepath)
```

### Returns

```python
pandas.DataFrame
```

---

## preprocessing.py

### Purpose

Prepare data for training.

### Functions

```python
create_target_variable()
```

```python
split_data()
```

### Responsibilities

* Feature selection
* Target creation
* Train-test split

---

## logistic_regression_model.py

### Purpose

Train and use Logistic Regression model.

### Functions

```python
train_model()
```

```python
predict()
```

```python
predict_probability()
```

---

## evaluation.py

### Purpose

Measure model performance.

### Functions

```python
evaluate_model()
```

Outputs:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## visualization.py

### Purpose

Generate plots.

### Functions

```python
plot_correlation_matrix()
```

```python
plot_histograms()
```

```python
plot_boxplots()
```

---

## file_handler.py

### Purpose

Manage file operations.

### Functions

```python
save_predictions()
```

Outputs:

```text
Predictions.csv
```

---

# Object-Oriented Programming Implementation

The project demonstrates key OOP concepts.

## Encapsulation

Data and methods are grouped within classes.

Example:

```python
class WineModel:
    def train(self):
        pass
```

---

## Abstraction

Complex machine learning operations are hidden behind simple method calls.

Example:

```python
model.train()
```

---

## Inheritance

Future machine learning models can inherit from a common base class.

Example:

```python
class BaseModel:
    pass

class LogisticRegressionModel(BaseModel):
    pass
```

---

## Polymorphism

Different model classes can implement identical interfaces.

Example:

```python
model.predict()
```

works regardless of the underlying algorithm.

---

# Unit Testing

Testing is implemented using:

```text
pytest
```

Run tests:

```bash
pytest
```

---

## Test Coverage

### Data Loader Tests

* File loading
* Error handling

### Preprocessing Tests

* Target variable creation
* Data splitting

### Model Tests

* Successful training
* Prediction generation

### Evaluation Tests

* Accuracy calculation
* Confusion matrix generation
