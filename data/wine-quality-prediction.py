import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load dataset
# Update the path if needed
wine_data = pd.read_csv('winequality-red.csv')

print("First 5 rows:")
print(wine_data.head())

print("\nDataset info:")
print(wine_data.info())

print("\nSummary statistics:")
print(wine_data.describe())

print("\nQuality distribution:")
print(wine_data['quality'].value_counts())

print("\nMissing values:")
print(wine_data.isnull().sum())

# Correlation matrix
plt.figure(figsize=(10,8))
corr = wine_data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Histograms
wine_data.hist(figsize=(12,10), bins=20)
plt.tight_layout()
plt.show()

# Boxplots for selected features
features = ['sulphates', 'alcohol', 'citric acid', 'fixed acidity']

for feature in features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='quality', y=feature, data=wine_data)
    plt.title(f"{feature} vs Quality")
    plt.show()

# Create binary response variable
wine_data['response'] = np.where(wine_data['quality'] >= 6, 1, 0)

# Features and target
X = wine_data.drop(columns=['quality', 'response'])
y = wine_data['response']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=100
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Logistic Regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Predictions
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save predictions
submission = pd.DataFrame({
    'Quality': y_pred_prob
})

submission.to_csv('Predictions.csv', index=False)
print("\nPredictions saved to Predictions.csv")
