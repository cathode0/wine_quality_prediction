# Split data into training and testing sets
from sklearn.model_selection import train_test_split

# Pandas for data manipulation
import pandas as pd

# Configuration setting
import config

# Read wine dataset
from data.data_loader import load_data

# create_target: Add response to column
# split_feautures: Separate x from y
from preprocessing.preprocess import create_target, split_features

# train_model: Creates and trains LogisticRegression
# predict: Returns probability of wine bein good
from models.logistic_model import train_model, predict

# Print confusion matrix, accuracy, and clasification report
from evaluation.evaluate import evaluate

# Create visualization to understand patterns
from visualization.plots import correlation_plot, histograms, boxplots

# 1. Load data
df = load_data() # Loads 1599 wine samples with 11 chemical feautures + quality score

print(df.head()) # Show 5 first data row
print(df.info()) # Show data types and non-null counts
print(df.describe()) # Show statistical summary (meann, std, min, max, quartiles)

print(df['quality'].value_counts()) # Distribution of quality scores
print(df.isnull().sum()) # Checks for missing values

# 2. Visualization
correlation_plot(df) # Heatmap
histograms(df) # Histograms

boxplots(df, ['sulphates', 'alcohol', 'citric acid', 'fixed acidity']) # SHow correlations against quality

# 3. Preprocessing
df = create_target(df) # Add response (quality)
X, y = split_features(df) # split columns to x (parameters) and y (response)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE
)
# Split data for train and test

# 5. Train model
model = train_model(X_train, y_train)

# 6. Predict
y_pred_prob = predict(model, X_test) # Returns probability for each wine test
y_pred = (y_pred_prob > 0.5).astype(int) # Convert probability into binary prediction

# 7. Evaluate
evaluate(y_test, y_pred)

# 8. Save predictions
submission = pd.DataFrame({
    "Quality": y_pred_prob
})

submission.to_csv("Predictions.csv", index=False)

print("\nPredictions saved successfully!")