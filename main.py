from sklearn.model_selection import train_test_split
import pandas as pd

import config
from data.data_loader import load_data
from preprocessing.preprocess import create_target, split_features
from models.logistic_model import train_model, predict
from evaluation.evaluate import evaluate
from visualization.plots import correlation_plot, histograms, boxplots

# 1. Load data
df = load_data()

print(df.head())
print(df.info())
print(df.describe())

print(df['quality'].value_counts())
print(df.isnull().sum())

# 2. Visualization
correlation_plot(df)
histograms(df)

boxplots(df, ['sulphates', 'alcohol', 'citric acid', 'fixed acidity'])

# 3. Preprocessing
df = create_target(df)
X, y = split_features(df)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE
)

# 5. Train model
model = train_model(X_train, y_train)

# 6. Predict
y_pred_prob = predict(model, X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# 7. Evaluate
evaluate(y_test, y_pred)

# 8. Save predictions
submission = pd.DataFrame({
    "Quality": y_pred_prob
})

submission.to_csv("Predictions.csv", index=False)

print("\nPredictions saved successfully!")