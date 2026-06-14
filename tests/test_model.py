import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from models.logistic_model import train_model, predict


def test_model_training():
    X = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [4, 3, 2, 1]
    })
    y = [0, 0, 1, 1]
    
    model = train_model(X, y)
    y_pred_prob = predict(model, X)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    assert len(y_pred) == 4
    print("✓ test_model_training passed")


if __name__ == "__main__":
    test_model_training()
    print("\n✅ Model tests passed!")
