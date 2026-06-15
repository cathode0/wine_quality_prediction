import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluate import evaluate


def test_evaluate():
    """Test that evaluate function runs without errors"""
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 0, 1]  # Perfect predictions
    
    try:
        evaluate(y_true, y_pred)
        print("✓ test_evaluate passed (perfect predictions)")
    except Exception as e:
        print(f"✗ test_evaluate failed: {e}")
        raise
    
    # Test with imperfect predictions
    y_pred_imperfect = [0, 1, 1, 0, 0, 1]
    
    try:
        evaluate(y_true, y_pred_imperfect)
        print("✓ test_evaluate passed (imperfect predictions)")
    except Exception as e:
        print(f"✗ test_evaluate failed: {e}")
        raise


def test_evaluate_with_different_lengths():
    """Test that evaluate handles different length arrays"""
    y_true = [0, 1, 0]
    y_pred = [0, 1, 0, 1]  # Different length
    
    try:
        evaluate(y_true, y_pred)
        print("⚠️ evaluate didn't complain about different lengths")
    except Exception as e:
        print(f"✓ evaluate correctly raised error: {e}")


if __name__ == "__main__":
    test_evaluate()
    test_evaluate_with_different_lengths()
    print("\n✅ All evaluation tests passed!")