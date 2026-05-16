from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def evaluate(y_test, y_pred):
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))