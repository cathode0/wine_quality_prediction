from sklearn.linear_model import LogisticRegression
from config import MAX_ITER

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=MAX_ITER)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    return model.predict_proba(X_test)[:, 1]