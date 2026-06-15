import numpy as np

def create_target(df):
    """
    Convert wine quality into binary classification:
    Good (1) if quality >= 6, else Bad (0)
    """
    df = df.copy()
    df['response'] = np.where(df['quality'] >= 6, 1, 0)
    return df


def split_features(df):
    """
    Split dataset into features (X) and target (y)
    """
    X = df.drop(columns=['quality', 'response'])
    y = df['response']
    return X, y