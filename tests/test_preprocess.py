import pandas as pd

from preprocessing.preprocess import create_target


def test_create_target():

    df = pd.DataFrame({
        "quality": [5, 6, 7]
    })

    result = create_target(df)

    assert list(
        result["response"]
    ) == [0, 1, 1]