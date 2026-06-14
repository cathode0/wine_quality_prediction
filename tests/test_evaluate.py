from evaluation.evaluate import evaluate


def test_evaluate():

    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]

    evaluate(
        y_true,
        y_pred
    )

    assert True