from src.metrics import calculate_metrics


def test_metrics():

    labels = [0, 1, 2, 3]

    predictions = [0, 1, 2, 3]

    metrics = calculate_metrics(
        labels,
        predictions,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0