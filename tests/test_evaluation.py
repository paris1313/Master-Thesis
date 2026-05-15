from src.evaluation import evaluate_regression

def test_evaluation():
    result = evaluate_regression([1], [1])
    assert result['MAE'] == 0
