import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_model(model, X_test, y_test, model_name="Model"):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results = {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
    }

    print(f"\nModel: {model_name}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return results


def evaluate_multiple_models(models, X_test, y_test):

    results = []

    for name, model in models.items():
        metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=name,
        )
        results.append(metrics)

    return pd.DataFrame(results)


def save_metrics(results_df, output_path="metrics_results.xlsx"):
   

    results_df.to_excel(output_path, index=False)
    print(f"Metrics saved to {output_path}")
