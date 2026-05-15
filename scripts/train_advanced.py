from sklearn.model_selection import train_test_split

from src.data_loader import load_excel_data
from src.advanced_models import get_advanced_models
from src.evaluation import evaluate_regression
from src.config import FEATURE_COLUMNS
from src.config import TARGET_COLUMN


ship_data = load_excel_data('data/raw/Par1.xlsx')

X = ship_data[FEATURE_COLUMNS]
y = ship_data[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = get_advanced_models()

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = evaluate_regression(y_test, predictions)

    print(name)
    print(metrics)
