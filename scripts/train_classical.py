from sklearn.model_selection import train_test_split

from src.data_loader import load_excel_data
from src.feature_engineering import add_mean_draft
from src.feature_engineering import add_course_drift
from src.preprocessing import filter_operational_conditions
from src.classical_models import get_models
from src.evaluation import evaluate_regression
from src.config import FEATURE_COLUMNS
from src.config import TARGET_COLUMN


ship_data = load_excel_data('data/raw/Par1.xlsx')

ship_data = add_mean_draft(ship_data)
ship_data = add_course_drift(ship_data)

ship_data = filter_operational_conditions(ship_data)

X = ship_data[FEATURE_COLUMNS]
y = ship_data[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = get_models()

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = evaluate_regression(y_test, predictions)

    print(name)
    print(metrics)
