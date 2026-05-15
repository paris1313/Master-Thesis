from sklearn.model_selection import train_test_split

from src.data_loader import load_excel_data
from src.deep_learning import build_ann
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

model = build_ann(X_train.shape[1])

model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_split=0.2,
    verbose=1
)

loss = model.evaluate(X_test, y_test)

print(loss)
