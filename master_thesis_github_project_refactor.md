# Master Thesis Repository Refactor

## Target Repository

Repository:
https://github.com/paris1313/Master-Thesis

---

# Proposed GitHub Project Structure

```text
Master-Thesis/
│
├── README.md
├── requirements.txt
├── .gitignore
├── setup.py
│
├── data/
│   ├── raw/
│   │   └── Par1.xlsx
│   ├── processed/
│   └── outputs/
│       ├── metrics/
│       ├── figures/
│       └── models/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── visualization.py
│   ├── evaluation.py
│   ├── classical_models.py
│   ├── advanced_models.py
│   ├── deep_learning.py
│   ├── prediction.py
│   └── utils.py
│
├── scripts/
│   ├── train_classical.py
│   ├── train_advanced.py
│   ├── train_deep_learning.py
│   ├── evaluate_models.py
│   └── generate_plots.py
│
└── tests/
    ├── test_preprocessing.py
    ├── test_models.py
    └── test_evaluation.py
```

---

# README.md

```markdown
# Ship Fuel Consumption Prediction using Machine Learning and Deep Learning

## Overview

This project predicts ship fuel oil consumption using Machine Learning and Deep Learning algorithms trained on real operational and environmental vessel data.

The framework evaluates multiple regression algorithms and neural network architectures for maritime fuel consumption prediction.

---

# Features

## Data Processing

- Excel dataset loading
- Outlier removal using Z-score
- Feature engineering
- Ballast/Laden voyage separation
- Filtering invalid operational conditions

## Machine Learning Models

### Classical ML

- Linear Regression
- Random Forest
- Decision Tree
- ElasticNet
- Gradient Boosting
- Support Vector Regression

### Advanced ML

- XGBoost
- Gaussian Process Regression
- Bayesian Ridge
- KNN Regression
- SGD Regressor
- Voting Regressor

### Deep Learning

- Artificial Neural Networks (ANN)
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory Networks (LSTM)

---

# Dataset

The dataset contains ship telemetry measurements sampled every 15 minutes.

Measurements include:

- Fuel Oil Consumption
- Engine RPM
- Brake Power
- Wind Force
- Sea Currents
- Draft Measurements
- Cargo Weight
- Ship Speed

---

# Installation

```bash
git clone https://github.com/paris1313/Master-Thesis.git
cd Master-Thesis
```

## Create Virtual Environment

### Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\\Scripts\\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

## Train Classical ML Models

```bash
python scripts/train_classical.py
```

## Train Advanced ML Models

```bash
python scripts/train_advanced.py
```

## Train Deep Learning Models

```bash
python scripts/train_deep_learning.py
```

## Generate Visualizations

```bash
python scripts/generate_plots.py
```

---

# Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

---

# Technologies

- Python
- Scikit-Learn
- TensorFlow/Keras
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

# License

MIT License
```

---

# requirements.txt

```txt
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
xgboost
tensorflow
keras
bayesian-optimization
scikit-plot
openpyxl
h5py
```

---

# src/config.py

```python
RANDOM_STATE = 42

TARGET_COLUMN = "M/E FOC (tons/day)"

TEST_SIZE = 0.2

FEATURE_COLUMNS = [
    'Slip SOW (%)',
    'Wind Force (BF)',
    'M/E RPM',
    'SOW (knots)',
    'M/E Brake Power (KW)',
    'Trim (m)',
    'Depth (m)',
    'True Wind Angle (deg off the bow)',
    'Engine Speed (knots)',
    'SOG (knots)',
    'Sea Currents Speed (knots)',
    'Sea Currents Angle (deg off the bow)',
    'Cargo Weight (tons)',
    'Mean Draft (m)',
    'Course Drift (deg)'
]
```

---

# src/data_loader.py

```python
import pandas as pd


def load_excel_data(path: str):
    df = pd.read_excel(path)
    return df
```

---

# src/feature_engineering.py

```python

def add_mean_draft(df):
    df['Mean Draft (m)'] = (
        df['DRAFT MP'] +
        df['DRAFT MS'] +
        df['DRAFT FORWARD'] +
        df['DRAFT AFT']
    ) / 4

    return df


def add_course_drift(df):
    df['Course Drift (deg)'] = (
        df['Gyro Heading (deg)'] -
        df['GPS Course (deg)']
    )

    return df
```

---

# src/preprocessing.py

```python
import numpy as np
from scipy import stats


def remove_outliers(df, columns, threshold=2.7):

    for col in columns:
        z_scores = np.abs(stats.zscore(df[col]))
        df = df[z_scores < threshold]

    return df


def filter_operational_conditions(df):

    df = df.loc[
        df['Voyage Status'].str.contains('Sailing') &
        (df['M/E FLOW METER GROUP ALARM'] != 1) &
        (df['SOW (knots)'] > 5) &
        (df['M/E RPM'] > 0) &
        (df['Depth (m)'] > 70)
    ]

    return df


def split_loading_states(df):

    ballast = df[df['Loading Status'].str.contains('BALLAST')]
    laden = df[df['Loading Status'].str.contains('LADEN')]

    return ballast, laden
```

---

# src/classical_models.py

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor


def get_models():

    return {
        'LR': LinearRegression(),
        'RF': RandomForestRegressor(n_estimators=100),
        'DT': DecisionTreeRegressor(max_depth=5),
        'GBR': HistGradientBoostingRegressor(),
        'SVR': SVR(C=1.0, epsilon=0.2),
        'EN': ElasticNet()
    }
```

---

# src/advanced_models.py

```python
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

kernel = DotProduct() + WhiteKernel()


def get_advanced_models():

    return {
        'Lasso': Lasso(alpha=0.1),
        'Bayesian': BayesianRidge(),
        'SGD': SGDRegressor(),
        'KNN': KNeighborsRegressor(n_neighbors=7),
        'DT': DecisionTreeRegressor(max_depth=5),
        'SVR': SVR(C=1.0, epsilon=0.2),
        'GPR': GaussianProcessRegressor(kernel=kernel)
    }


def build_voting_regressor(models):

    estimators = [(name, model) for name, model in models.items()]

    return VotingRegressor(estimators=estimators)
```

---

# src/deep_learning.py

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM


def build_ann(input_dim):

    model = Sequential()

    model.add(Dense(12, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_absolute_error'
    )

    return model



def build_rnn(input_shape):

    model = Sequential()

    model.add(SimpleRNN(100, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(SimpleRNN(100, activation='relu'))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_absolute_error'
    )

    return model



def build_lstm(input_shape):

    model = Sequential()

    model.add(LSTM(150, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_absolute_error'
    )

    return model
```

---

# src/evaluation.py

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



def evaluate_regression(y_true, y_pred):

    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
```

---

# src/visualization.py

```python
import matplotlib.pyplot as plt
import seaborn as sns



def plot_predictions(y_true, y_pred):

    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.show()



def plot_metrics(labels, values, title, xlabel):

    fig, ax = plt.subplots()

    ax.barh(labels, values)

    ax.set_xlabel(xlabel)
    ax.set_title(title)

    plt.show()
```

---

# scripts/train_classical.py

```python
from sklearn.model_selection import train_test_split

from src.data_loader import load_excel_data
from src.feature_engineering import add_mean_draft
from src.feature_engineering import add_course_drift
from src.preprocessing import remove_outliers
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
```

---

# scripts/train_advanced.py

```python
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
```

---

# scripts/train_deep_learning.py

```python
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
    epochs=100,
    validation_split=0.2,
    verbose=1
)

loss = model.evaluate(X_test, y_test)

print('Test Loss:', loss)
```

---

# tests/test_preprocessing.py

```python
from src.preprocessing import remove_outliers


def test_remove_outliers():
    assert callable(remove_outliers)
```

---

# tests/test_models.py

```python
from src.classical_models import get_models


def test_get_models():

    models = get_models()

    assert len(models) > 0
```

---

# tests/test_evaluation.py

```python
from src.evaluation import evaluate_regression


def test_evaluate_regression():

    result = evaluate_regression([1, 2], [1, 2])

    assert result['MAE'] == 0
```

---

# setup.py

```python
from setuptools import setup, find_packages

setup(
    name='ship-fuel-consumption-ml',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'scikit-learn',
        'xgboost',
        'tensorflow'
    ]
)
```

---

# .gitignore

```gitignore
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
*.h5
*.xlsx
*.csv
.DS_Store
```

---

# Recommended Next Steps

1. Move the original monolithic notebook code into the modular files above.
2. Remove duplicated code blocks.
3. Add logging support.
4. Save trained models using joblib.
5. Add CLI arguments using argparse.
6. Add Docker support.
7. Add GitHub Actions CI/CD.
8. Add Streamlit dashboard.

