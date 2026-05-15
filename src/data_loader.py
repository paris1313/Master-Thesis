import pandas as pd

def load_excel_data(path: str):
    return pd.read_excel(path)
