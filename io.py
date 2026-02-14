import os
import pandas as pd

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def append_row_csv(path: str, row: dict):
    ensure_dir(os.path.dirname(path) or ".")
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)
