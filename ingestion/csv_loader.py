import pandas as pd


def extract_csv_text(file):
    df = pd.read_csv(file)
    return df.to_string(index=False)