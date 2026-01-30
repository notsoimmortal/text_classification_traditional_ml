import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='latin1')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df