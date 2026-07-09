import pandas as pd

def load_data(path="dataset/dataset.csv"):
    data = pd.read_csv(path)
    print(data.head())
    return data
