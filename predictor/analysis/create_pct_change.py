import pandas as pd

symbols = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NFLX",
    "JPM", "BAC", "C", "WFC", "GS",
    "KO", "PG", "MCD", "DIS", "NKE",
    "JNJ", "PFE", "MRK", "ABT", "BMY",
    "XOM", "CVX", "COP", "SLB", "PSX",
]

pct_change_columns = [ 
    "Open",  "High",  "Low",  "Close",  "Volume",  
    "SMA_10",  "SMA_50",  "SMA_100",  
    "EMA_10",  "EMA_50",  "EMA_100",  
    "VWAP"
]


def create_pct_change():
    dataset_path = "predictor/dataset/"

    for symbol in symbols:
        data_path = dataset_path + symbol + ".csv"

        data = pd.read_csv(data_path)

        for column in data.columns:
            if column in pct_change_columns:
                data[column] = data[column].pct_change()

        # remove first row
        data = data.iloc[1:]

        print(data.head())

        data.to_csv(dataset_path + symbol + "_pct.csv", index=False)


create_pct_change()
