import pandas as pd


dataset_path = "predictor/dataset/"
data_path = "predictor/dataset/" + "MSFT.csv"

data = pd.read_csv(data_path)
data = data.pct_change()

# remove first row
data = data[1:]

print(data.head(10))
