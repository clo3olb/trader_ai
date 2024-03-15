from dotenv import dotenv_values
from market import getMonths, createMergedCSV
import os

# Load the environment variables from the .env file
env_vars = dotenv_values("local.env")
api_key = env_vars['ALPHA_VANTAGE_API_KEY']

# create csv file
symbol = "AAPL"
interval = "1min"
month = "2021-02"

months = getMonths("2021-01", "2021-02")

result_file = "{}_intraday.csv".format(symbol)
os.remove(result_file) if os.path.exists(result_file) else None

for month in months:
    createMergedCSV(symbol, interval, month, api_key,
                    result_file)