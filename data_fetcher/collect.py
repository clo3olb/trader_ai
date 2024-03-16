from dotenv import dotenv_values
from market import getMonths, createMergedCSV
import os

# Load the environment variables from the .env file
env_vars = dotenv_values("local.env")


def main():
    symbol = "AAPL"
    interval = "1min"
    start_date = "2022-01"
    end_date = "2022-01"
    api_key = env_vars['ALPHA_VANTAGE_API_KEY']
    result_file = "data_fetcher/merged_data.csv"

    if os.path.exists(result_file):
        os.remove(result_file)

    months = getMonths(start_date, end_date)

    for month in months:
        createMergedCSV(symbol, interval, month, api_key, result_file)


main()
