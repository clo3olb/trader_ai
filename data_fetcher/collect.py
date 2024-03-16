from dotenv import dotenv_values
from market import getMonths, createMergedCSV
import asyncio
import os

# Load the environment variables from the .env file
env_vars = dotenv_values("local.env")


async def main():
    symbol = "AAPL"
    interval = "1min"
    start_date = "2022-01"
    end_date = "2022-03"
    api_key = env_vars['ALPHA_VANTAGE_API_KEY']
    result_file = "data_fetcher/merged_data.csv"

    if os.path.exists(result_file):
        os.remove(result_file)

    months = getMonths(start_date, end_date)

    for month in months:
        await createMergedCSV(symbol, interval, month, api_key, result_file)


asyncio.run(main())
