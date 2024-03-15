from dotenv import dotenv_values
from market import getMonths, createMergedCSV
import asyncio

# Load the environment variables from the .env file
env_vars = dotenv_values("local.env")
api_key = env_vars['ALPHA_VANTAGE_API_KEY']


async def main():
    symbol = "AAPL"
    interval = "1min"
    start_date = "2022-01"
    end_date = "2022-03"
    api_key = "YOUR_API_KEY"
    result_file = "merged_data.csv"

    months = getMonths(start_date, end_date)

    tasks = [createMergedCSV(symbol, interval, month, api_key, result_file) for month in months]

    await asyncio.gather(*tasks)


asyncio.run(main())