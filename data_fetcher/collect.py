from dotenv import dotenv_values
from market import getMonths, createMergedCSV
import os
import asyncio
import aiohttp

# Load the environment variables from the .env file
env_vars = dotenv_values("local.env")
api_key = env_vars['ALPHA_VANTAGE_API_KEY']

# create csv file
symbol = "AAPL"
interval = "1min"
month = "2021-02"


months = getMonths("2021-01", "2021-01")

result_file = "{}_intraday.csv".format(symbol)
os.remove(result_file) if os.path.exists(result_file) else None

async def fetch_data(symbol, interval, month, api_key, result_file):
    await createMergedCSV(symbol, interval, month, api_key, result_file)

async def main():
    tasks = []
    for month in months:
        task = asyncio.create_task(fetch_data(symbol, interval, month, api_key, result_file))
        tasks.append(task)
    await asyncio.gather(*tasks)

asyncio.run(main())