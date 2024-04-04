from dotenv import dotenv_values
from market import getMonths, createMergedCSV, fetchDailyPriceData
import os
import pandas as pd

# Load the environment variables from the .env file
env_vars = dotenv_values("local.env")


def fetchMonthlyIntraday():
    symbol = "AAPL"
    interval = "1min"
    start_date = "2022-01"
    end_date = "2022-12"
    api_key = env_vars['ALPHA_VANTAGE_API_KEY']
    result_file = "data_fetcher/merged_data.csv"

    if os.path.exists(result_file):
        os.remove(result_file)

    months = getMonths(start_date, end_date)

    for month in months:
        createMergedCSV(symbol, interval, month, api_key, result_file)


def fetchHistoricalData():
    api_key = env_vars['ALPHA_VANTAGE_API_KEY']

    symbols = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "NFLX",
        "JPM", "BAC", "C", "WFC", "GS",
        "KO", "PG", "MCD", "DIS", "NKE",
        "JNJ", "PFE", "MRK", "ABT", "BMY",
        "XOM", "CVX", "COP", "SLB", "PSX",
    ]
    for symbol in symbols:
        result_file = f"predictor/dataset/{symbol}.csv"
        if os.path.exists(result_file):
            os.remove(result_file)

        data = fetchDailyPriceData(symbol, api_key)

        # only use 2022 data
        data["Date"] = pd.to_datetime(data["Date"])
        data = data[data["Date"].dt.year == 2022]

        data.rename(columns={"Date": "Timestamp"}, inplace=True)
        data.set_index("Timestamp", inplace=True)
        data.sort_index(ascending=True, inplace=True)
        data.to_csv(result_file, mode='w', header=True, index=True)
        print(f"{symbol} - Done")


fetchHistoricalData()

"""
기술 기업:
(AAPL) Apple Inc.
(MSFT) Microsoft Corporation
(AMZN) Amazon.com, Inc.
(GOOGL) Alphabet Inc.
(NFLX) Netflix, Inc.

금융 기업:
(JPM) JPMorgan Chase & Co.
(BAC) Bank of America Corporation
(C) Citigroup Inc.
(WFC) Wells Fargo & Company
(GS) Goldman Sachs Group, Inc.

소비재 기업:
(KO) The Coca-Cola Company
(PG) Procter & Gamble Company
(MCD) McDonald's Corporation
(DIS) The Walt Disney Company
(NKE) Nike, Inc.

의료 기업:
(JNJ) Johnson & Johnson
(PFE) Pfizer Inc.
(MRK) Merck & Co., Inc.
(ABT) Abbott Laboratories
(BMY) Bristol-Myers Squibb Company

에너지 기업:
(XOM) Exxon Mobil Corporation
(CVX) Chevron Corporation
(COP) ConocoPhillips
(SLB) Schlumberger Limited
(PSX) Phillips 66
"""
