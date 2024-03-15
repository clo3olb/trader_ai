import pandas as pd
import os
import requests


def fetchIntradyData(symbol: str, interval: str, month: str, api_key: str) -> pd.DataFrame:
    api_domain = "www.alphavantage.co"
    function = "TIME_SERIES_INTRADAY"
    output_size = "full"

    url = "https://{}/query?function={}&symbol={}&interval={}&month={}&outputsize={}&apikey={}".format(
        api_domain, function, symbol, interval, month, output_size, api_key)

    r = requests.get(url)
    data = r.json()

    price_data = data.get("Time Series (1min)")

    if not price_data:
        print(f"Error: {data.get('Error Message')}")
        return

    # create data frame
    data_frame = pd.DataFrame(
        columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    for date, price in price_data.items():
        data_frame = pd.concat([data_frame, pd.DataFrame([{
            "Date": date,
            "Open": price['1. open'],
            "High": price['2. high'],
            "Low": price['3. low'],
            "Close": price['4. close'],
            "Volume": price['5. volume']
        }])], ignore_index=True)

    data_frame["Date"] = pd.to_datetime(data_frame["Date"])

    return data_frame


def fetchSMAData(symbol: str, interval: str, month: str, time_period: int, api_key: str) -> pd.DataFrame:
    api_domain = "www.alphavantage.co"
    function = "SMA"
    series_type = "close"

    url = "https://{}/query?function={}&month={}&symbol={}&interval={}&time_period={}&series_type={}&apikey={}".format(
        api_domain, function, month, symbol, interval, time_period, series_type, api_key)

    r = requests.get(url)
    data = r.json()

    sma_data = data.get("Technical Analysis: SMA")

    if not sma_data:
        print(f"Error: {data.get('Error Message')}")
        return

    # create data frame
    data_frame = pd.DataFrame(columns=["Date", "SMA_{}".format(time_period)])

    for date, sma in sma_data.items():
        data_frame = pd.concat([data_frame, pd.DataFrame([{
            "Date": date,
            "SMA_{}".format(time_period): sma['SMA']
        }])], ignore_index=True)

    data_frame["Date"] = pd.to_datetime(data_frame["Date"])

    print(data_frame.head())

    return data_frame


def createMergedCSV(symbol: str, interval: str, month: str, api_key: str, result_file: str):
    price_data = fetchIntradyData(symbol, interval, month, api_key)
    sma_10_data = fetchSMAData(symbol, interval, month, 10, api_key)
    sma_50_data = fetchSMAData(symbol, interval, month, 50, api_key)
    sma_100_data = fetchSMAData(symbol, interval, month, 100, api_key)

    # merge
    df = price_data.merge(sma_10_data, how="outer", on="Date")
    df = df.merge(sma_50_data, how="outer", on="Date")
    df = df.merge(sma_100_data, how="outer", on="Date")

    print(df.head())

    if not os.path.exists(result_file):
        df.to_csv(result_file, mode='w', header=True, index=False)
        return

    df.to_csv(result_file, mode='a', header=False, index=False)


def getMonths(start: str, end: str):
    start_year, start_month = map(int, start.split('-'))
    end_year, end_month = map(int, end.split('-'))

    months = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == start_year and month < start_month:
                continue
            if year == end_year and month > end_month:
                break
            months.append(f"{year}-{month:02d}")

    return months