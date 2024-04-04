import pandas as pd
import os
import requests


def createUrl(api_domain: str, function: str, api_key: str, params: dict) -> str:
    url = f"https://{api_domain}/query?function={function}&apikey={api_key}"

    for key, value in params.items():
        url += f"&{key}={value}"

    return url


def fetch_data(
    url: str,
    data_key: str,
    column_mapping: dict
) -> pd.DataFrame:
    with requests.get(url, timeout=10) as response:
        data = response.json()
        fetched_data = data.get(data_key)

        if not fetched_data:
            raise Exception(f"Error: {data.get('Information')}")

        # create data frame
        data_frame = pd.DataFrame(
            columns=["Date"] + list(column_mapping.keys()))

        rows = []
        for date, item in fetched_data.items():
            row = {"Date": date}
            for column, key in column_mapping.items():
                row[column] = item[key]
            rows.append(row)
        data_frame = pd.concat([data_frame, pd.DataFrame(rows)])
        data_frame["Date"] = pd.to_datetime(data_frame["Date"])
        return data_frame


def fetchDailyPriceData(symbol: str, api_key: str) -> pd.DataFrame:
    domain = "www.alphavantage.co"
    function = "TIME_SERIES_DAILY"
    data_key = "Time Series (Daily)"
    column_mapping = {
        "Open": "1. open",
        "High": "2. high",
        "Low": "3. low",
        "Close": "4. close",
        "Volume": "5. volume"
    }

    params = {
        "apikey": api_key,
        "function": function,
        "symbol": symbol,
        "outputsize": "full",

    }

    url = createUrl(domain, function, api_key, params)
    print(f"{symbol} Daily Price - Fetching...")
    data_frame = fetch_data(url, data_key, column_mapping)
    print(f"{symbol} Daily Price - Done")
    return data_frame


def fetchIntradyPriceData(symbol: str, interval: str, month: str, api_key: str) -> pd.DataFrame:
    domain = "www.alphavantage.co"
    function = "TIME_SERIES_INTRADAY"
    data_key = "Time Series ({})".format(interval)
    column_mapping = {
        "Open": "1. open",
        "High": "2. high",
        "Low": "3. low",
        "Close": "4. close",
        "Volume": "5. volume"
    }

    params = {
        "apikey": api_key,
        "function": function,
        "month": month,
        "interval": interval,
        "symbol": symbol,
        "outputsize": "full",

    }

    url = createUrl(domain, function, api_key, params)
    print(f"{month} Intraday Price - Fetching...")
    data_frame = fetch_data(url, data_key, column_mapping)
    print(f"{month} Intraday Price - Done")
    return data_frame


def fetchSMAData(symbol: str, interval: str, month: str, time_period: int, api_key: str) -> pd.DataFrame:
    domain = "www.alphavantage.co"
    function = "SMA"
    data_key = "Technical Analysis: SMA"
    column_mapping = {
        "SMA_{}".format(time_period): "SMA"
    }

    params = {
        "apikey": api_key,
        "function": function,
        "symbol": symbol,
        "interval": interval,
        "time_period": time_period,
        "series_type": "close",
        "month": month,
    }

    url = createUrl(domain, function, api_key, params)
    print(f"{month} SMA_{time_period} - Fetching...")
    data_frame = fetch_data(url, data_key, column_mapping)
    print(f"{month} SMA_{time_period} - Done")
    return data_frame


def fetchEMAData(symbol: str, interval: str, month: str, time_period: int, api_key: str) -> pd.DataFrame:
    domain = "www.alphavantage.co"
    function = "EMA"
    data_key = "Technical Analysis: EMA"
    column_mapping = {
        "EMA_{}".format(time_period): "EMA"
    }

    params = {
        "apikey": api_key,
        "function": function,
        "symbol": symbol,
        "interval": interval,
        "time_period": time_period,
        "series_type": "close",
        "month": month,
    }

    url = createUrl(domain, function, api_key, params)
    print(f"{month} EMA_{time_period} - Fetching...")
    data_frame = fetch_data(url, data_key, column_mapping)
    print(f"{month} EMA_{time_period} - Done")
    return data_frame


def fetchRSIData(symbol: str, interval: str, month: str, time_period: int, api_key: str) -> pd.DataFrame:
    domain = "www.alphavantage.co"
    function = "RSI"
    data_key = "Technical Analysis: RSI"
    column_mapping = {
        "RSI_{}".format(time_period): "RSI"
    }

    params = {
        "apikey": api_key,
        "function": function,
        "symbol": symbol,
        "interval": interval,
        "time_period": time_period,
        "series_type": "close",
        "month": month,
    }

    url = createUrl(domain, function, api_key, params)
    print(f"{month} RSI_{time_period} - Fetching...")
    data_frame = fetch_data(url, data_key, column_mapping)
    print(f"{month} RSI_{time_period} - Done")
    return data_frame


def fetchMACDData(symbol: str, interval: str, month: str, api_key: str) -> pd.DataFrame:
    domain = "www.alphavantage.co"
    function = "MACD"
    data_key = "Technical Analysis: MACD"
    column_mapping = {
        "MACD": "MACD",
        "MACD_Hist": "MACD_Hist",
        "MACD_Signal": "MACD_Signal"
    }

    params = {
        "apikey": api_key,
        "function": function,
        "symbol": symbol,
        "interval": interval,
        "series_type": "close",
        "month": month,
    }

    url = createUrl(domain, function, api_key, params)
    print(f"{month} MACD - Fetching...")
    data_frame = fetch_data(url, data_key, column_mapping)
    print(f"{month} MACD - Done")
    return data_frame


def fetchVWAPData(symbol: str, interval: str, month: str, api_key: str) -> pd.DataFrame:
    domain = "www.alphavantage.co"
    function = "VWAP"
    data_key = "Technical Analysis: VWAP"
    column_mapping = {
        "VWAP": "VWAP"
    }

    params = {
        "apikey": api_key,
        "function": function,
        "symbol": symbol,
        "interval": interval,
        "month": month,
    }

    url = createUrl(domain, function, api_key, params)
    print(f"{month} VWAP - Fetching...")
    data_frame = fetch_data(url, data_key, column_mapping)
    print(f"{month} VWAP - Done")
    return data_frame


def fetchCryptoData(symbol: str, interval: str, month: str, api_key: str) -> pd.DataFrame:
    domain = "www.alphavantage.co"
    function = "CRYPTO_INTRADAY"
    data_key = "Time Series (Digital Currency Intraday)"
    column_mapping = {
        symbol: "4a. close (USD)",
    }

    params = {
        "apikey": api_key,
        "function": function,
        "symbol": symbol,
        "market": "USD",
        "interval": interval,
        "month": month,
    }

    url = createUrl(domain, function, api_key, params)
    print(f"{month} Crypto({symbol}) - Fetching...")
    data_frame = fetch_data(url, data_key, column_mapping)
    print(f"{month} Crypto({symbol}) - Done")
    return data_frame


def fetchCrudeOildata(interval: str, oil_type: str, api_key: str) -> pd.DataFrame:
    domain = "www.alphavantage.co"
    function = oil_type
    data_key = "data"
    column_mapping = {
        "WTI": "value",
    }

    params = {
        "apikey": api_key,
        "function": function,
        "symbol": "CL",
        "market": "USD",
        "interval": interval,
        "month": month,
    }

    url = createUrl(domain, function, api_key, params)
    print(f"{month} WTI - Fetching...")
    data_frame = fetch_data(url, data_key, column_mapping)
    print(f"{month} WTI - Done")
    return data_frame


def createMergedCSV(symbol: str, interval: str, month: str, api_key: str, result_file: str):
    price_data = fetchIntradyPriceData(
        symbol, interval, month, api_key=api_key)
    sma_10_data = fetchSMAData(symbol, interval, month, 10, api_key=api_key)
    sma_50_data = fetchSMAData(symbol, interval, month, 50, api_key=api_key)
    sma_100_data = fetchSMAData(symbol, interval, month, 100, api_key=api_key)
    ema_10_data = fetchEMAData(symbol, interval, month, 10, api_key=api_key)
    ema_50_data = fetchEMAData(symbol, interval, month, 50, api_key=api_key)
    ema_100_data = fetchEMAData(symbol, interval, month, 100, api_key=api_key)
    rsi_10_data = fetchRSIData(symbol, interval, month, 10, api_key=api_key)
    rsi_50_data = fetchRSIData(symbol, interval, month, 50, api_key=api_key)
    rsi_100_data = fetchRSIData(symbol, interval, month, 100, api_key=api_key)
    vwap_data = fetchVWAPData(symbol, interval, month, api_key=api_key)
    macd_data = fetchMACDData(symbol, interval, month, api_key=api_key)

    # merge
    df = price_data.merge(sma_10_data, how="outer", on="Date")
    df = df.merge(sma_50_data, how="outer", on="Date")
    df = df.merge(sma_100_data, how="outer", on="Date")
    df = df.merge(ema_10_data, how="outer", on="Date")
    df = df.merge(ema_50_data, how="outer", on="Date")
    df = df.merge(ema_100_data, how="outer", on="Date")
    df = df.merge(rsi_10_data, how="outer", on="Date")
    df = df.merge(rsi_50_data, how="outer", on="Date")
    df = df.merge(rsi_100_data, how="outer", on="Date")
    df = df.merge(vwap_data, how="outer", on="Date")
    df = df.merge(macd_data, how="outer", on="Date")

    df.bfill(inplace=True)
    df.sort_values("Date", inplace=True)
    df.rename(columns={"Date": "Timestamp"}, inplace=True)

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
