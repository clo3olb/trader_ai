import pandas as pd
import os
import pandas as pd
import os
import asyncio
import aiohttp


def createUrl(api_domain: str, function: str, api_key: str, params: dict) -> str:
    url = f"https://{api_domain}/query?function={function}&apikey={api_key}"

    for key, value in params.items():
        url += f"&{key}={value}"

    return url


async def fetch_data(
    url: str,
    data_key: str,
    column_mapping: dict
) -> pd.DataFrame:
    print(url)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()

            fetched_data = data.get(data_key)

            if not fetched_data:
                raise Exception(f"Error: {data.get('Information')}")

            # create data frame
            data_frame = pd.DataFrame(
                columns=["Date"] + list(column_mapping.values()))

            for date, item in fetched_data.items():
                row = {"Date": date}
                for column, key in column_mapping.items():
                    row[column] = item[key]
                data_frame = pd.concat(
                    [data_frame, pd.DataFrame([row])], ignore_index=True)

            data_frame["Date"] = pd.to_datetime(data_frame["Date"])

            return data_frame


async def fetchIntradyPriceData(symbol: str, interval: str, month: str, api_key: str) -> pd.DataFrame:
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

    url = createUrl("www.alphavantage.co", function, api_key, params)

    print(f"{month} Intraday Price - Fetching...")
    data_frame = await fetch_data(url, data_key, column_mapping)
    print(f"{month} Intraday Price - Done")

    return data_frame


async def fetchSMAData(symbol: str, interval: str, month: str, time_period: int, api_key: str) -> pd.DataFrame:
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

    url = createUrl("www.alphavantage.co", function, api_key, params)

    print(f"{month} SMA_{time_period} - Fetching...")

    data_frame = await fetch_data(url, data_key, column_mapping)

    print(f"{month} SMA_{time_period} - Done")

    return data_frame


async def createMergedCSV(symbol: str, interval: str, month: str, api_key: str, result_file: str):
    tasks = [
        fetchIntradyPriceData(symbol, interval, month, api_key=api_key),
        fetchSMAData(symbol, interval, month, 10, api_key=api_key),
        fetchSMAData(symbol, interval, month, 50, api_key=api_key),
        fetchSMAData(symbol, interval, month, 100, api_key=api_key)
    ]

    results = await asyncio.gather(*tasks)

    price_data, sma_10_data, sma_50_data, sma_100_data = results

    # merge
    df = price_data.merge(sma_10_data, how="outer", on="Date")
    df = df.merge(sma_50_data, how="outer", on="Date")
    df = df.merge(sma_100_data, how="outer", on="Date")

    # remove rows with missing values
    df = df.dropna()

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
