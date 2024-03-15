import asyncio
import aiohttp
import asyncio
import aiohttp


async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            asyncio.sleep(1)
            data = await response.json()
            return data

async def main():
    url = "https://api.example.com/data"
    data = await fetch_data(url)
    print(data)

main()