from alpaca_trade_api.common import URL
from alpaca_trade_api.stream import Stream
import datetime
from dotenv import load_dotenv
import os


async def trade_callback(trade):
    print(trade)
    pass


async def quote_callback(quote):
    print(quote)
    pass


def main():
    load_dotenv()
    stream = Stream(
        os.environ.get('APCA_API_KEY_ID'),
        os.environ.get('APCA_API_SECRET_KEY'),
        base_url=URL(os.environ.get('APCA_API_BASE_URL')),
        data_feed='iex')
    stream.subscribe_trades(trade_callback, 'SPY')
    stream.subscribe_quotes(quote_callback, 'SPY')

    stream.run()


if __name__ == "__main__":
    main()
