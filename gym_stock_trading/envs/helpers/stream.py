import os
import threading

import alpaca_trade_api as tradeapi

try:
    PAPER_APCA_API_KEY_ID = os.environ['PAPER_APCA_API_KEY_ID']
    PAPER_APCA_API_SECRET_KEY = os.environ['PAPER_APCA_API_SECRET_KEY']
    PAPER_APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
    LIVE_APCA_API_KEY_ID = os.environ['LIVE_APCA_API_KEY_ID']
    LIVE_APCA_API_SECRET_KEY = os.environ['LIVE_APCA_API_SECRET_KEY']
except KeyError:
    # TODO need to raise error here
    pass

# TODO need to ensure the websocket is ended when nothing is using it


class Stream():
    channels = ['trade_updates']

    def __init__(self, symbol):
        self.live_conn = tradeapi.StreamConn(
            LIVE_APCA_API_KEY_ID,
            LIVE_APCA_API_SECRET_KEY
        )
        self.start()
        self.subscribe(symbol)

    def start(self):

        try:
            tWS = threading.Thread(
                target=self.live_conn.run, args=[self.channels])
            tWS.start()
        except RuntimeError as e:
            # Already running
            print(e)

    def subscribe(self, symbol):
        channel = 'AM.' + symbol
        if channel not in self.channels:
            self.channels.append(channel)
            # TODO test this
            self.live_conn.subscribe(channel)

    def unsubscribe(self, symbol):
        channel = 'AM.' + symbol
        if channel in self.channels:
            self.channels.remove(channel)
            # TODO test this
            self.live_conn.unsubscribe(channel)

    def close(self):
        # Stop websocket
        pass
