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
    paper_channels = ['trade_updates']
    live_channels = ['trade_updates']
    self.live_conn = tradeapi.StreamConn(
        LIVE_APCA_API_KEY_ID,
        LIVE_APCA_API_SECRET_KEY
    )
    self.paper_conn = tradeapi.StreamConn(
        PAPER_APCA_API_KEY_ID,
        PAPER_APCA_API_SECRET_KEY,
        PAPER_APCA_API_BASE_URL
    )

    def __init__(self, symbol, live):
        self.live = live
        self.start()
        self.subscribe(symbol)

    def start(self):
        # TODO test that both can be running at once
        try:
            if self.live:
                tWS = threading.Thread(
                    target=self.live_conn.run, args=[self.live_channels])
                tWS.start()
            else:
                tWS = threading.Thread(
                    target=self.paper_conn.run, args=[self.paper_channels])
                tWS.start()
        except RuntimeError as e:
            # Already running
            print(e)

    def subscribe(self, symbol):
        channel = 'AM.' + symbol
        # TODO test this
        if self.live:
            if channel not in self.live_channels:
                self.live_channels.append(channel)
                self.live_conn.subscribe(channel)
        else:
            if channel not in self.paper_channels:
                self.paper_channels.append(channel)
                self.paper_conn.subscribe(channel)

    def unsubscribe(self, symbol):
        channel = 'AM.' + symbol
        if channel in self.channels:
            self.channels.remove(channel)
            # TODO test this
            if self.live:
                self.live_conn.unsubscribe(channel)
            else:
                self.paper_conn.unsubscribe(channel)

    def close(self):
        # Stop websocket
        pass
