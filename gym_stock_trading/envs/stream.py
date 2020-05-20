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


class Stream():
    def __init__(self):
        self.live_conn = tradeapi.StreamConn(
            LIVE_APCA_API_KEY_ID,
            LIVE_APCA_API_SECRET_KEY
        )
        self.channels = ['trade_updates']
        self.running = False

    def start(self):

        if not self.running:
            tWS = threading.Thread(
                target=self.live_conn.run, args=[self.channels])
            tWS.start()
            self.running = True

    def subscribe(self, channel):
        # check if stream already there, if not append
        pass

    def unsubscribe(self, channel):
        # check if stream already there, if not remove
        pass

    def close(self):
        # Stop websocket
        self.running = False
