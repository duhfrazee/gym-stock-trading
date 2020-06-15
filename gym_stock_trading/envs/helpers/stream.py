import asyncio
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
    live_conn = tradeapi.StreamConn(
        LIVE_APCA_API_KEY_ID,
        LIVE_APCA_API_SECRET_KEY
    )
    paper_conn = tradeapi.StreamConn(
        PAPER_APCA_API_KEY_ID,
        PAPER_APCA_API_SECRET_KEY,
        PAPER_APCA_API_BASE_URL
    )

    def __init__(self, symbol, live):
        self.live = live
        channel = 'AM.' + symbol
        if self.live:
            self.live_channels.append(channel)
        else:
            self.paper_channels.append(channel)
        self.start()

        # self.paper_conn.loop.run_until_complete(self.subscribe(symbol))

    def start(self):
        # TODO test that both can be running at once
        try:
            if self.live:
                tLWS = threading.Thread(
                    target=self.live_conn.run, args=[self.live_channels])
                tLWS.start()
            else:
                tPWS = threading.Thread(
                    target=self.paper_conn.run, args=[self.paper_channels])
                tPWS.start()
        except RuntimeError as e:
            # Already running
            pass

    # async def subscribe(self, symbol):
    #     channel = 'AM.' + symbol
    #     # TODO test this
    #     if self.live:
    #         if channel not in self.live_channels:
    #             self.live_channels.append(channel)
    #             await self.live_conn.subscribe([channel])
    #     else:
    #         if channel not in self.paper_channels:
    #             self.paper_channels.append(channel)
    #             await self.paper_conn.subscribe([channel])

    # async def unsubscribe(self, symbol):
    #     channel = 'AM.' + symbol
    #     if self.live:
    #         if channel in self.live_channels:
    #             self.live_channels.remove(channel)
    #             await self.live_conn.unsubscribe(channel)
    #     else:
    #         if channel in self.paper_channels:
    #             self.paper_channels.remove(channel)
    #             await self.paper_conn.unsubscribe(channel)

    def close(self):
        print('CLOSING')
        # asyncio.get_event_loop().run_until_complete(self.live_conn.close())
        # time.sleep(10)
        # self.paper_conn.loop.run_until_complete(self.paper_conn.close())
        # self.paper_conn.loop.close()
        pass
