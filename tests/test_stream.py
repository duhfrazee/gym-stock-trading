import threading
import unittest

from gym_stock_trading.envs.helpers.stream import Stream


class TestStream(unittest.TestCase):
    def setUp(self):
        live = False
        symbol = 'TSLA'
        self.paper_stream = Stream(symbol, live)

        # live = True
        # symbol = 'GPRO'
        # self.live_stream = Stream(symbol, live)

    async def _on_minute_bars(self, conn, channel, bar):
        print(bar)

    def test_inititalize_paper_stream(self):
        correct_paper_channels = ['trade_updates', 'AM.TSLA']
        self.assertEqual(
            self.paper_stream.paper_channels,
            correct_paper_channels
        )
        # self._on_minute_bars =\
        #     self.paper_stream.paper_conn.on(r'AM$')(self._on_minute_bars)

    # def test_inititalize_live_stream(self):
    #     correct_live_channels = ['trade_updates', 'AM.GPRO']
    #     self.assertEqual(self.live_stream.live_channels, correct_live_channels)

    def tearDown(self):
        self.paper_stream.close()
        # self.live_stream.close()


if __name__ == '__main__':
    unittest.main()
