import unittest

import gym


class TestAlpacaStockTradingEnv(unittest.TestCase):
    def setUp(self):
        self.symbol = 'TSLA'
        self.previous_close = 827.60
        self.daily_avg_volume = 31140787

        self.env = gym.make(
            'gym_stock_trading:AlpacaStockTrading-v0',
            symbol=self.symbol,
            previous_close=self.previous_close,
            daily_avg_volume=self.daily_avg_volume
        )

        # self.env2 = gym.make(
        #     'gym_stock_trading:AlpacaStockTrading-v0',
        #     symbol='AAPL',
        #     previous_close=self.previous_close,
        #     daily_avg_volume=self.daily_avg_volume
        # )

    def test_inititalize_env(self):

        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.current_episode, 0)
        self.assertEqual(self.env.live, False)
        self.assertEqual(self.env.symbol, self.symbol)
        self.assertEqual(self.env.market, None)
        self.assertEqual(self.env.volume_enabled, True)
        self.assertEqual(self.env.daily_avg_volume, self.daily_avg_volume)
        self.assertEqual(self.env.asset_data, None)
        self.assertEqual(self.env.normalized_asset_data, None)
        self.assertEqual(self.env.previous_close, self.previous_close)
        self.assertEqual(self.env.observation_size, 1)
        self.assertEqual(self.env.base_value, 10000.0)
        self.assertEqual(self.env.equity, [10000.0])
        self.assertEqual(self.env.cash, [10000.0])
        self.assertEqual(self.env.profit_loss, [0.0])
        self.assertEqual(self.env.positions, [(0, 0.0)])
        self.assertEqual(self.env.alpaca_positions, [(0, 0.0)])
        self.assertEqual(self.env.current_alpaca_position, (0, 0.0))
        self.assertEqual(self.env.rewards, [0.0])
        self.assertEqual(self.env.max_qty, None)

    def test_reset_env(self):
        # If market is closed, reset will sleep
        # correct_max_qty
        # self.assertEqual(self.env.max_qty, )
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
