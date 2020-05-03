import unittest

import numpy as np
import pandas as pd

from gym_stock_trading.envs import StockTradingEnv

TSLA_AVG_DAILY_VOLUME = 18840000    # avg volume over last 30 days

class TestStockTradingEnv(unittest.TestCase):
    def setUp(self):
        self.env = StockTradingEnv()

    def test_inititalize_env(self):
        #self.env = StockTradingEnv()
        base_value = self.env.base_value
        initial_equity = self.env.equity[-1]
        profit_loss = self.env.profit_loss[-1]
        initial_cash = self.env.cash[-1]
        intitial_position = self.env.positions[-1]
        reward = self.env.rewards[-1]
        max_qty = self.env.max_qty

        self.assertEqual(base_value, 10000)
        self.assertEqual(base_value, initial_equity)
        self.assertEqual(base_value, initial_cash)
        self.assertEqual(profit_loss, 0.0)
        self.assertEqual(intitial_position, (0, 0.0))
        self.assertEqual(reward, 0.0)

        # int((allotted_amount / self.asset_data.iloc[self.current_step]['close']) * 0.95)  # Leave 5% room for price fluctuations
        self.assertEqual(max_qty, 0)
        
    def test_reset(self):
        obs = self.env.reset(self._initialize_data())

        base_value = self.env.base_value
        initial_equity = self.env.equity[-1]
        profit_loss = self.env.profit_loss[-1]
        initial_cash = self.env.cash[-1]
        intitial_position = self.env.positions[-1]
        reward = self.env.rewards[-1]
        max_qty = self.env.max_qty

        self.assertEqual(base_value, 10000)
        self.assertEqual(base_value, initial_equity)
        self.assertEqual(base_value, initial_cash)
        self.assertEqual(profit_loss, 0.0)
        self.assertEqual(intitial_position, (0, 0.0))
        self.assertEqual(reward, 0.0)

        correct_obs = np.array([261.89/(2*291.81), 262.77/(2*291.81), 260.59/(2*291.81), 261.625/(2*291.81), 706650*10/TSLA_AVG_DAILY_VOLUME])
        np.testing.assert_array_equal(obs, correct_obs)

        # int((allotted_amount / self.asset_data.iloc[self.current_step]['close']) * 0.95)  # Leave 5% room for price fluctuations
        self.assertEqual(max_qty, 38)
        
    def test_simple_long_trade_of_50_percent(self):
        _ = self.env.reset(self._initialize_data())

        # go long 50%
        action = 0.5
        observation_, reward, done, info = self.env.step(action)

        correct_observation = np.array([261.68/(2*291.81), 263.5/(2*291.81), 261.59/(2*291.81), 263.3605/(2*291.81), 378093*10/TSLA_AVG_DAILY_VOLUME])

        np.testing.assert_array_equal(observation_, correct_observation)
        self.assertAlmostEqual(reward, 32.9745)
        self.assertEqual(done, False)
        self.assertEqual(info, {})
        self.assertEqual(self.env.profit_loss[-1], 0.0)
        self.assertEqual(self.env.cash[-1], 10000.0-4970.875)
        self.assertAlmostEqual(self.env.equity[-1], 10000.0+32.9745)

        correct_position = (19, 261.625)
        self.assertEqual(self.env.positions[-1], correct_position)

    def test_simple_short_trade_of_50_percent(self):
        _ = self.env.reset(self._initialize_data())

        # go short 50%
        action = -0.5
        observation_, reward, done, info = self.env.step(action)

        correct_observation = np.array([261.68/(2*291.81), 263.5/(2*291.81), 261.59/(2*291.81), 263.3605/(2*291.81), 378093*10/TSLA_AVG_DAILY_VOLUME])

        np.testing.assert_array_equal(observation_, correct_observation)
        self.assertAlmostEqual(reward, -32.9745)
        self.assertEqual(done, False)
        self.assertEqual(info, {})
        self.assertEqual(self.env.profit_loss[-1], 0.0)
        self.assertEqual(self.env.cash[-1], 10000.0-4970.875)
        self.assertAlmostEqual(self.env.equity[-1], 10000.0-32.9745)

        correct_position = (-19, 261.625)
        self.assertEqual(self.env.positions[-1], correct_position)

    def test_simple_long_trade_of_65_percent(self):
        _ = self.env.reset(self._initialize_data())

        # go short 65%
        action = 0.65
        observation_, reward, done, info = self.env.step(action)

        correct_observation = np.array([261.68/(2*291.81), 263.5/(2*291.81), 261.59/(2*291.81), 263.3605/(2*291.81), 378093*10/TSLA_AVG_DAILY_VOLUME])

        np.testing.assert_array_equal(observation_, correct_observation)
        self.assertAlmostEqual(reward, 41.652)
        self.assertEqual(done, False)
        self.assertEqual(info, {})
        self.assertEqual(self.env.profit_loss[-1], 0.0)
        self.assertEqual(self.env.cash[-1], 10000.0-6279.0)
        self.assertAlmostEqual(self.env.equity[-1], 10000.0+41.652)

        correct_position = (24, 261.625)
        self.assertEqual(self.env.positions[-1], correct_position)

    def test_simple_short_trade_of_65_percent(self):
        _ = self.env.reset(self._initialize_data())

        # go short 65%
        action = -0.65
        observation_, reward, done, info = self.env.step(action)

        correct_observation = np.array([261.68/(2*291.81), 263.5/(2*291.81), 261.59/(2*291.81), 263.3605/(2*291.81), 378093*10/TSLA_AVG_DAILY_VOLUME])

        np.testing.assert_array_equal(observation_, correct_observation)
        self.assertAlmostEqual(reward, -41.652)
        self.assertEqual(done, False)
        self.assertEqual(info, {})
        self.assertEqual(self.env.profit_loss[-1], 0.0)
        self.assertEqual(self.env.cash[-1], 10000.0-6279.0)
        self.assertAlmostEqual(self.env.equity[-1], 10000.0-41.652)

        correct_position = (-24, 261.625)
        self.assertEqual(self.env.positions[-1], correct_position)

    def test_simple_long_trade_of_100_percent(self):
        _ = self.env.reset(self._initialize_data())

        # go long 100%
        action = 1.0
        observation_, reward, done, info = self.env.step(action)

        correct_observation = np.array([261.68/(2*291.81), 263.5/(2*291.81), 261.59/(2*291.81), 263.3605/(2*291.81), 378093*10/TSLA_AVG_DAILY_VOLUME])

        np.testing.assert_array_equal(observation_, correct_observation)
        self.assertAlmostEqual(reward, 65.949)
        self.assertEqual(done, False)
        self.assertEqual(info, {})
        self.assertEqual(self.env.profit_loss[-1], 0.0)
        self.assertEqual(self.env.cash[-1], 10000.0-9941.75)
        self.assertAlmostEqual(self.env.equity[-1], 10000.0 + 65.949)

        correct_position = (38, 261.625)
        self.assertEqual(self.env.positions[-1], correct_position)

    def test_simple_short_trade_of_100_percent(self):
        _ = self.env.reset(self._initialize_data())

        # go short 100%
        action = -1.0
        observation_, reward, done, info = self.env.step(action)

        correct_observation = np.array([261.68/(2*291.81), 263.5/(2*291.81), 261.59/(2*291.81), 263.3605/(2*291.81), 378093*10/TSLA_AVG_DAILY_VOLUME])

        np.testing.assert_array_equal(observation_, correct_observation)
        self.assertAlmostEqual(reward, -65.949)
        self.assertEqual(done, False)
        self.assertEqual(info, {})
        self.assertEqual(self.env.profit_loss[-1], 0.0)
        self.assertEqual(self.env.cash[-1], 10000.0-9941.75)
        self.assertAlmostEqual(self.env.equity[-1], 10000.0-65.949)

        correct_position = (-38, 261.625)
        self.assertEqual(self.env.positions[-1], correct_position)

    def test_add_to_long_position(self):
        _ = self.env.reset(self._initialize_data())

        # go long 50%
        action1 = 0.5
        self.env.step(action1)

        # increase long position by 20%
        action2 = 0.7
        observation_, reward, done, info = self.env.step(action2)

        correct_observation = np.array([263.425/(2*291.81), 264.49/(2*291.81), 262.87/(2*291.81), 263.5859/(2*291.81), 369398*10/TSLA_AVG_DAILY_VOLUME])

        np.testing.assert_array_equal(observation_, correct_observation)

        correct_reward = 5.8604
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, False)
        self.assertEqual(info, {})
        self.assertEqual(self.env.profit_loss[-1], 0.0)

        cash = 10000.0-6814.3985
        curr_stock_value = 26 * 263.3605
        self.assertEqual(self.env.cash[-1], cash)
        self.assertAlmostEqual(self.env.equity[-1], cash + curr_stock_value + correct_reward)

        # 8 more shares at average price of 263.3605
        correct_position = (26, 262.09225)
        self.assertEqual(self.env.positions[-1][0], correct_position[0])
        self.assertAlmostEqual(self.env.positions[-1][1], correct_position[1])

    def test_add_to_short_position(self):
        _ = self.env.reset(self._initialize_data())

        # go short 50%
        action1 = -0.5
        self.env.step(action1)

        # increase short position by 20%
        action2 = -0.7
        observation_, reward, done, info = self.env.step(action2)

        correct_observation = np.array([263.425/(2*291.81), 264.49/(2*291.81), 262.87/(2*291.81), 263.5859/(2*291.81), 369398*10/TSLA_AVG_DAILY_VOLUME])

        np.testing.assert_array_equal(observation_, correct_observation)

        correct_reward = -5.8604
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, False)
        self.assertEqual(info, {})
        self.assertEqual(self.env.profit_loss[-1], 0.0)

        
        cash = 10000-4970.875-1843.5235
        curr_stock_value = self._calculate_short_equity_value(26, 262.09225, 263.3605)
        self.assertEqual(self.env.cash[-1], cash)
        self.assertAlmostEqual(self.env.equity[-1], cash + curr_stock_value + correct_reward)

        # 7 more shares at average price of 263.3605
        correct_position = (-26, 262.09225)
        self.assertEqual(self.env.positions[-1][0], correct_position[0])
        self.assertAlmostEqual(self.env.positions[-1][1], correct_position[1])

    def test_reduce_long_position(self):
        _ = self.env.reset(self._initialize_data())

        # go long 50%
        action1 = 0.5
        self.env.step(action1)

        # decrease long position by 35%
        action2 = 0.15
        observation_, reward, done, info = self.env.step(action2)

        correct_observation = np.array([263.425/(2*291.81), 264.49/(2*291.81), 262.87/(2*291.81), 263.5859/(2*291.81), 369398*10/TSLA_AVG_DAILY_VOLUME])

        np.testing.assert_array_equal(observation_, correct_observation)

        correct_reward = 1.3524
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, False)
        self.assertEqual(info, {})
        self.assertAlmostEqual(self.env.profit_loss[-1], 22.5615)

        cash = 10000.0 - 4970.875 + 3423.6865
        curr_stock_value = 6 * 263.3605
        self.assertEqual(self.env.cash[-1], cash)
        self.assertAlmostEqual(self.env.equity[-1], cash + curr_stock_value + correct_reward)

        # selling 13 shares at 263.3605
        correct_position = (6, 261.625)
        self.assertEqual(self.env.positions[-1][0], correct_position[0])
        self.assertAlmostEqual(self.env.positions[-1][1], correct_position[1])

    def test_reduce_short_position(self):
        _ = self.env.reset(self._initialize_data())

        # go short 50%
        action1 = -0.5
        self.env.step(action1)

        # decrease short position by 35%
        action2 = -0.15
        observation_, reward, done, info = self.env.step(action2)

        correct_observation = np.array([263.425/(2*291.81), 264.49/(2*291.81), 262.87/(2*291.81), 263.5859/(2*291.81), 369398*10/TSLA_AVG_DAILY_VOLUME])

        np.testing.assert_array_equal(observation_, correct_observation)

        correct_reward = -1.3524
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, False)
        self.assertEqual(info, {})
        self.assertAlmostEqual(self.env.profit_loss[-1], -22.5615)

        cash = 10000.0 - 4970.875 + self._calculate_short_equity_value(13, 261.625, 263.3605)
        curr_stock_value = self._calculate_short_equity_value(6, 261.625, 263.3605)
        self.assertEqual(self.env.cash[-1], cash)
        self.assertAlmostEqual(self.env.equity[-1], cash + curr_stock_value + correct_reward)

        # selling 13 shares at 263.3605
        correct_position = (-6, 261.625)
        self.assertEqual(self.env.positions[-1][0], correct_position[0])
        self.assertAlmostEqual(self.env.positions[-1][1], correct_position[1])

    def test_close_long_position(self):
        _ = self.env.reset(self._initialize_data())

        # go long 50%
        action1 = 0.5
        self.env.step(action1)

        # close long position
        action2 = 0.0
        observation_, reward, done, info = self.env.step(action2)

        correct_observation = np.array([263.425/(2*291.81), 264.49/(2*291.81), 262.87/(2*291.81), 263.5859/(2*291.81), 369398*10/TSLA_AVG_DAILY_VOLUME])

        np.testing.assert_array_equal(observation_, correct_observation)

        correct_reward = 0.0
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, False)
        self.assertEqual(info, {})
        self.assertAlmostEqual(self.env.profit_loss[-1], 32.9745)

        cash = 10000.0 + self.env.profit_loss[-1]
        curr_stock_value = 0
        self.assertEqual(self.env.cash[-1], cash)
        self.assertAlmostEqual(self.env.equity[-1], cash + curr_stock_value + correct_reward)

        correct_position = (0, 0.0)
        self.assertEqual(self.env.positions[-1][0], correct_position[0])
        self.assertAlmostEqual(self.env.positions[-1][1], correct_position[1])

    def test_close_short_position(self):
        _ = self.env.reset(self._initialize_data())

        # go short 50%
        action1 = -0.5
        self.env.step(action1)

        # close short position
        action2 = 0.0
        observation_, reward, done, info = self.env.step(action2)

        correct_observation = np.array([263.425/(2*291.81), 264.49/(2*291.81), 262.87/(2*291.81), 263.5859/(2*291.81), 369398*10/TSLA_AVG_DAILY_VOLUME])

        np.testing.assert_array_equal(observation_, correct_observation)

        correct_reward = 0.0
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, False)
        self.assertEqual(info, {})
        self.assertAlmostEqual(self.env.profit_loss[-1], -32.9745)

        cash = 10000.0 + self.env.profit_loss[-1]
        curr_stock_value = 0
        self.assertEqual(self.env.cash[-1], cash)
        self.assertAlmostEqual(self.env.equity[-1], cash + curr_stock_value + correct_reward)

        correct_position = (0, 0.0)
        self.assertEqual(self.env.positions[-1][0], correct_position[0])
        self.assertAlmostEqual(self.env.positions[-1][1], correct_position[1])

    def test_cross_from_long_to_short(self):
        _ = self.env.reset(self._initialize_data())

        # go long 50%
        action1 = 0.5
        self.env.step(action1)

        # go short 85%
        action2 = -0.85
        observation_, reward, done, info = self.env.step(action2)

        correct_observation = np.array([263.425/(2*291.81), 264.49/(2*291.81), 262.87/(2*291.81), 263.5859/(2*291.81), 369398*10/TSLA_AVG_DAILY_VOLUME])

        np.testing.assert_array_equal(observation_, correct_observation)

        correct_reward = -7.2128
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, False)
        self.assertEqual(info, {})
        self.assertAlmostEqual(self.env.profit_loss[-1], 32.9745)

        curr_stock_value = 32 * 263.3605
        cash = 10000 + self.env.profit_loss[-1] - curr_stock_value
        self.assertAlmostEqual(self.env.cash[-1], cash)
        self.assertAlmostEqual(self.env.equity[-1], cash + curr_stock_value + correct_reward)

        # 8 more shares at average price of 263.3605
        correct_position = (-32, 263.3605)
        self.assertEqual(self.env.positions[-1][0], correct_position[0])
        self.assertAlmostEqual(self.env.positions[-1][1], correct_position[1])

    def test_cross_from_short_to_long(self):
        _ = self.env.reset(self._initialize_data())

        # go short 50%
        action1 = -0.5
        self.env.step(action1)

        # go long 85%
        action2 = 0.85
        observation_, reward, done, info = self.env.step(action2)

        correct_observation = np.array([263.425/(2*291.81), 264.49/(2*291.81), 262.87/(2*291.81), 263.5859/(2*291.81), 369398*10/TSLA_AVG_DAILY_VOLUME])

        np.testing.assert_array_equal(observation_, correct_observation)

        correct_reward = 7.2128
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, False)
        self.assertEqual(info, {})
        self.assertAlmostEqual(self.env.profit_loss[-1], -32.9745)

        curr_stock_value = 32 * 263.3605
        cash = 10000 + self.env.profit_loss[-1] - curr_stock_value
        self.assertAlmostEqual(self.env.cash[-1], cash)
        self.assertAlmostEqual(self.env.equity[-1], cash + curr_stock_value + correct_reward)

        # 8 more shares at average price of 263.3605
        correct_position = (32, 263.3605)
        self.assertEqual(self.env.positions[-1][0], correct_position[0])
        self.assertAlmostEqual(self.env.positions[-1][1], correct_position[1])

    def test_stay_long_50_percent_entire_episode(self):
        _ = self.env.reset(self._initialize_data())

        # go long 50%
        action = 0.5

        done = False
        while not done:
            observation_, reward, done, info = self.env.step(action)

        correct_observation = np.array([267.78/(2*291.81), 267.8/(2*291.81), 267.64/(2*291.81), 267.64/(2*291.81), 4627*10/TSLA_AVG_DAILY_VOLUME])

        np.testing.assert_array_equal(observation_, correct_observation)
        
        correct_reward = -3.04
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, True)
        self.assertEqual(info, {})
        self.assertAlmostEqual(self.env.profit_loss[-1], 0.0)

        cash = 10000.0-4970.875
        curr_stock_value = 19 * 267.64
        self.assertEqual(self.env.cash[-1], cash)
        self.assertAlmostEqual(self.env.equity[-1], cash + curr_stock_value)

        correct_position = (19, 261.625)
        self.assertEqual(self.env.positions[-1], correct_position)

    def test_stay_short_50_percent_entire_episode(self):
        _ = self.env.reset(self._initialize_data())

        # go short 50%
        action = -0.5

        done = False
        while not done:
            observation_, reward, done, info = self.env.step(action)

        correct_observation = np.array([267.78/(2*291.81), 267.8/(2*291.81), 267.64/(2*291.81), 267.64/(2*291.81), 4627*10/TSLA_AVG_DAILY_VOLUME])

        np.testing.assert_array_equal(observation_, correct_observation)

        correct_reward = 3.04
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, True)
        self.assertEqual(info, {})
        self.assertAlmostEqual(self.env.profit_loss[-1], 0.0)

        cash = 10000.0-4970.875
        curr_stock_value = (19 * (261.625 - (267.64 - 261.625)))
        self.assertEqual(self.env.cash[-1], cash)
        self.assertAlmostEqual(self.env.equity[-1], cash + curr_stock_value)

        correct_position = (-19, 261.625)
        self.assertEqual(self.env.positions[-1], correct_position)

    def _initialize_env(self):
        asset_data = self._initialize_data()
        return StockTradingEnv(asset_data)

    def _initialize_data(self):
        """Creates two DataFrames from csv file. One raw, one normalized.

        Returns:
            (pd.DataFrame, pd.DataFrame) -- 1min candle stick data (including volume) for 1 day
        """

        # make sure file is '.csv'
        path = '/Users/d/Documents/Projects/Python/gym-env/data/TSLA/'
        filename = 'TSLA2019-04-04.csv'

        # convert to data frame
        dataframe = pd.read_csv(path + filename)


        with open(path + filename[:-4] + '-prev_close.txt') as f:
            content = f.read()
            
        prev_close = float(content)
    
        # normalize  data
        normalized_dataframe = dataframe.copy()

        normalized_dataframe['open'] = normalized_dataframe['open'] / (2 * prev_close)
        normalized_dataframe['high'] = normalized_dataframe['high'] / (2 * prev_close)
        normalized_dataframe['low'] = normalized_dataframe['low'] / (2 * prev_close)
        normalized_dataframe['close'] = normalized_dataframe['close'] / (2 * prev_close)
        # Potential bug if volume in one minute is 1/10 of average daily volume
        normalized_dataframe['volume'] = normalized_dataframe['volume'] * 10 / TSLA_AVG_DAILY_VOLUME
         
        return (dataframe, normalized_dataframe)

    def _calculate_short_equity_value(self, shares, avg_price, curr_price):
        return abs(shares) * (avg_price - (curr_price - avg_price))

if __name__ == '__main__':
    unittest.main()