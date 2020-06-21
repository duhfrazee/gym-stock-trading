import unittest
import os

import gym
import numpy as np
import pandas as pd

# test submit order for qty pos, neg, and 0
# test close position
# test websocket
# test correct observations
# test await market open (already created)
# test more without volume enabled

# test several episodes
# test generators!!


class TestStockTradingEnv(unittest.TestCase):
    # def setUp(self):
    #     self.previous_close = 291.81
    #     self.daily_avg_volume = 31140787

    #     self.market_data = self._yield_test_market_data(['TSLA2019-04-04.csv'])
    #     self.previous_closes = self._yield_test_previous_closes([self.previous_close])
    #     self.env = gym.make(
    #         'gym_stock_trading:StockTrading-v0',
    #         market_data=self.market_data,
    #         previous_closes=self.previous_closes,
    #         daily_avg_volume=self.daily_avg_volume
    #     )

    #     step0_open = 261.89
    #     step0_high = 262.77
    #     step0_low = 260.59
    #     step0_close = 261.625
    #     step0_volume = 706650

    #     self.correct_step0_obs = np.array([
    #         [step0_open / (2*self.previous_close)],
    #         [step0_high / (2*self.previous_close)],
    #         [step0_low / (2*self.previous_close)],
    #         [step0_close / (2*self.previous_close)],
    #         [step0_volume / self.daily_avg_volume]
    #     ])

    #     step1_open = 261.68
    #     step1_high = 263.5
    #     step1_low = 261.59
    #     step1_close = 263.3605
    #     step1_volume = 378093

    #     self.correct_step1_obs = np.array([
    #         [step1_open / (2*self.previous_close)],
    #         [step1_high / (2*self.previous_close)],
    #         [step1_low / (2*self.previous_close)],
    #         [step1_close / (2*self.previous_close)],
    #         [step1_volume / self.daily_avg_volume]
    #     ])

    #     step2_open = 263.425
    #     step2_high = 264.49
    #     step2_low = 262.87
    #     step2_close = 263.5859
    #     step2_volume = 369398

    #     self.correct_step2_obs = np.array([
    #         [step2_open / (2*self.previous_close)],
    #         [step2_high / (2*self.previous_close)],
    #         [step2_low / (2*self.previous_close)],
    #         [step2_close / (2*self.previous_close)],
    #         [step2_volume / self.daily_avg_volume]
    #     ])

    #     laststep_open = 267.78
    #     laststep_high = 267.8
    #     laststep_low = 267.64
    #     laststep_close = 267.64
    #     laststep_volume = 4627

    #     self.correct_laststep_obs = np.array([
    #         [laststep_open / (2*self.previous_close)],
    #         [laststep_high / (2*self.previous_close)],
    #         [laststep_low / (2*self.previous_close)],
    #         [laststep_close / (2*self.previous_close)],
    #         [laststep_volume / self.daily_avg_volume]
    #     ])

    def passing_test(self):
        assert 1 == 1

    # def test_inititalize_env(self):

    #     initial_equity = self.env.equity[-1]
    #     profit_loss = self.env.profit_loss[-1]
    #     initial_cash = self.env.cash[-1]
    #     intitial_position = self.env.positions[-1]
    #     reward = self.env.rewards[-1]
    #     max_qty = self.env.max_qty

    #     self.assertEqual(self.env.current_step, 0)
    #     self.assertEqual(self.env.current_episode, 0)
    #     self.assertEqual(self.env.asset_data, None)
    #     self.assertEqual(self.env.normalized_asset_data, None)
    #     self.assertEqual(self.env.observation_size, 1)
    #     self.assertEqual(self.env.base_value, 10000)
    #     self.assertEqual(initial_equity, self.env.base_value)
    #     self.assertEqual(initial_cash, self.env.base_value)
    #     self.assertEqual(profit_loss, 0.0)
    #     self.assertEqual(intitial_position, (0, 0.0))
    #     self.assertEqual(reward, 0.0)
    #     self.assertEqual(max_qty, None)

    #     # self.assertEqual(type(self.env.market_data), generator)

    # def test_default_reset(self):
    #     obs = self._reset_env()

    #     initial_equity = self.env.equity[-1]
    #     profit_loss = self.env.profit_loss[-1]
    #     initial_cash = self.env.cash[-1]
    #     intitial_position = self.env.positions[-1]
    #     reward = self.env.rewards[-1]
    #     max_qty = self.env.max_qty

    #     self.assertEqual(self.env.previous_close, self.previous_close)
    #     self.assertEqual(
    #         self.env.daily_avg_volume, self.daily_avg_volume)
    #     self.assertEqual(self.env.current_step, 0)
    #     self.assertEqual(self.env.observation_size, 1)
    #     self.assertEqual(self.env.base_value, 10000)
    #     self.assertEqual(initial_equity, self.env.base_value)
    #     self.assertEqual(initial_cash, self.env.base_value)
    #     self.assertEqual(profit_loss, 0.0)
    #     self.assertEqual(intitial_position, (0, 0.0))
    #     self.assertEqual(reward, 0.0)
    #     self.assertEqual(max_qty, 38)

    #     np.testing.assert_array_equal(obs, self.correct_step0_obs)

    # def test_reset_for_observation_size_2_to_10(self):

    #     for i in range(2, 11):
    #         self.env = gym.make(
    #             'gym_stock_trading:StockTrading-v0',
    #             market_data=self.market_data,
    #             previous_closes=self.previous_closes,
    #             daily_avg_volume=self.daily_avg_volume,
    #             observation_size=i
    #         )

    #         obs = self._reset_env()

    #         initial_equity = self.env.equity[-1]
    #         profit_loss = self.env.profit_loss[-1]
    #         initial_cash = self.env.cash[-1]
    #         intitial_position = self.env.positions[-1]
    #         reward = self.env.rewards[-1]
    #         max_qty = self.env.max_qty

    #         self.assertEqual(self.env.current_step, 0)
    #         self.assertEqual(self.env.observation_size, i)
    #         self.assertEqual(self.env.base_value, 10000)
    #         self.assertEqual(initial_equity, self.env.base_value)
    #         self.assertEqual(initial_cash, self.env.base_value)
    #         self.assertEqual(profit_loss, 0.0)
    #         self.assertEqual(intitial_position, (0, 0.0))
    #         self.assertEqual(reward, 0.0)
    #         self.assertEqual(max_qty, 38)

    #         correct_obs_zeros = np.zeros([5, i-1])
    #         correct_obs = np.concatenate(
    #             (self.correct_step0_obs, correct_obs_zeros), axis=1)

    #         np.testing.assert_array_equal(obs, correct_obs)
    #         self.assertEqual(obs.shape[1], i)

    # def test_step1_for_observation_size_2_to_10(self):

    #     for i in range(2, 11):
    #         self.env = gym.make(
    #             'gym_stock_trading:StockTrading-v0',
    #             market_data=self.market_data,
    #             previous_closes=self.previous_closes,
    #             daily_avg_volume=self.daily_avg_volume,
    #             observation_size=i
    #         )

    #         _ = self._reset_env()

    #         # Go long 50%
    #         action = np.array([0.5])
    #         observation_, _, _, _ = self.env.step(action)

    #         correct_obs_zeros = np.zeros([5, abs(2 - i)])
    #         correct_obs = np.concatenate(
    #             (self.correct_step0_obs, self.correct_step1_obs), axis=1)
    #         correct_obs = np.concatenate(
    #             (correct_obs, correct_obs_zeros), axis=1)

    #         np.testing.assert_array_equal(observation_, correct_obs)
    #         self.assertEqual(observation_.shape[1], i)

    # def test_observation_size_2_to_10_for_step_0_to_10(self):

    #     for obs_size in range(2, 11):
    #         self.env = gym.make(
    #             'gym_stock_trading:StockTrading-v0',
    #             market_data=self.market_data,
    #             previous_closes=self.previous_closes,
    #             daily_avg_volume=self.daily_avg_volume,
    #             observation_size=obs_size
    #         )

    #         obs = self._reset_env()

    #         correct_obs_zeros = np.zeros([5, obs_size-1])
    #         correct_obs = np.concatenate(
    #             (self.correct_step0_obs, correct_obs_zeros), axis=1)

    #         np.testing.assert_array_equal(obs, correct_obs)

    #         for step in range(1, 11):

    #             # Go long 50%
    #             action = np.array([0.5])
    #             observation_, _, _, _ = self.env.step(action)

    #             offset = step+1 - obs_size

    #             if offset < 0:
    #                 # Less data than observation_size
    #                 observation_zeros = np.zeros([5, abs(offset)])
    #                 offset = 0

    #             correct_obs = np.array([
    #                 self.env.normalized_asset_data.iloc[
    #                     offset: step+1]['open'].values,
    #                 self.env.normalized_asset_data.iloc[
    #                     offset: step+1]['high'].values,
    #                 self.env.normalized_asset_data.iloc[
    #                     offset: step+1]['low'].values,
    #                 self.env.normalized_asset_data.iloc[
    #                     offset: step+1]['close'].values,
    #                 self.env.normalized_asset_data.iloc[
    #                     offset: step+1]['volume'].values,
    #             ])

    #             if correct_obs.shape[1] < obs_size:
    #                 correct_obs = np.concatenate(
    #                     (correct_obs, observation_zeros), axis=1)

    #             np.testing.assert_array_equal(observation_, correct_obs)
    #             self.assertEqual(observation_.shape[1], obs_size)

    # def test_simple_long_trade_of_50_percent(self):
    #     _ = self._reset_env()

    #     # Go long 50%
    #     action = np.array([0.5])
    #     observation_, reward, done, info = self.env.step(action)

    #     np.testing.assert_array_equal(observation_, self.correct_step1_obs)

    #     correct_reward = 32.9745
    #     self.assertAlmostEqual(reward, correct_reward)
    #     self.assertEqual(done, False)
    #     self.assertEqual(info, {})
    #     self.assertEqual(self.env.profit_loss[-1], 0.0)

    #     correct_position = (19, 261.625)
    #     self.assertEqual(self.env.positions[-1], correct_position)

    #     purchase_amount = correct_position[0] * correct_position[1]
    #     self.assertEqual(self.env.cash[-1], 10000.0 - purchase_amount)
    #     self.assertAlmostEqual(self.env.equity[-1], 10000.0 + correct_reward)

    # def test_simple_short_trade_of_50_percent(self):
    #     _ = self._reset_env()

    #     # Go short 50%
    #     action = np.array([-0.5])
    #     observation_, reward, done, info = self.env.step(action)

    #     np.testing.assert_array_equal(observation_, self.correct_step1_obs)

    #     correct_reward = -32.9745
    #     self.assertAlmostEqual(reward, correct_reward)
    #     self.assertEqual(done, False)
    #     self.assertEqual(info, {})
    #     self.assertEqual(self.env.profit_loss[-1], 0.0)

    #     correct_position = (-19, 261.625)
    #     self.assertEqual(self.env.positions[-1], correct_position)

    #     purchase_amount = abs(correct_position[0] * correct_position[1])
    #     self.assertEqual(self.env.cash[-1], 10000.0 - purchase_amount)
    #     self.assertAlmostEqual(self.env.equity[-1], 10000.0 + correct_reward)

    # def test_simple_long_trade_of_65_percent(self):
    #     _ = self._reset_env()

    #     # Go long 65%
    #     action = np.array([0.65])
    #     observation_, reward, done, info = self.env.step(action)

    #     np.testing.assert_array_equal(observation_, self.correct_step1_obs)

    #     correct_reward = 41.652
    #     self.assertAlmostEqual(reward, correct_reward)
    #     self.assertEqual(done, False)
    #     self.assertEqual(info, {})
    #     self.assertEqual(self.env.profit_loss[-1], 0.0)

    #     correct_position = (24, 261.625)
    #     self.assertEqual(self.env.positions[-1], correct_position)

    #     purchase_amount = correct_position[0] * correct_position[1]
    #     self.assertEqual(self.env.cash[-1], 10000.0 - purchase_amount)
    #     self.assertAlmostEqual(self.env.equity[-1], 10000.0 + correct_reward)

    # def test_simple_short_trade_of_65_percent(self):
    #     _ = self._reset_env()

    #     # Go short 65%
    #     action = np.array([-0.65])
    #     observation_, reward, done, info = self.env.step(action)

    #     np.testing.assert_array_equal(observation_, self.correct_step1_obs)

    #     correct_reward = -41.652
    #     self.assertAlmostEqual(reward, correct_reward)
    #     self.assertEqual(done, False)
    #     self.assertEqual(info, {})
    #     self.assertEqual(self.env.profit_loss[-1], 0.0)

    #     correct_position = (-24, 261.625)
    #     self.assertEqual(self.env.positions[-1], correct_position)

    #     purchase_amount = abs(correct_position[0] * correct_position[1])
    #     self.assertEqual(self.env.cash[-1], 10000.0 - purchase_amount)
    #     self.assertAlmostEqual(self.env.equity[-1], 10000.0 + correct_reward)

    # def test_simple_long_trade_of_100_percent(self):
    #     _ = self._reset_env()

    #     # Go long 100%
    #     action = np.array([1.0])
    #     observation_, reward, done, info = self.env.step(action)

    #     np.testing.assert_array_equal(observation_, self.correct_step1_obs)

    #     correct_reward = 65.949
    #     self.assertAlmostEqual(reward, correct_reward)
    #     self.assertEqual(done, False)
    #     self.assertEqual(info, {})
    #     self.assertEqual(self.env.profit_loss[-1], 0.0)

    #     correct_position = (38, 261.625)
    #     self.assertEqual(self.env.positions[-1], correct_position)

    #     purchase_amount = correct_position[0] * correct_position[1]
    #     self.assertEqual(self.env.cash[-1], 10000.0 - purchase_amount)
    #     self.assertAlmostEqual(self.env.equity[-1], 10000.0 + correct_reward)

    # def test_simple_short_trade_of_100_percent(self):
    #     _ = self._reset_env()

    #     # Go short 100%
    #     action = np.array([-1.0])
    #     observation_, reward, done, info = self.env.step(action)

    #     np.testing.assert_array_equal(observation_, self.correct_step1_obs)

    #     correct_reward = -65.949
    #     self.assertAlmostEqual(reward, correct_reward)
    #     self.assertEqual(done, False)
    #     self.assertEqual(info, {})
    #     self.assertEqual(self.env.profit_loss[-1], 0.0)

    #     correct_position = (-38, 261.625)
    #     self.assertEqual(self.env.positions[-1], correct_position)

    #     purchase_amount = abs(correct_position[0] * correct_position[1])
    #     self.assertEqual(self.env.cash[-1], 10000.0 - purchase_amount)
    #     self.assertAlmostEqual(self.env.equity[-1], 10000.0 + correct_reward)

    # def test_add_to_long_position(self):
    #     _ = self._reset_env()

    #     # Go long 50%
    #     action1 = np.array([0.5])
    #     self.env.step(action1)

    #     # Increase long position by 20%
    #     action2 = np.array([0.7])
    #     observation_, reward, done, info = self.env.step(action2)

    #     np.testing.assert_array_equal(observation_, self.correct_step2_obs)

    #     correct_reward = 5.8604
    #     self.assertAlmostEqual(reward, correct_reward)
    #     self.assertEqual(done, False)
    #     self.assertEqual(info, {})
    #     self.assertEqual(self.env.profit_loss[-1], 0.0)

    #     # 8 more shares at average price of 263.3605
    #     correct_position = (26, 262.09225)
    #     self.assertEqual(self.env.positions[-1][0], correct_position[0])
    #     self.assertAlmostEqual(self.env.positions[-1][1], correct_position[1])

    #     cash = 10000.0-6814.3985
    #     curr_stock_value = 26 * 263.3605
    #     self.assertEqual(self.env.cash[-1], cash)
    #     self.assertAlmostEqual(
    #         self.env.equity[-1], cash + curr_stock_value + correct_reward)

    # def test_add_to_short_position(self):
    #     _ = self._reset_env()

    #     # Go short 50%
    #     action1 = np.array([-0.5])
    #     self.env.step(action1)

    #     # Increase short position by 20%
    #     action2 = np.array([-0.7])
    #     observation_, reward, done, info = self.env.step(action2)

    #     np.testing.assert_array_equal(observation_, self.correct_step2_obs)

    #     correct_reward = -5.8604
    #     self.assertAlmostEqual(reward, correct_reward)
    #     self.assertEqual(done, False)
    #     self.assertEqual(info, {})
    #     self.assertEqual(self.env.profit_loss[-1], 0.0)

    #     cash = 10000 - 4970.875 - 1843.5235
    #     curr_stock_value = self._calculate_short_equity_value(
    #         26, 262.09225, 263.3605)
    #     self.assertEqual(self.env.cash[-1], cash)
    #     self.assertAlmostEqual(
    #         self.env.equity[-1], cash + curr_stock_value + correct_reward)

    #     # 7 more shares at average price of 263.3605
    #     correct_position = (-26, 262.09225)
    #     self.assertEqual(self.env.positions[-1][0], correct_position[0])
    #     self.assertAlmostEqual(self.env.positions[-1][1], correct_position[1])

    # def test_reduce_long_position(self):
    #     _ = self._reset_env()

    #     # Go long 50%
    #     action1 = np.array([0.5])
    #     self.env.step(action1)

    #     # Decrease long position by 35%
    #     action2 = np.array([0.15])
    #     observation_, reward, done, info = self.env.step(action2)

    #     np.testing.assert_array_equal(observation_, self.correct_step2_obs)

    #     correct_reward = 1.3524
    #     self.assertAlmostEqual(reward, correct_reward)
    #     self.assertEqual(done, False)
    #     self.assertEqual(info, {})
    #     self.assertAlmostEqual(self.env.profit_loss[-1], 22.5615)

    #     cash = 10000.0 - 4970.875 + 3423.6865
    #     curr_stock_value = 6 * 263.3605
    #     self.assertEqual(self.env.cash[-1], cash)
    #     self.assertAlmostEqual(
    #         self.env.equity[-1], cash + curr_stock_value + correct_reward)

    #     # Selling 13 shares at 263.3605
    #     correct_position = (6, 261.625)
    #     self.assertEqual(self.env.positions[-1][0], correct_position[0])
    #     self.assertAlmostEqual(
    #         self.env.positions[-1][1], correct_position[1])

    # def test_reduce_short_position(self):
    #     _ = self._reset_env()

    #     # Go short 50%
    #     action1 = np.array([-0.5])
    #     self.env.step(action1)

    #     # Decrease short position by 35%
    #     action2 = np.array([-0.15])
    #     observation_, reward, done, info = self.env.step(action2)

    #     np.testing.assert_array_equal(observation_, self.correct_step2_obs)

    #     correct_reward = -1.3524
    #     self.assertAlmostEqual(reward, correct_reward)
    #     self.assertEqual(done, False)
    #     self.assertEqual(info, {})
    #     self.assertAlmostEqual(self.env.profit_loss[-1], -22.5615)

    #     cash = 10000.0 - 4970.875\
    #         + self._calculate_short_equity_value(13, 261.625, 263.3605)
    #     curr_stock_value = self._calculate_short_equity_value(
    #         6, 261.625, 263.3605)
    #     self.assertEqual(self.env.cash[-1], cash)
    #     self.assertAlmostEqual(
    #         self.env.equity[-1], cash + curr_stock_value + correct_reward)

    #     # Selling 13 shares at 263.3605
    #     correct_position = (-6, 261.625)
    #     self.assertEqual(self.env.positions[-1][0], correct_position[0])
    #     self.assertAlmostEqual(self.env.positions[-1][1], correct_position[1])

    # def test_close_long_position(self):
    #     _ = self._reset_env()

    #     # Go long 50%
    #     action1 = np.array([0.5])
    #     self.env.step(action1)

    #     # Close long position
    #     action2 = np.array([0.0])
    #     observation_, reward, done, info = self.env.step(action2)

    #     np.testing.assert_array_equal(observation_, self.correct_step2_obs)

    #     correct_reward = 0.0
    #     self.assertAlmostEqual(reward, correct_reward)
    #     self.assertEqual(done, False)
    #     self.assertEqual(info, {})
    #     self.assertAlmostEqual(self.env.profit_loss[-1], 32.9745)

    #     cash = 10000.0 + self.env.profit_loss[-1]
    #     curr_stock_value = 0
    #     self.assertEqual(self.env.cash[-1], cash)
    #     self.assertAlmostEqual(
    #         self.env.equity[-1], cash + curr_stock_value + correct_reward)

    #     correct_position = (0, 0.0)
    #     self.assertEqual(self.env.positions[-1][0], correct_position[0])
    #     self.assertAlmostEqual(
    #         self.env.positions[-1][1], correct_position[1])

    # def test_close_short_position(self):
    #     _ = self._reset_env()

    #     # Go short 50%
    #     action1 = np.array([-0.5])
    #     self.env.step(action1)

    #     # Close short position
    #     action2 = np.array([0.0])
    #     observation_, reward, done, info = self.env.step(action2)

    #     np.testing.assert_array_equal(observation_, self.correct_step2_obs)

    #     correct_reward = 0.0
    #     self.assertAlmostEqual(reward, correct_reward)
    #     self.assertEqual(done, False)
    #     self.assertEqual(info, {})
    #     self.assertAlmostEqual(self.env.profit_loss[-1], -32.9745)

    #     cash = 10000.0 + self.env.profit_loss[-1]
    #     curr_stock_value = 0
    #     self.assertEqual(self.env.cash[-1], cash)
    #     self.assertAlmostEqual(
    #         self.env.equity[-1], cash + curr_stock_value + correct_reward)

    #     correct_position = (0, 0.0)
    #     self.assertEqual(self.env.positions[-1][0], correct_position[0])
    #     self.assertAlmostEqual(
    #         self.env.positions[-1][1], correct_position[1])

    # def test_cross_from_long_to_short(self):
    #     _ = self._reset_env()

    #     # Go long 50%
    #     action1 = np.array([0.5])
    #     self.env.step(action1)

    #     # Go short 85%
    #     action2 = np.array([-0.85])
    #     observation_, reward, done, info = self.env.step(action2)

    #     np.testing.assert_array_equal(observation_, self.correct_step2_obs)

    #     correct_reward = -7.2128
    #     self.assertAlmostEqual(reward, correct_reward)
    #     self.assertEqual(done, False)
    #     self.assertEqual(info, {})
    #     self.assertAlmostEqual(self.env.profit_loss[-1], 32.9745)

    #     curr_stock_value = 32 * 263.3605
    #     cash = 10000 + self.env.profit_loss[-1] - curr_stock_value
    #     self.assertAlmostEqual(self.env.cash[-1], cash)
    #     self.assertAlmostEqual(
    #         self.env.equity[-1], cash + curr_stock_value + correct_reward)

    #     # 8 more shares at average price of 263.3605
    #     correct_position = (-32, 263.3605)
    #     self.assertEqual(self.env.positions[-1][0], correct_position[0])
    #     self.assertAlmostEqual(self.env.positions[-1][1], correct_position[1])

    # def test_cross_from_short_to_long(self):
    #     _ = self._reset_env()

    #     # Go short 50%
    #     action1 = np.array([-0.5])
    #     self.env.step(action1)

    #     # Go long 85%
    #     action2 = np.array([0.85])
    #     observation_, reward, done, info = self.env.step(action2)

    #     np.testing.assert_array_equal(observation_, self.correct_step2_obs)

    #     correct_reward = 7.2128
    #     self.assertAlmostEqual(reward, correct_reward)
    #     self.assertEqual(done, False)
    #     self.assertEqual(info, {})
    #     self.assertAlmostEqual(self.env.profit_loss[-1], -32.9745)

    #     curr_stock_value = 32 * 263.3605
    #     cash = 10000 + self.env.profit_loss[-1] - curr_stock_value
    #     self.assertAlmostEqual(self.env.cash[-1], cash)
    #     self.assertAlmostEqual(
    #         self.env.equity[-1], cash + curr_stock_value + correct_reward)

    #     # 8 more shares at average price of 263.3605
    #     correct_position = (32, 263.3605)
    #     self.assertEqual(self.env.positions[-1][0], correct_position[0])
    #     self.assertAlmostEqual(self.env.positions[-1][1], correct_position[1])

    # def test_stay_long_50_percent_entire_episode(self):
    #     _ = self._reset_env()

    #     # Go long 50%
    #     action = np.array([0.5])

    #     done = False
    #     while not done:
    #         observation_, reward, done, info = self.env.step(action)

    #     np.testing.assert_array_equal(observation_, self.correct_laststep_obs)

    #     correct_reward = -3.04
    #     self.assertAlmostEqual(reward, correct_reward)
    #     self.assertEqual(done, True)
    #     self.assertEqual(info, {})
    #     self.assertAlmostEqual(self.env.profit_loss[-1], 0.0)

    #     correct_position = (19, 261.625)
    #     self.assertEqual(self.env.positions[-1], correct_position)

    #     purchase_amount = correct_position[0] * correct_position[1]
    #     cash = 10000.0 - purchase_amount
    #     curr_stock_value = 19 * 267.64
    #     self.assertEqual(self.env.cash[-1], cash)
    #     self.assertAlmostEqual(self.env.equity[-1], cash + curr_stock_value)

    # def test_stay_short_50_percent_entire_episode(self):
    #     _ = self._reset_env()

    #     # Go short 50%
    #     action = np.array([-0.5])

    #     done = False
    #     while not done:
    #         observation_, reward, done, info = self.env.step(action)

    #     np.testing.assert_array_equal(observation_, self.correct_laststep_obs)

    #     correct_reward = 3.04
    #     self.assertAlmostEqual(reward, correct_reward)
    #     self.assertEqual(done, True)
    #     self.assertEqual(info, {})
    #     self.assertAlmostEqual(self.env.profit_loss[-1], 0.0)

    #     correct_position = (-19, 261.625)
    #     self.assertEqual(self.env.positions[-1], correct_position)

    #     purchase_amount = abs(correct_position[0] * correct_position[1])
    #     cash = 10000.0 - purchase_amount
    #     curr_stock_value = self._calculate_short_equity_value(
    #         19, 261.625, 267.64)
    #     self.assertEqual(self.env.cash[-1], cash)
    #     self.assertAlmostEqual(self.env.equity[-1], cash + curr_stock_value)

    # def test_previous_close_over_several_episodes(self):
    #     previous_closes = [279.18, 252.48, 267.53, 301.15, 342.95]
    #     files = [
    #         'TSLA2018-03-28.csv',
    #         'TSLA2018-04-03.csv',
    #         'TSLA2018-04-04.csv',
    #         'TSLA2018-05-03.csv',
    #         'TSLA2018-07-02.csv'
    #     ]
    #     self.env = gym.make(
    #         'gym_stock_trading:StockTrading-v0',
    #         market_data=self._yield_test_market_data(files),
    #         previous_closes=self._yield_test_previous_closes(previous_closes),
    #         daily_avg_volume=self.daily_avg_volume
    #     )

    #     for correct_close in previous_closes:
    #         done = False
    #         self.env.reset()
    #         while not done:
    #             _, _, done, _ = self.env.step(np.array([0.0]))
    #         self.assertEqual(self.env.previous_close, correct_close)

    # def _yield_test_market_data(self, files):

    #     while True:
    #         for filename in files:
    #             path =\
    #                 '/Users/d/Documents/Projects/Python/openai/gym-stock-trading/tests/test_data/'
    #             # Convert to data frame
    #             asset_data = pd.read_csv(path + filename)
    #             asset_data['timestamp'] = pd.to_datetime(asset_data['timestamp'])
    #             asset_data.set_index('timestamp', inplace=True)
    #             yield asset_data

    # def _yield_test_previous_closes(self, previous_closes):
    #     while True:
    #         for previous_close in previous_closes:
    #             yield previous_close

    # def _calculate_short_equity_value(self, shares, avg_price, curr_price):
    #     return abs(shares) * (avg_price - (curr_price - avg_price))

    # def _reset_env(self):
        return self.env.reset()


if __name__ == '__main__':
    unittest.main()
