import os
import unittest

import gym
import numpy as np
import pandas as pd
import pytest

from tests.test_data.AAPL import stock_data
# test submit order for qty pos, neg, and 0
# test close position
# test websocket
# test correct observations
# test await market open (already created)
# test more without volume enabled

# test several episodes
# test generators!!


class TestStockTradingEnv(unittest.TestCase):

    def setUp(self):
        self.correct_max_shares = 34
        self.volume_enabled = True
        self.previous_close = 293.8
        self.daily_avg_volume = 34300000
        self.normalized_asset_data = self._normalize_dataframe(
            stock_data.AAPL_MAY_1_2020_MARKET_DATA
        )

        self.market_data = self._yield_market_data(
            stock_data.AAPL_MAY_1_2020_MARKET_DATA
        )
        self.env = gym.make(
            'gym_stock_trading:StockTrading-v0',
            market_data=self.market_data,
            daily_avg_volume=self.daily_avg_volume
        )

    def test_inititalize_env(self):

        initial_equity = self.env.equity[-1]
        profit_loss = self.env.profit_loss[-1]
        initial_cash = self.env.cash[-1]
        intitial_position = self.env.positions[-1]
        reward = self.env.rewards[-1]
        max_qty = self.env.max_qty

        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.current_episode, 0)
        self.assertEqual(self.env.asset_data, None)
        self.assertEqual(self.env.normalized_asset_data, None)
        self.assertEqual(self.env.observation_size, 1)
        self.assertEqual(self.env.base_value, 10000)
        self.assertEqual(initial_equity, self.env.base_value)
        self.assertEqual(initial_cash, self.env.base_value)
        self.assertEqual(profit_loss, 0.0)
        self.assertEqual(intitial_position, (0, 0.0))
        self.assertEqual(reward, 0.0)
        self.assertEqual(max_qty, None)

        # self.assertEqual(type(self.env.market_data), generator)

    def test_default_reset(self):
        obs = self._reset_env()

        initial_equity = self.env.equity[-1]
        profit_loss = self.env.profit_loss[-1]
        initial_cash = self.env.cash[-1]
        intitial_position = self.env.positions[-1]
        reward = self.env.rewards[-1]
        max_qty = self.env.max_qty

        self.assertEqual(self.env.previous_close, self.previous_close)
        self.assertEqual(
            self.env.daily_avg_volume,
            self.daily_avg_volume
        )
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.observation_size, 1)
        self.assertEqual(self.env.base_value, 10000)
        self.assertEqual(initial_equity, self.env.base_value)
        self.assertEqual(initial_cash, self.env.base_value)
        self.assertEqual(profit_loss, 0.0)
        self.assertEqual(intitial_position, (0, 0.0))
        self.assertEqual(reward, 0.0)
        self.assertEqual(max_qty, 34)

        correct_obs = self._get_correct_observation(0, 1)
        np.testing.assert_array_equal(
            obs,
            correct_obs
        )

    def test_reset_for_observation_size_2_to_400(self):

        for i in range(2, 401):
            self.env = gym.make(
                'gym_stock_trading:StockTrading-v0',
                market_data=self.market_data,
                daily_avg_volume=self.daily_avg_volume,
                observation_size=i
            )

            obs = self._reset_env()

            initial_equity = self.env.equity[-1]
            profit_loss = self.env.profit_loss[-1]
            initial_cash = self.env.cash[-1]
            intitial_position = self.env.positions[-1]
            reward = self.env.rewards[-1]
            max_qty = self.env.max_qty

            self.assertEqual(self.env.current_step, 0)
            self.assertEqual(self.env.observation_size, i)
            self.assertEqual(self.env.base_value, 10000)
            self.assertEqual(initial_equity, self.env.base_value)
            self.assertEqual(initial_cash, self.env.base_value)
            self.assertEqual(profit_loss, 0.0)
            self.assertEqual(intitial_position, (0, 0.0))
            self.assertEqual(reward, 0.0)
            self.assertEqual(max_qty, self.correct_max_shares)

            correct_obs = self._get_correct_observation(0, i)
            np.testing.assert_array_equal(obs, correct_obs)
            self.assertEqual(obs.shape[1], i)

    def test_step1_for_observation_size_2_to_4OO(self):

        for i in range(2, 401):
            self.env = gym.make(
                'gym_stock_trading:StockTrading-v0',
                market_data=self.market_data,
                daily_avg_volume=self.daily_avg_volume,
                observation_size=i
            )

            _ = self._reset_env()

            action = np.array([0.0])
            observation_, _, _, _ = self.env.step(action)
            
            correct_obs = self._get_correct_observation(1, i)

            np.testing.assert_array_equal(observation_, correct_obs)
            self.assertEqual(observation_.shape[1], i)

    def test_observation_size_2_to_400_for_step_1_to_done(self):

        for obs_size in range(2, 401):
            self.env = gym.make(
                'gym_stock_trading:StockTrading-v0',
                market_data=self.market_data,
                daily_avg_volume=self.daily_avg_volume,
                observation_size=obs_size
            )

            obs = self._reset_env()
            step = 1
            done = False
            while not done:

                action = np.array([0.0])
                observation_, _, done, _ = self.env.step(action)

                correct_obs = self._get_correct_observation(step, obs_size)

                np.testing.assert_array_equal(observation_, correct_obs)
                self.assertEqual(observation_.shape[1], obs_size)

                step += 1

    def test_simple_long_trade_of_50_percent_for_1_step(self):
        _ = self._reset_env()
        step0_close = 289.2786
        step1_close = 289.04

        # Go long 50%
        action = np.array([0.5])
        _, reward, done, _ = self.env.step(action)

        correct_reward = -4.0562
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, False)
        self.assertEqual(self.env.profit_loss, [0.0, 0.0])

        correct_position = (17, 289.2786)
        self.assertEqual(self.env.positions, [(0, 0.0), correct_position])

        purchase_amount = correct_position[0] * correct_position[1]
        self.assertEqual(self.env.cash[-1], 10000.0 - purchase_amount)
        self.assertAlmostEqual(self.env.equity[-1], 10000.0 + correct_reward)

    def test_simple_short_trade_of_50_percent_for_1_step(self):
        _ = self._reset_env()

        # Go short 50%
        action = np.array([-0.5])
        _, reward, done, _ = self.env.step(action)

        correct_reward = 4.0562
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, False)
        self.assertEqual(self.env.profit_loss, [0.0, 0.0])

        correct_position = (-17, 289.2786)
        self.assertEqual(self.env.positions, [(0, 0.0), correct_position])

        purchase_amount = abs(correct_position[0] * correct_position[1])
        self.assertEqual(self.env.cash[-1], 10000.0 - purchase_amount)
        self.assertAlmostEqual(self.env.equity[-1], 10000.0 + correct_reward)

    def test_simple_long_trade_of_65_percent(self):
        _ = self._reset_env()
        step0_close = 289.2786
        step1_close = 289.04

        # Go long 65%
        action = np.array([0.65])
        _, reward, done, _ = self.env.step(action)

        correct_reward = (step1_close - step0_close) * 22
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, False)
        self.assertEqual(self.env.profit_loss, [0.0, 0.0])

        correct_position = (22, 289.2786)
        self.assertEqual(self.env.positions[-1], correct_position)

        purchase_amount = correct_position[0] * correct_position[1]
        self.assertEqual(self.env.cash[-1], 10000.0 - purchase_amount)
        self.assertAlmostEqual(self.env.equity[-1], 10000.0 + correct_reward)

    def test_simple_short_trade_of_65_percent(self):
        _ = self._reset_env()
        step0_close = 289.2786
        step1_close = 289.04

        # Go short 65%
        action = np.array([-0.65])
        _, reward, done, _ = self.env.step(action)

        correct_reward = (step1_close - step0_close) * -22
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, False)
        self.assertEqual(self.env.profit_loss[-1], 0.0)

        correct_position = (-22, 289.2786)
        self.assertEqual(self.env.positions[-1], correct_position)

        purchase_amount = abs(correct_position[0] * correct_position[1])
        self.assertEqual(self.env.cash[-1], 10000.0 - purchase_amount)
        self.assertAlmostEqual(self.env.equity[-1], 10000.0 + correct_reward)

    def test_simple_long_trade_of_100_percent(self):
        _ = self._reset_env()
        step0_close = 289.2786
        step1_close = 289.04

        # Go long 100%
        action = np.array([1.0])
        _, reward, done, _ = self.env.step(action)

        correct_reward = (step1_close - step0_close) * 34
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, False)
        self.assertEqual(self.env.profit_loss, [0.0, 0.0])

        correct_position = (34, 289.2786)
        self.assertEqual(self.env.positions[-1], correct_position)

        purchase_amount = correct_position[0] * correct_position[1]
        self.assertEqual(self.env.cash[-1], 10000.0 - purchase_amount)
        self.assertAlmostEqual(self.env.equity[-1], 10000.0 + correct_reward)

    def test_simple_short_trade_of_100_percent(self):
        _ = self._reset_env()
        step0_close = 289.2786
        step1_close = 289.04

        # Go short 100%
        action = np.array([-1.0])
        _, reward, done, _ = self.env.step(action)

        correct_reward = (step1_close - step0_close) * -34
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, False)
        self.assertEqual(self.env.profit_loss[-1], 0.0)

        correct_position = (-34, 289.2786)
        self.assertEqual(self.env.positions[-1], correct_position)

        purchase_amount = abs(correct_position[0] * correct_position[1])
        self.assertEqual(self.env.cash[-1], 10000.0 - purchase_amount)
        self.assertAlmostEqual(self.env.equity[-1], 10000.0 + correct_reward)

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

    def test_reduce_long_position(self):
        _ = self._reset_env()
        step0_close = 289.2786
        step1_close = 289.04
        step2_close = 288.45

        # Go long 50%
        action1 = np.array([0.5])
        self.env.step(action1)

        # Decrease long position by 35%
        action2 = np.array([0.15])
        _, reward, done, _ = self.env.step(action2)

        correct_reward = (step2_close - step1_close) * 5
        correct_profit_loss = (step1_close - step0_close) * 12
        self.assertAlmostEqual(reward, correct_reward)
        self.assertEqual(done, False)
        self.assertListEqual(
            self.env.profit_loss, [0.0, 0.0, correct_profit_loss])

        cash = 10000.0 - 4917.7362 + 3468.48
        curr_stock_value = 5 * step1_close      # TODO determine if step2 or step1
        self.assertEqual(self.env.cash[-1], cash)
        self.assertAlmostEqual(
            self.env.equity[-1], cash + curr_stock_value + correct_reward)

        correct_position = (5, step0_close)
        self.assertEqual(self.env.positions[-1][0], correct_position[0])
        self.assertAlmostEqual(
            self.env.positions[-1][1], correct_position[1])

        action3 = np.array([0.15])
        _, reward, done, _ = self.env.step(action3)

        self.assertListEqual(
            self.env.profit_loss, [0.0, 0.0, correct_profit_loss, 0.0])


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

    def _yield_market_data(self, asset_data_json):
        while True:
            asset_data = pd.read_json(asset_data_json)
            yield asset_data, self.previous_close

    def _normalize_dataframe(self, asset_data_json):
        normalized_dataframe = pd.read_json(asset_data_json)

        normalized_dataframe['open'] =\
            normalized_dataframe['open'] / (2 * self.previous_close)

        normalized_dataframe['high'] =\
            normalized_dataframe['high'] / (2 * self.previous_close)

        normalized_dataframe['low'] =\
            normalized_dataframe['low'] / (2 * self.previous_close)

        normalized_dataframe['close'] =\
            normalized_dataframe['close'] / (2 * self.previous_close)

        normalized_dataframe['volume'] =\
            normalized_dataframe['volume'] / self.daily_avg_volume
        
        return normalized_dataframe

    def _get_correct_observation(self, current_step=0, obs_size=1):
        offset = current_step+1 - obs_size

        if offset < 0:
            # Less data than observation_size
            if self.volume_enabled:
                observation_zeros = np.zeros([5, abs(offset)])
            else:
                observation_zeros = np.zeros([4, abs(offset)])
            offset = 0
        
        observation = np.array([
            self.normalized_asset_data.iloc[
                offset: current_step+1]['open'].values,
            self.normalized_asset_data.iloc[
                offset: current_step+1]['high'].values,
            self.normalized_asset_data.iloc[
                offset: current_step+1]['low'].values,
            self.normalized_asset_data.iloc[
                offset: current_step+1]['close'].values
        ])

        if self.volume_enabled:
            observation = np.vstack((
                observation,
                self.normalized_asset_data.iloc[
                    offset: current_step+1]['volume'].values
            ))

        if observation.shape[1] < obs_size:
            observation = np.concatenate(
                (observation, observation_zeros), axis=1)

        return observation

    # def _calculate_short_equity_value(self, shares, avg_price, curr_price):
    #     return abs(shares) * (avg_price - (curr_price - avg_price))

    def _reset_env(self):
        return self.env.reset()


# if __name__ == '__main__':
#     unittest.main()
