"""
This module is a stock trading environment for OpenAI gym
including matplotlib visualizations.
"""

import gym
import numpy as np
import pandas as pd

from gym import error, spaces, utils
from gym.utils import seeding

from chart import Chart

LOOKBACK_WINDOW_SIZE = 40


class StockTradingEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum
        starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: Box(4)
        Num	Observation               Min             Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                -24 deg         24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is
        pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the
        cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {'render.modes': ['human']}
    visualization = None

    def __init__(self, observation_size=1, allotted_amount=10000.0):
        super(StockTradingEnv, self).__init__()

        self.current_step = 0

        self.asset_data = None
        self.normalized_asset_data = None
        self.previous_close = None
        self.daily_avg_volume = None
        self.observation_size = observation_size

        self.base_value = allotted_amount
        self.equity = [allotted_amount]
        self.profit_loss = [0.0]
        self.cash = [allotted_amount]
        self.positions = [(0, 0.0)]     # (qty, price)
        self.rewards = [0.0]
        self.max_qty = 0

        # Each action represents the amount of the portfolio that should be
        # invested ranging from -1 to 1. Negative is short, positive is long.
        self.action_space = spaces.Box(
            low=np.array([-1]), high=np.array([1]), dtype=np.float16)

        # TODO needs to be a dynamic observation space
        # Normalized values for: Open, High, Low, Close, Volume
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, 1]),
            dtype=np.float16)

    # def _initialize_data(self):

    #     # make sure file is '.csv'
    #     path = '/Users/d/Documents/Projects/Python/gym-env/data/TSLA/'
    #     filename = ''
    #     while filename[-4:] != '.csv':
    #         # get random data file
    #         filename = random.choice(os.listdir(path))

    #     print("Chosen csv file: " + filename)

    #     # convert to data frame
    #     dataframe = pd.read_csv(path + filename)

    #     normalized_dataframe = dataframe.copy()

    #     with open(path + filename[:-4] + '-prev_close.txt') as f:
    #         content = f.read()
    #     prev_close = float(content)

    #     self.open_price = dataframe.iloc[0]['open']

    #     # normalize  data
    #     normalized_dataframe['open'] = normalized_dataframe['open'] / (2 * prev_close)
    #     normalized_dataframe['high'] = normalized_dataframe['high'] / (2 * prev_close)
    #     normalized_dataframe['low'] = normalized_dataframe['low'] / (2 * prev_close)
    #     normalized_dataframe['close'] = normalized_dataframe['close'] / (2 * prev_close)
    #     # Potential bug if volume in one minute is 1/10 of average daily volume
    #     normalized_dataframe['volume'] = normalized_dataframe['volume'] * 10 / TSLA_AVG_DAILY_VOLUME

    #     return (dataframe, normalized_dataframe)

    def _normalize_data(self, asset_data):
        normalized_dataframe = asset_data.copy()

        normalized_dataframe['open'] =\
            normalized_dataframe['open'] / (2 * self.previous_close)

        normalized_dataframe['high'] =\
            normalized_dataframe['high'] / (2 * self.previous_close)

        normalized_dataframe['low'] =\
            normalized_dataframe['low'] / (2 * self.previous_close)

        normalized_dataframe['close'] =\
            normalized_dataframe['close'] / (2 * self.previous_close)

        # Potential bug if volume in one minute is 1/10 of average daily volume
        normalized_dataframe['volume'] =\
            normalized_dataframe['volume'] * 10 / self.daily_avg_volume

        return normalized_dataframe

    def _next_observation(self):
        # Get the stock data for the current step
        obs = np.array([
            self.normalized_asset_data.loc[
                self.current_step: self.current_step + self.observation_size,
                'open'].values,
            self.normalized_asset_data.loc[
                self.current_step: self.current_step + self.observation_size,
                'high'].values,
            self.normalized_asset_data.loc[
                self.current_step: self.current_step + self.observation_size,
                'low'].values,
            self.normalized_asset_data.loc[
                self.current_step: self.current_step + self.observation_size,
                'close'].values,
            self.normalized_asset_data.loc[
                self.current_step: self.current_step + self.observation_size,
                'volume'].values,
        ])
        return obs

    def _take_action(self, action):
        curr_price = self.asset_data.iloc[self.current_step]['close']

        # Current position
        curr_qty, avg_price = self.positions[-1]
        curr_invested = curr_qty / self.max_qty

        if action == 0:
            # Close position
            trade_qty = -curr_qty
        else:
            target_change = action - curr_invested
            trade_qty = int(target_change * self.equity[-1] / curr_price)

        if curr_qty == 0:
            # Simple short or long trade
            purchase_amount = abs(trade_qty * curr_price)
            new_position = (trade_qty, curr_price)
            self.positions.append(new_position)
            self.cash.append(self.cash[-1] - purchase_amount)
            self.equity.append(self.cash[-1] + purchase_amount)
            self.profit_loss.append(0.0)

        elif curr_qty > 0 and curr_qty + trade_qty < 0 or\
                curr_qty < 0 and curr_qty + trade_qty > 0:
            # Trade crosses from short to long or long to short

            # Close current position, update P/L and cash
            if curr_qty > 0:
                # Closing long position
                self.cash.append(self.cash[-1] + curr_qty * curr_price)
                self.profit_loss.append((curr_price - avg_price) * curr_qty)
            else:
                # Closing short position
                self.cash.append(
                    self.cash[-1]
                    + abs(curr_qty)
                    * (avg_price - (curr_price - avg_price))
                )
                self.profit_loss.append(
                    (avg_price - curr_price) * abs(curr_qty))

            # Simple short or long trade
            trade_qty += curr_qty
            purchase_amount = abs(trade_qty * curr_price)
            new_position = (trade_qty, curr_price)
            self.positions.append(new_position)
            self.cash.append(self.cash[-1] - purchase_amount)
            self.equity.append(self.cash[-1] + purchase_amount)

        else:
            # Trade increases or reduces position (including closing out)

            if curr_qty > 0 and trade_qty > 0 or\
                    curr_qty < 0 and trade_qty < 0:
                # Adding to position

                purchase_amount = abs(trade_qty * curr_price)

                while self.cash[-1] < purchase_amount:
                    # Descrease trade_qty if not enough cash
                    trade_qty = trade_qty - 1 if trade_qty > 0\
                            else trade_qty + 1
                    purchase_amount = abs(trade_qty * curr_price)

                total_qty = trade_qty + curr_qty
                avg_price = (
                    ((trade_qty * curr_price) + (curr_qty * avg_price))
                    / total_qty
                )
                new_position = (total_qty, avg_price)
                self.positions.append(new_position)
                self.cash.append(self.cash[-1] - purchase_amount)

                if total_qty > 0:
                    # Long position
                    self.equity.append(
                        self.cash[-1] + (total_qty * curr_price))
                else:
                    # Short position
                    self.equity.append(
                        self.cash[-1]
                        + abs(total_qty)
                        * (avg_price - (curr_price - avg_price))
                    )

            # Reducing position or not changing
            else:
                if trade_qty > 0:
                    # Reducing short position
                    self.cash.append(self.cash[-1]
                                     + abs(trade_qty)
                                     * (avg_price - (curr_price - avg_price)))
                    self.profit_loss.append(
                        (avg_price - curr_price) * trade_qty)
                else:
                    # Reducing long position
                    self.cash.append(
                        self.cash[-1] + abs(trade_qty * curr_price))
                    self.profit_loss.append(
                        (curr_price - avg_price) * abs(trade_qty))

                net_qty = curr_qty + trade_qty

                if net_qty == 0:
                    new_position = (net_qty, 0.0)
                else:
                    new_position = (net_qty, avg_price)

                self.positions.append(new_position)

                if net_qty > 0:
                    # Long position
                    self.equity.append(
                        self.cash[-1] + abs(net_qty * curr_price))
                else:
                    # Short position
                    self.equity.append(
                        self.cash[-1]
                        + abs(net_qty)
                        * (avg_price - (curr_price - avg_price))
                    )

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        curr_price = self.asset_data.iloc[self.current_step]['close']

        self.current_step += 1

        next_price = self.asset_data.iloc[self.current_step]['close']

        reward = (next_price - curr_price) * self.positions[-1][0]
        self.equity[-1] += reward
        self.rewards.append(reward)

        # Episode ends when down 5% or DataFrame ends
        if self.current_step + 1 == len(self.asset_data):
            done = True
        else:
            done = True if self.equity[-1] / self.base_value <= -0.05\
                        else False

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self, asset_data, previous_close, daily_avg_volume):
        """Reset the state of the environment to an initial state"""
        self.asset_data = asset_data
        self.normalized_asset_data = self._normalize_data(asset_data)
        self.previous_close = previous_close
        self.daily_avg_volume = daily_avg_volume

        self.current_step = 0

        self.equity = [self.base_value]
        self.profit_loss = [0.0]
        self.cash = [self.base_value]
        self.positions = [(0, 0.0)]
        self.rewards = [0.0]
        self.max_qty = int((self.base_value
                            / self.asset_data.iloc[self.current_step]['open']))

        return self._next_observation()

    def render(self, mode='human', close=False):
        """Render the environment to the screen"""
        if self.visualization is None:
            self.visualization = Chart(self.asset_data)

        if self.current_step > LOOKBACK_WINDOW_SIZE:
            self.visualization.render(
                self.current_step,
                self.equity[-1],
                self.positions,
                window_size=LOOKBACK_WINDOW_SIZE)

    def close(self):
        pass
