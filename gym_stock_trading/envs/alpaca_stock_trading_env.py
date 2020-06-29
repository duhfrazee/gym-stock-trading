"""
This module is a stock trading environment for OpenAI gym
utilizing Alpaca for live and paper trading.
"""
import datetime
import logging
import os
import random
import threading
import time

import alpaca_trade_api as tradeapi
import gym
import numpy as np
import pandas as pd

from datetime import timedelta
from gym import error, spaces, utils
from gym.utils import seeding
from pytz import timezone

logger = logging.getLogger(__name__)
try:
    PAPER_APCA_API_KEY_ID = os.environ['PAPER_APCA_API_KEY_ID']
    PAPER_APCA_API_SECRET_KEY = os.environ['PAPER_APCA_API_SECRET_KEY']
    PAPER_APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
    LIVE_APCA_API_KEY_ID = os.environ['LIVE_APCA_API_KEY_ID']
    LIVE_APCA_API_SECRET_KEY = os.environ['LIVE_APCA_API_SECRET_KEY']
except KeyError:
    logger.error("KeyError for Alpaca credentials.")


class AlpacaStockTradingEnv(gym.Env):
    """
    Description:
        A live stock trading environment utilizing Alpaca Trade API.
        The environment utilizes websockets for trade updates and market data.
        Alpaca API keys are required for this environment to operate.
        The data is the normalized OHLC and volume values for
        1 minute candlesticks.

    Observation:
        Type: Box(low=0, high=1, shape=(5, observation_size), dtype=np.float16)
        Description: The observation_size is set during environment creation
            and represents the number of 1min candlesticks in the observation.
        Num	Observation               Min             Max
        0	Open                      0               1
        1	High                      0               1
        2	Low                       0               1
        3	Close                     0               1
        4   Volume                    0               1

    Actions:
        Type: spaces.Box(
            low=np.array([-1]), high=np.array([1]), dtype=np.float16)
        Description: The action space represents the percentage of the account
            to invest. Negative is short, Positive is long.
        Num	Action                              Min         Max
        0	Percentage of account to invest     -1          1

    Reward:
        Reward is the unrealized profit/loss for the next observation.
        Realized Profit/Loss is also tracked in the info dict.
    Starting State:
        First candlestick observation once the market opens
    Episode Termination:
        Agent loses more than 5% or 11 minutes before market close.
    """

    metadata = {'render.modes': ['human']}
    eastern = timezone('US/Eastern')
    channels = ['AM.*', 'trade_updates']

    live_conn = tradeapi.StreamConn(
        LIVE_APCA_API_KEY_ID,
        LIVE_APCA_API_SECRET_KEY
    )
    paper_conn = tradeapi.StreamConn(
        PAPER_APCA_API_KEY_ID,
        PAPER_APCA_API_SECRET_KEY,
        PAPER_APCA_API_BASE_URL
    )

    try:
        # tLWS = threading.Thread(
        #     target=live_conn.run, args=[channels])
        # tLWS.start()
        logger.info('Connecting to channels: %s', channels)
        tPWS = threading.Thread(
            target=paper_conn.run, args=[channels])
        tPWS.start()
    except RuntimeError as e:
        # Already running
        logger.error(e)

    def __init__(self, symbol, previous_close, daily_avg_volume=None,
                 live=False, observation_size=1, volume_enabled=True,
                 allotted_amount=10000.0):
        super(AlpacaStockTradingEnv, self).__init__()

        self.current_step = 0
        self.current_episode = 0

        self.live = live

        self.symbol = symbol
        self.market = None

        self.paper_api = tradeapi.REST(
            PAPER_APCA_API_KEY_ID,
            PAPER_APCA_API_SECRET_KEY,
            PAPER_APCA_API_BASE_URL,
            api_version='v2'
        )

        self.live_api = tradeapi.REST(
            LIVE_APCA_API_KEY_ID,
            LIVE_APCA_API_SECRET_KEY,
            api_version='v2'
        )

        if self.live:
            self._on_minute_bars =\
                self.live_conn.on(r'^AM$')(self._on_minute_bars)
            self._on_trade_updates =\
                self.live_conn.on(r'trade_updates$')(self._on_trade_updates)
        else:
            self._on_minute_bars =\
                self.paper_conn.on(r'^AM$')(self._on_minute_bars)
            self._on_trade_updates =\
                self.paper_conn.on(r'trade_updates$')(self._on_trade_updates)

        self.volume_enabled = volume_enabled
        self.asset_data = None
        self.normalized_asset_data = None
        self.previous_close = previous_close

        if self.volume_enabled:
            self.daily_avg_volume = daily_avg_volume
        self.observation_size = observation_size

        self.base_value = allotted_amount
        self.equity = [allotted_amount]
        self.cash = [allotted_amount]
        self.profit_loss = [0.0]
        self.positions = [(0, 0.0)]     # (qty, price)
        self.alpaca_positions = [(0, 0.0)]  # actual traded positions
        self.current_alpaca_position = (0, 0.0)
        self.trades = []
        self.rewards = [0.0]
        self.max_qty = None

        # Each action represents the amount of the portfolio that should be
        # invested ranging from -1 to 1. Negative is short, positive is long.
        self.action_space = spaces.Box(
            low=np.array([-1]), high=np.array([1]), dtype=np.float16)

        # Normalized values for: Open, High, Low, Close, Volume
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, observation_size), dtype=np.float16)

        logger.info('Initialized AlpacaStockTradingEnv: ')
        logger.info('Live: %s', self.live)
        logger.info('Symbol: %s', self.symbol)
        logger.info('volume_enabled: %s', self.volume_enabled)
        logger.info('previous_close: %s', self.previous_close)
        if self.volume_enabled:
            logger.info('daily_avg_volume: %s', self.daily_avg_volume)
        logger.info('observation_size: %s', self.observation_size)
        logger.info('base_value: %s', self.base_value)
        logger.info('equity: %s', self.equity)
        logger.info('cash: %s', self.cash)
        logger.info('profit_loss: %s', self.profit_loss)

    def _normalize_data(self):

        normalized_dataframe = self.asset_data.copy()

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

    def _initialize_data(self):
        self.market = self.live_api.get_clock()

        today = datetime.datetime.now(self.eastern)

        if self.market.is_open:
            _open = int(self.eastern.localize(
                datetime.datetime.combine(
                    today,
                    datetime.time(9, 30)
                )
            ).timestamp() * 1000)
        else:
            _open = int(self.market.next_open.timestamp() * 1000)

        close = int(self.market.next_close.timestamp() * 1000)

        self.asset_data = self.live_api.polygon.historic_agg_v2(
            self.symbol, 1, 'minute', _open, close).df

        self.current_step = len(self.asset_data)

        self.normalized_asset_data = self._normalize_data()

    def _await_market_open(self):
        while not self.market.is_open:
            curr_time = datetime.datetime.now(self.eastern)
            next_open = self.market.next_open.astimezone(self.eastern)
            wait_time = (next_open-curr_time).seconds + 10

            print('Waiting ' + str(wait_time) + ' seconds for market to open.')

            time.sleep(wait_time)
            self.market = self.live_api.get_clock()

    async def _on_minute_bars(self, conn, channel, bar):
        if self.normalized_asset_data is not None:
            if self.market.is_open:
                if bar.symbol == self.symbol:

                    # if len(self.asset_data) != 0:
                    #     # TODO test
                    #     # Missing a bar or market is frozen
                    #     if bar.start >\
                    #             self.asset_data.index[-1]\
                    #             + timedelta(seconds=90):
                    #         self._initialize_data()

                    new_row = {
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume
                    }
                    self.asset_data.loc[bar.start] = new_row
                    self.normalized_asset_data = self._normalize_data()

    async def _on_trade_updates(self, conn, channel, account):
        event = account.event
        order = account.order
        # logger.info('async _on_trade_updates called for %s', order)
        if order['symbol'] == self.symbol:
            if event == 'fill' or event == 'partial_fill':
                if order['side'] == 'buy':
                    fill_qty = int(account.qty)
                else:
                    fill_qty = -int(account.qty)

                fill_price = float(account.price)
                position_qty = int(account.position_qty)
                curr_qty = self.current_alpaca_position[0]
                curr_avg_price = self.current_alpaca_position[1]

                if position_qty != 0:
                    if curr_qty >= 0 and fill_qty > 0\
                            or curr_qty <= 0 and fill_qty < 0:
                        avg_price = (
                            (abs(fill_qty) * fill_price)
                            + (abs(curr_qty) * curr_avg_price)
                        ) / abs(position_qty)
                    else:
                        avg_price = curr_avg_price
                else:
                    avg_price = 0.0

                self.current_alpaca_position = (position_qty, avg_price)

    def _next_observation(self):
        """Get the stock data for the current observation size."""

        if not self.market.is_open:
            tAMO = threading.Thread(target=self._await_market_open)
            tAMO.start()
            tAMO.join()

        # TODO clean up with wait() or threading
        while len(self.normalized_asset_data) <= self.current_step:
            # Wait for new data to be appended
            continue

        offset = self.current_step+1 - self.observation_size

        if offset < 0:
            # Less data than observation_size
            if self.volume_enabled:
                observation_zeros = np.zeros([5, abs(offset)])
            else:
                observation_zeros = np.zeros([4, abs(offset)])
            offset = 0

        observation = np.array([
            self.normalized_asset_data.iloc[
                offset: self.current_step+1]['open'].values,
            self.normalized_asset_data.iloc[
                offset: self.current_step+1]['high'].values,
            self.normalized_asset_data.iloc[
                offset: self.current_step+1]['low'].values,
            self.normalized_asset_data.iloc[
                offset: self.current_step+1]['close'].values
        ])

        if self.volume_enabled:
            observation = np.vstack((
                observation,
                self.normalized_asset_data.iloc[
                    offset: self.current_step+1]['volume'].values
            ))

        if observation.shape[1] < self.observation_size:
            observation = np.concatenate(
                (observation, observation_zeros), axis=1)

        return observation

    def _submit_order(self, qty, order_type='market'):
        if qty == 0:
            # no trade needed
            return

        side = 'buy' if qty > 0 else 'sell'

        try:
            if self.live:
                self.live_api.submit_order(
                    symbol=self.symbol,
                    qty=abs(qty),
                    side=side,
                    type=order_type,
                    time_in_force='day'
                )
            else:
                self.paper_api.submit_order(
                    symbol=self.symbol,
                    qty=abs(qty),
                    side=side,
                    type=order_type,
                    time_in_force='day'
                )
        except Exception as e:
            print(e)
            # TODO return error
            # check for:
            # 403 Forbidden: Buying power or shares is not sufficient.
            # 422 Unprocessable: Input parameters are not recognized.
            return e

    def _close_position(self):
        if self.current_alpaca_position[0] != 0:
            if self.live:
                self.live_api.close_position(self.symbol)
            else:
                self.paper_api.close_position(self.symbol)

            # TODO clean up with wait() or threading
            while self.current_alpaca_position[0] != 0:
                # wait for position to close
                continue

    def _take_action(self, action):
        curr_price = self.asset_data.iloc[self.current_step]['close']

        if self.positions[-1][0] != self.current_alpaca_position[0]:
            print('Order did not complete..')
            # TODO determine how to cancel incomplete orders
            pass

        # Current position
        curr_qty, avg_price = self.current_alpaca_position
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

            # Submit order
            tOrder = threading.Thread(
                target=self._submit_order, args=[trade_qty])
            tOrder.start()

        elif curr_qty > 0 and curr_qty + trade_qty < 0 or\
                curr_qty < 0 and curr_qty + trade_qty > 0:
            # Trade crosses from short to long or long to short

            # Close current position, update P/L and cash
            if curr_qty > 0:
                # Closing long position
                self.cash.append(self.cash[-1] + curr_qty * curr_price)
                self.profit_loss.append((curr_price - avg_price) * curr_qty)

                tOrder = threading.Thread(
                    target=self._close_position)
                tOrder.start()
                tOrder.join()
            else:
                # Closing short position
                self.cash.append(
                    self.cash[-1]
                    + abs(curr_qty)
                    * (avg_price - (curr_price - avg_price))
                )
                self.profit_loss.append(
                    (avg_price - curr_price) * abs(curr_qty))

                tOrder = threading.Thread(
                    target=self._close_position)
                tOrder.start()
                tOrder.join()

            # Simple short or long trade
            trade_qty += curr_qty
            purchase_amount = abs(trade_qty * curr_price)
            new_position = (trade_qty, curr_price)
            self.positions.append(new_position)
            self.cash.append(self.cash[-1] - purchase_amount)
            self.equity.append(self.cash[-1] + purchase_amount)

            # Submit order
            tOrder = threading.Thread(
                target=self._submit_order, args=[trade_qty])
            tOrder.start()
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
                self.profit_loss.append(0.0)

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

                # Submit order
                tOrder = threading.Thread(
                    target=self._submit_order, args=[trade_qty])
                tOrder.start()

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

                # Submit order
                tOrder = threading.Thread(
                    target=self._submit_order, args=[trade_qty])
                tOrder.start()

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        curr_price = self.asset_data.iloc[self.current_step]['close']

        self.current_step += 1

        obs = self._next_observation()

        next_price = self.asset_data.iloc[self.current_step]['close']

        reward = (next_price - curr_price) * self.positions[-1][0]
        self.equity[-1] += reward
        self.rewards.append(reward)
        self.alpaca_positions.append(self.current_alpaca_position)

        info = {
            "profit_loss": {
                "date_time": self.asset_data.index[-2],
                "profit_loss": self.profit_loss[-1],
                "reward": reward,
                "mode": 'live' if self.live else 'paper'
            },
            "trades": {},
            "positions": {
                "env_position": self.positions[-1],
                "actual_position": self.current_alpaca_position
            }
        }

        # Close 11 minutes before end of day
        now = datetime.datetime.now(self.eastern)

        stop_time = self.eastern.localize(
            datetime.datetime.combine(
                now,
                datetime.time(15, 49)
            )
        )

        if now > stop_time:
            tOrder = threading.Thread(
                target=self._close_position)
            tOrder.start()
            done = True
        # TODO this needs to be more reflective of real data in future
        elif (self.equity[-1] - self.base_value) / self.base_value <= -0.05:
            tOrder = threading.Thread(
                target=self._close_position)
            tOrder.start()
            done = True
        else:
            done = False

        return obs, reward, done, info

    def reset(self):
        """Reset the state of the environment to an initial state"""
        self.current_step = 0

        self._initialize_data()

        self.equity = [self.base_value]
        self.profit_loss = [0.0]
        self.cash = [self.base_value]
        self.positions = [(0, 0.0)]
        self.rewards = [0.0]
        self.trades = []

        observation = self._next_observation()

        self.max_qty = int((self.base_value
                            / self.asset_data.iloc[self.current_step]['open']))

        return observation

    def render(self, mode='human', close=False):
        """Render the environment to the screen"""
        pass

    def close(self):
        # Ensure no positions are held over night
        tOrder = threading.Thread(
            target=self._close_position)
        tOrder.start()
        tOrder.join()

        # TODO figure out how to cancel open orders
