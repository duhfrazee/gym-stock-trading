"""
This module is a stock trading environment for OpenAI gym
including matplotlib visualizations.
"""
import datetime
import os
import random
import threading
import time

import alpaca_trade_api as tradeapi
import gym
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from datetime import timedelta
from gym import error, spaces, utils
from gym.utils import seeding
from mplfinance.original_flavor import candlestick_ochl as candlestick
from pytz import timezone

LOOKBACK_WINDOW_SIZE = 40
VOLUME_CHART_HEIGHT = 0.33
UP_COLOR = '#27A59A'
DOWN_COLOR = '#EF534F'
UP_TEXT_COLOR = '#73D3CC'
DOWN_TEXT_COLOR = '#DC2C27'

try:
    PAPER_APCA_API_KEY_ID = os.environ['PAPER_APCA_API_KEY_ID']
    PAPER_APCA_API_SECRET_KEY = os.environ['PAPER_APCA_API_SECRET_KEY']
    PAPER_APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
    LIVE_APCA_API_KEY_ID = os.environ['LIVE_APCA_API_KEY_ID']
    LIVE_APCA_API_SECRET_KEY = os.environ['LIVE_APCA_API_SECRET_KEY']
except KeyError:
    pass


class Chart():
    """A stock chart visualization using matplotlib
    made to render StockTrading OpenAI Gym Environment
    """

    def __init__(self, asset_data):
        self.asset_data = asset_data
        self.account_values = np.zeros(len(self.asset_data))

        # Create a figure on screen
        self.fig = plt.figure()

        # Create top subplot for account value axis
        self.account_value_ax = plt.subplot2grid(
            (6, 1), (0, 0), rowspan=2, colspan=1)

        # Create bottom subplot for shared price/volume axis
        self.price_ax = plt.subplot2grid(
            (6, 1), (2, 0), rowspan=8, colspan=1, sharex=self.account_value_ax)

        # Create a new axis for volume which shares its x-axis with price
        self.volume_ax = self.price_ax.twinx()

        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.11, bottom=0.24,
                            right=0.90, top=0.90, wspace=0.2, hspace=0)

        # Show the graph without blocking the rest of the program
        plt.show(block=False)

    def _render_account_value(
            self, current_step, account_value, step_range, dates):

        # Clear the frame rendered last step
        self.account_value_ax.clear()

        # Plot account values
        self.account_value_ax.plot_date(
            dates, self.account_values[step_range], '-', label='Account Value')

        # Show legend, which uses the label we defined for the plot above
        self.account_value_ax.legend()
        legend = self.account_value_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_date = self._date2num(
            self.asset_data['timestamp'].values[current_step])
        last_account_value = self.account_values[current_step]

        # Annotate the current account value on the account value graph
        self.account_value_ax.annotate(
            '{0:.2f}'.format(account_value),
            (last_date, last_account_value),
            xytext=(last_date, last_account_value),
            bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
            color="black",
            fontsize="small"
        )

        # Add space above and below min/max account value
        self.account_value_ax.set_ylim(
            min(self.account_values[np.nonzero(self.account_values)])
            / 1.25, max(self.account_values) * 1.25
        )

    def _render_price(self, current_step, account_value, dates, step_range):
        self.price_ax.clear()

        # Format data for OHCL candlestick graph
        candlesticks = zip(
            dates,
            self.asset_data['open'].values[step_range],
            self.asset_data['close'].values[step_range],
            self.asset_data['high'].values[step_range],
            self.asset_data['low'].values[step_range]
        )

        # Plot price using candlestick graph from mpl_finance
        candlestick(self.price_ax, candlesticks, width=0.5/(24*60),
                    colorup=UP_COLOR, colordown=DOWN_COLOR)

        last_date = self._date2num(
            self.asset_data['timestamp'].values[current_step])
        last_close = self.asset_data['close'].values[current_step]

        # Print the close price to the price axis
        self.price_ax.annotate(
            '{0:.2f}'.format(last_close),
            (last_date, last_close),
            xytext=(last_date, last_close),
            bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
            color="black",
            fontsize="small"
        )

        # Shift price axis up to give volume chart space
        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0])
                               * VOLUME_CHART_HEIGHT, ylim[1])

    def _render_volume(self, current_step, account_value, dates, step_range):
        self.volume_ax.clear()

        volume = np.array(self.asset_data['volume'].values[step_range])

        pos = self.asset_data['open'].values[step_range] - \
            self.asset_data['close'].values[step_range] < 0
        neg = self.asset_data['open'].values[step_range] - \
            self.asset_data['close'].values[step_range] > 0

        # Color volume bars based on price direction on that date
        self.volume_ax.bar(dates[pos], volume[pos], color=UP_COLOR,
                           alpha=0.4, width=0.5/(24*60), align='center')
        self.volume_ax.bar(dates[neg], volume[neg], color=DOWN_COLOR,
                           alpha=0.4, width=0.5/(24*60), align='center')

        # Cap volume axis height below price chart and hide ticks
        self.volume_ax.set_ylim(0, max(volume) / VOLUME_CHART_HEIGHT)
        self.volume_ax.yaxis.set_ticks([])

    def _render_trades(self, current_step, positions, step_range):

        for step, position in enumerate(positions):
            if step in step_range:
                date = self._date2num(
                    self.asset_data['timestamp'].values[step-1])
                high = self.asset_data['high'].values[step]
                low = self.asset_data['low'].values[step]

                curr_qty = position[0]
                prev_qty = positions[step-1][0]

                if curr_qty > prev_qty:
                    # bought
                    high_low = low
                    color = UP_TEXT_COLOR
                elif curr_qty < prev_qty:
                    # sold
                    high_low = high
                    color = DOWN_TEXT_COLOR
                else:
                    continue

                trade_qty = '{0}'.format(curr_qty - prev_qty)

                # Print the trade quantity to the price axis
                self.price_ax.annotate(f'{trade_qty}', (date, high_low),
                                       xytext=(date, high_low),
                                       color=color,
                                       fontsize=8,
                                       arrowprops=(dict(color=color)))

    def _date2num(self, date):
        return mdates.datestr2num(date)

    def render(self, current_step, account_value, positions, window_size=40):
        """render() utilizes matplotlib to visualize the observations and
           actions.

        Arguments:
            current_step {int} -- current_step of environment
            account_value {float} -- total equity at current_step
            positions {list} -- list of all positions over current episode

        Keyword Arguments:
            window_size {int} -- maximum number of candlesticks to show
            (default: {40})
        """
        self.account_values[current_step] = account_value

        window_start = max(current_step - window_size, 0)
        step_range = range(window_start, current_step + 1)

        self.fig.suptitle(positions[current_step])

        # Format dates as timestamps, necessary for candlestick graph
        dates = np.array(
            [self._date2num(x)
                for x in self.asset_data['timestamp'].values[step_range]])

        self._render_account_value(
            current_step, account_value, step_range, dates)
        self._render_price(current_step, account_value, dates, step_range)
        self._render_volume(current_step, account_value, dates, step_range)
        self._render_trades(current_step, positions, step_range)

        # Format the date ticks to be more easily read
        self.price_ax.set_xticklabels(
            self.asset_data['timestamp'].values[step_range],
            rotation=45,
            horizontalalignment='right'
        )

        # Hide duplicate net worth date labels
        plt.setp(self.account_value_ax.get_xticklabels(), visible=False)

        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

    def close(self):
        plt.close()


class StockTradingEnv(gym.Env):
    """
    Description:
        A simulated stock trading environment with the ability go long and
        short. The data is the normalized OHLC and volume values for
        1 minute candlesticks.

    Observation:
        Type: Box(low=0, high=1, shape=(5, observation_size), dtype=np.float16)
        Description: The observation_size is set during environment creation
            and represents the number of 1min candlesticks in the observation.
        Num	Observation               Min             Max
        0	Open                      -1              1
        1	High                      -1              1
        2	Low                       -1              1
        3	Close                     -1              1
        4   Volume                    -1              1

    Actions:
        Type: spaces.Box(
            low=np.array([-1]), high=np.array([1]), dtype=np.float16)
        Description: The action space represents the percentage of the account
            to invest. Negative is short, Positive is long.
        Num	Action                              Min         Max
        0	Percentage of account to invest     -1          1

    Reward:
        Reward is the unrealized profit/loss for the next observation.
    Starting State:
        First candlestick observation in dataset.
    Episode Termination:
        Agent loses more than 5% or day ends.
    """

    metadata = {'render.modes': ['human']}
    visualization = None
    eastern = timezone('US/Eastern')

    def __init__(self, mode='backtest', symbol=None, filepath=None,
                 observation_size=1, volume_enabled=True,
                 random_data_selection=True, allotted_amount=10000.0):
        super(StockTradingEnv, self).__init__()

        self.current_step = 0
        self.current_episode = 0

        self.mode = mode

        if self.mode == 'backtest':
            self.path = filepath
            self.filename = ''
            self.current_filename = ''
        else:
            self.symbol = symbol
            self.market = None

            self.paper = tradeapi.REST(
                PAPER_APCA_API_KEY_ID,
                PAPER_APCA_API_SECRET_KEY,
                PAPER_APCA_API_BASE_URL,
                api_version='v2'
            )

            self.live = tradeapi.REST(
                LIVE_APCA_API_KEY_ID,
                LIVE_APCA_API_SECRET_KEY,
                api_version='v2'
            )

            self.live_conn = tradeapi.StreamConn(
                LIVE_APCA_API_KEY_ID,
                LIVE_APCA_API_SECRET_KEY
            )

        self.volume_enabled = volume_enabled
        self.random_data_selection = random_data_selection
        self.asset_data = None
        self.normalized_asset_data = None
        self.previous_close = None
        self.daily_avg_volume = None
        self.observation_size = observation_size

        self.base_value = allotted_amount
        self.equity = [allotted_amount]
        self.cash = [allotted_amount]
        self.profit_loss = [0.0]
        self.positions = [(0, 0.0)]     # (qty, price)
        self.rewards = [0.0]
        self.max_qty = None

        # Each action represents the amount of the portfolio that should be
        # invested ranging from -1 to 1. Negative is short, positive is long.
        self.action_space = spaces.Box(
            low=np.array([-1]), high=np.array([1]), dtype=np.float16)

        # Normalized values for: Open, High, Low, Close, Volume
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, observation_size), dtype=np.float16)

    def _normalize_data(self):
        # TODO fix normalization with previous close
        normalized_dataframe = self.asset_data.copy()

        # highest_price = max(self.asset_data['high'])
        # highest_volume = max(self.asset_data['volume'])

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

    def _initialize_backtest_data(self, filename):
        # TODO add prev_close and avg daily volume
        files = os.listdir(self.path)
        files = [fi for fi in files if fi.endswith(".csv")]

        if self.random_data_selection and filename == '':
            filename = random.choice(files)

        elif not self.random_data_selection and filename == '':
            if self.current_episode >= len(files):
                self.current_episode = 0

            filename = sorted(files)[self.current_episode]

        if filename[-4:] != '.csv':
            raise TypeError('File must be .csv')

        self.current_episode += 1

        # Convert to data frame
        asset_data = pd.read_csv(self.path + filename)
        asset_data['timestamp'] = pd.to_datetime(asset_data['timestamp'])

        files = os.listdir(self.path + 'day/')
        files = [fi for fi in files if fi.endswith(".csv")]

        daily_data = pd.read_csv(self.path + 'day/' + files[0])
        daily_data['timestamp'] = pd.to_datetime(daily_data['timestamp'])

        date = asset_data.iloc[0]['timestamp']

        self.previous_close = daily_data[
            daily_data['timestamp'] < date]['close'].iloc[-2]
        # TODO should include most recent day but data was incomplete
        self.daily_avg_volume = int(daily_data['volume'].iloc[-31:-1].mean())

        self.filename = filename
        self.asset_data = asset_data
        self.normalized_asset_data = self._normalize_data()

    def _initialize_live_data(self):
        self.market = self.live.get_clock()

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

        self.asset_data = self.live.polygon.historic_agg_v2(
            self.symbol, 1, 'minute', _open, close).df

        self.current_step = len(self.asset_data)

        # TODO needs to be tested
        if not self.market.is_open\
                and self.market.next_close.date() != today.date():
            previous_close_date = today.date()
        else:
            # Get previous market day from alpaca calendar
            _from = today.date() - timedelta(days=7)
            to = today.date() - timedelta(days=1)

            cal = self.live.get_calendar(_from, to)

            previous_close_date = cal[-1].date.date()

        previous_close_date_plus_one = previous_close_date + timedelta(days=1)

        self.previous_close = self.live.polygon.historic_agg_v2(
            self.symbol,
            1,
            'day',
            previous_close_date,
            previous_close_date_plus_one
        ).df['close'][0]

        today_minus_30 = today.date() - timedelta(days=30)
        self.daily_avg_volume = int(self.live.polygon.historic_agg_v2(
            'MSFT', 1, 'day', today_minus_30, today.date()).df['volume'].mean()
        )

    def _initialize_data(self, filename):
        """Initializes environment data from files in path"""

        if self.mode == 'backtest':
            self._initialize_backtest_data(filename)
        else:
            self._initialize_live_data()

    def _await_market_open(self):
        while not self.market.is_open:
            print('Waiting for market to open...')
            curr_time = datetime.datetime.now(self.eastern)
            next_open = self.market.next_open
            time.sleep((next_open-curr_time).seconds)
            self.market = self.live.get_clock()

    async def _on_minute_bars(self, conn, channel, bar):
        if bar.symbol == self.symbol:
            new_row = {
                'timestamp': datetime.datetime.fromtimestamp(
                    bar.start,
                    self.eastern
                ),
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            }
            self.asset_data = self.asset_data.append(
                new_row,
                ignore_index=True
            )
            self._normalize_data()

    def _next_observation(self):
        """Get the stock data for the current observation size."""

        if self.mode != 'backtest':
            tAMO = threading.Thread(target=self._await_market_open)
            tAMO.start()
            tAMO.join()

            # at 9:30AM EDT, subscribe to polygon websocket
            _on_minute_bars =\
                self.live_conn.on(r'AM$')(self._on_minute_bars)
            tWS = threading.Thread(
                target=self.live_conn.run, args=[['AM.' + self.symbol]])
            tWS.start()

            while len(self.normalized_asset_data) == self.current_step:
                # wait for new data to be appended
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
            self.normalized_asset_data.loc[
                offset: self.current_step]['open'].values,
            self.normalized_asset_data.loc[
                offset: self.current_step]['high'].values,
            self.normalized_asset_data.loc[
                offset: self.current_step]['low'].values,
            self.normalized_asset_data.loc[
                offset: self.current_step]['close'].values
        ])

        if self.volume_enabled:
            observation = np.vstack((
                observation,
                self.normalized_asset_data.loc[
                    offset: self.current_step]['volume'].values
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
            if self.mode == 'live':
                self.live.submit_order(
                    symbol=self.symbol,
                    qty=abs(qty),
                    side=side,
                    type=order_type,
                    time_in_force='day'
                )
            elif self.mode == 'paper':
                self.paper.submit_order(
                    symbol=self.symbol,
                    qty=abs(qty),
                    side=side,
                    type=order_type,
                    time_in_force='day'
                )
        # TODO log difference between close price and filled price
        except Exception as e:
            # TODO return error
            # check for:
            # 403 Forbidden: Buying power or shares is not sufficient.
            # 422 Unprocessable: Input parameters are not recognized.
            return e

    def _close_position(self):
        try:
            if self.mode == 'live':
                position = self.live.get_position(symbol=self.symbol)
            elif self.mode == 'paper':
                position = self.live.get_position(symbol=self.symbol)

            if position.side == 'long':
                trade_qty = -int(position.qty)
            else:
                trade_qty = int(position.qty)

            tOrder = threading.Thread(
                target=self._submit_order, args=[trade_qty])
            tOrder.start()
            tOrder.join()

        except Exception as e:
            if str(e) != 'position does not exist':
                # TODO return error
                return e

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

            if self.mode != 'backtest':
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

                if self.mode != 'backtest':
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

                if self.mode != 'backtest':
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

            if self.mode != 'backtest':
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

                if self.mode != 'backtest':
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

                if self.mode != 'backtest':
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

        if self.mode == 'backtest':
            # Episode ends when down 5% or DataFrame ends
            if self.current_step + 1 == len(self.asset_data):
                done = True
            else:
                done = True if self.equity[-1] / self.base_value <= -0.05\
                            else False
        else:
            # Close 11 minutes before end of day
            now = datetime.datetime.now(self.eastern)

            stop_time = self.eastern.localize(
                datetime.datetime.combine(
                    now,
                    datetime.time(3, 49)
                )
            )

            if now > stop_time:
                tOrder = threading.Thread(
                    target=self._close_position)
                tOrder.start()
                done = True
            # TODO this needs to be more reflective of real data in future
            elif self.equity[-1] / self.base_value <= -0.05:
                tOrder = threading.Thread(
                    target=self._close_position)
                tOrder.start()
                done = True
            else:
                done = False

        return obs, reward, done, {}

    def reset(self, filename=''):
        """Reset the state of the environment to an initial state"""
        self.current_step = 0

        try:
            self._initialize_data(filename)
        except TypeError:
            return TypeError('File must be .csv')

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

        self.visualization.render(
            self.current_step,
            self.equity[-1],
            self.positions,
            window_size=LOOKBACK_WINDOW_SIZE)

    def close(self):
        if self.mode != 'backtest':
            # Ensure no positions are held over night
            tOrder = threading.Thread(
                target=self._close_position)
            tOrder.start()
            tOrder.join()

            self.live.cancel_all_orders()
