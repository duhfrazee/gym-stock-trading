import numpy as np
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ochl as candlestick

VOLUME_CHART_HEIGHT = 0.33
UP_COLOR = '#27A59A'
DOWN_COLOR = '#EF534F'
UP_TEXT_COLOR = '#73D3CC'
DOWN_TEXT_COLOR = '#DC2C27'

class Chart():
    """A stock chart visualization using matplotlib
    made to render TradingEnv OpenAI Gym Environment
    """

    def __init__(self, asset_data, title=None):
        self.asset_data = asset_data
        self.account_values = np.zeros(len(self.asset_data))

        # Create a figure on screen and set the title
        self.fig = plt.figure()
        self.fig.suptitle(title)

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

    def _render_account_value(self, current_step, account_value, step_range, dates):
        # Clear the frame rendered last step
        self.account_value_ax.clear()

        # Plot net worths
        self.account_value_ax.plot_date(
            dates, self.account_values[step_range], '-', label='Net Worth')

        # Show legend, which uses the label we defined for the plot above
        self.account_value_ax.legend()
        legend = self.account_value_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_date = self._date2num(self.asset_data['timestamp'].values[current_step])
        last_net_worth = self.account_values[current_step]

        # Annotate the current net worth on the net worth graph
        self.account_value_ax.annotate('{0:.2f}'.format(account_value), (last_date, last_net_worth),
                                   xytext=(last_date, last_net_worth),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")

        # Add space above and below min/max net worth
        self.account_value_ax.set_ylim(
            min(self.account_values[np.nonzero(self.account_values)]) / 1.25, max(self.account_values) * 1.25)

    def _render_price(self, current_step, account_value, dates, step_range):
        self.price_ax.clear()

        # Format data for OHCL candlestick graph
        candlesticks = zip(dates,
                           self.asset_data['open'].values[step_range], self.asset_data['close'].values[step_range],
                           self.asset_data['high'].values[step_range], self.asset_data['low'].values[step_range])

        # Plot price using candlestick graph from mpl_finance
        candlestick(self.price_ax, candlesticks, width= 0.5/(24*60),
                    colorup=UP_COLOR, colordown=DOWN_COLOR)

        last_date = self._date2num(self.asset_data['timestamp'].values[current_step])
        last_close = self.asset_data['close'].values[current_step]
        last_high = self.asset_data['high'].values[current_step]

        # Print the current price to the price axis
        self.price_ax.annotate('{0:.2f}'.format(last_close), (last_date, last_close),
                               xytext=(last_date, last_high),
                               bbox=dict(boxstyle='round',
                                         fc='w', ec='k', lw=1),
                               color="black",
                               fontsize="small")

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

        positions = [(i, position[0], position[1]) for i,position in enumerate(positions)]
        for i,position in enumerate(positions):
            step = position[0]
            if step in step_range:
                date = self._date2num(self.asset_data['timestamp'].values[step])
                high = self.asset_data['high'].values[step]
                low = self.asset_data['low'].values[step]

                curr_qty = position[1]
                prev_qty = positions[i-1][1]

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
        self.account_values[current_step] = account_value

        window_start = max(current_step - window_size, 0)
        step_range = range(window_start, current_step + 1)

        self.fig.suptitle(positions[current_step])

        # Format dates as timestamps, necessary for candlestick graph
        dates = np.array([self._date2num(x)
                          for x in self.asset_data['timestamp'].values[step_range]])

        self._render_account_value(current_step, account_value, step_range, dates)
        self._render_price(current_step, account_value, dates, step_range)
        self._render_volume(current_step, account_value, dates, step_range)
        self._render_trades(current_step, positions, step_range)

        # Format the date ticks to be more easily read
        self.price_ax.set_xticklabels(self.asset_data['timestamp'].values[step_range], rotation=45,
                                      horizontalalignment='right')

        # Hide duplicate net worth date labels
        plt.setp(self.account_value_ax.get_xticklabels(), visible=False)

        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

    def close(self):
        plt.close()
