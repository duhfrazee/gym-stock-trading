# Gym Stock Trading
Gym Stock Trading is an OpenAI Gym environment for stock trading with integrations into Alpaca trade api for live and paper trading.

## Basics
For a better understanding of OpenAI Gym, check out the docs: https://github.com/openai/gym

### There are two environments included:

**Gym Stock Trading Environment** (intended for historical data backtesting) uses 1min OHLCV (Open, High, Low, Close, Volume) aggregate bars as market data and provides unrealized profit/loss as a reward to the agent. Details of the operations of the environment can be found in the class docstring.

**Alpaca Stock Trading Environment** allows you to use the model trained in the Stock Trading Gym to test the live market in the form of paper or live trading. You must have an Alpaca Brokerage account (and API keys) to use this environment. Its free (although you have to fund the account) so don't let this hold you back. You can find Alpaca's website here: https://alpaca.markets

## Installation
```
git clone https://github.com/duhfrazee/gym-stock-trading
cd gym-stock-trading
pip install -e .
```

## Example


### Environment initialization
**Gym Stock Trading Environment**
- market_data:
  - generator that yields a tuple of (asset_data, previous_close)
    - asset_data: dataframe with 1 full day of 1 min aggregate bars
    - previous_close: previous day closing price
- daily_avg_volume=None:
  - integer used to normalize volume (I use 30-day daily average volume for the asset)
- observation_size=1:
  - number of aggregate bars returned in the observation
    - Note: if the current step is 0 (first observation) and the observation size is 5, all excess observations will be 0
- volume_enabled=True:
  - denotes wether to include volume as part of observation
- trade_penalty=0.0
  - amount to penalize agent per trade
- allotted_amount=10000.0
  - amount of money the agent has to trade

**Alpaca Stock Trading Environment**
- symbol:
  - str: symbol to trade, ex. 'AAPL'
- previous_close:
  - float: Previous day close for asset 
- daily_avg_volume=None:
  - integer used to normalize volume (I use 30-day daily average volume for the asset)
- live=False:
  - Whether to trade paper or live assets
- observation_size=1:
  - number of aggregate bars returned in the observation
    - Note: if the current step is 0 (first observation) and the observation size is 5, all excess observations will be 0 
- volume_enabled=True
  - denotes wether to include volume as part of observation
- allotted_amount=10000.0)
  - amount of money the agent has to trade

## Environment Details
