# Gym Stock Trading
Gym Stock Trading is an OpenAI Gym environment for stock trading with integrations into Alpaca trade api for live and paper trading.

## Basics
For a better understanding of OpenAI Gym, check out the docs: https://github.com/openai/gym

### There are two environments included

**Gym Stock Trading Environment** uses 1min OHLCV (Open, High, Low, Close, Volume) aggregate bars as market data and provides unrealized profit/loss as a reward to the agent. Details of the operations of the environment can be found in the class docstring.

**Alpaca Stock Trading Environment** allows you to use the model trained in the Stock Trading Gym to test the live market in the form of paper or live trading. You must have an Alpaca Brokerage account (and API keys) to use this environment. Its free (although you have to fund the account) so don't let this hold you back. You can find Alpaca's website here: https://alpaca.markets

## Installation
```
git clone https://github.com/duhfrazee/gym-stock-trading
cd gym-stock-trading
pip install -e .
```

## Example


## Environment Details
