from gym.envs.registration import register

register(
    id='StockTrading-v0',
    entry_point='gym_stock_trading.envs.stock_trading_env:StockTradingEnv'
)

register(
    id='AlpacaStockTrading-v0',
    entry_point='gym_stock_trading.envs.alpaca_stock_trading_env:AlpacaStockTradingEnv'
)

register(
    id='PaperAlpacaStockTrading-v0',
    entry_point='gym_stock_trading.envs.paper_alpaca_stock_trading_env:PaperAlpacaStockTradingEnv'
)

