from setuptools import setup

setup(
    name='gym_stock_trading',
    version='0.0.1',
    install_requires=[
        'gym',
        'pandas',
        'matplotlib',
        'mplfinance',
        'alpaca-trade-api'
    ]
)
