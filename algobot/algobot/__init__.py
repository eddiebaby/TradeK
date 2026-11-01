"""algobot — a small, safety-first algorithmic trading bot.

Workflow: backtest a strategy -> paper trade it -> (optionally) go live
with a tiny budget on Robinhood. Every layer is broker-agnostic so the
same strategy code runs in all three modes.
"""

__version__ = "0.1.0"
