# algobot — a small, safety-first trading bot

A beginner-friendly algorithmic trading bot for a small account (built with
~$200 on Robinhood in mind). It follows the path every algo trader should take:

```
1. backtest  →  2. paper trade (fake money)  →  3. live, tiny budget
```

The same strategy code runs in all three stages, so what you test is what you trade.

> **Honest expectations.** With $200 the goal is *learning*, not income. Most simple
> strategies do **worse** than just buying and holding an index fund once you count
> trading costs, and a backtest that looks great is usually curve-fit. The bot always
> shows you buy-and-hold next to your strategy so you can judge for yourself.

---

## How Robinhood automation works (as of late 2026)

Robinhood has **no official public API for stocks** that custom code can use. Your options:

| Route | Official? | Notes |
|---|---|---|
| **`robin_stocks`** (what this bot uses for live trading) | ❌ Unofficial | Uses the same private API as the Robinhood app. Works and is widely used, but Robinhood can change it without notice or restrict accounts that automate against it. Keep the budget small. |
| **Robinhood Agentic Trading / Trading MCP** (launched May 2026) | ✅ Official | OAuth login, trades from a **separate account** you fund. It is built for *AI agents* (Claude, etc.) to make trades, not for running a fixed algorithm on a schedule. |
| **Robinhood Crypto Trading API** | ✅ Official | Real API keys, but crypto only. |
| A broker with an official stock API (for example Alpaca, which has free paper trading) | ✅ Official | Worth considering if you outgrow this. Only a new adapter in `algobot/brokers/` would be needed. |

The bot is **broker-agnostic**. Backtesting and paper trading never touch Robinhood at all.

---

## Quick start

Requires Python 3.11+.

```bash
cd algobot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp algobot.example.toml algobot.toml        # your settings (git-ignored)
```

### 1. Backtest

```bash
python -m algobot backtest SPY --all                    # every strategy vs buy-and-hold, 10 years
python -m algobot backtest QQQ --strategy sma --param fast=10 --param slow=40
python -m algobot backtest SPY --strategy rsi --period 5y
```

```
strategy                 final $   return    CAGR   max DD  sharpe  in mkt  trades
----------------------------------------------------------------------------------
buy_and_hold                 ...
sma_20_50                    ...
```

- **max DD** is the worst peak-to-trough drop. Ask yourself whether you could watch that happen without panicking.
- **in mkt** is how often the strategy holds shares.
- **trades** counts entries plus exits. Each one costs `--cost-bps` (default 5 bps = 0.05% for spread and slippage).

### 2. Paper trade (fake money, real prices)

```bash
python -m algobot run --dry-run      # show what it *would* do
python -m algobot run                # trade the paper account (state/paper_account.json)
python -m algobot status             # ledger, positions, equity
```

Run it on a schedule (below) for a few weeks. Compare what happened with what the backtest promised.

### 3. Go live (real money)

1. Set `broker = "robinhood"` in `algobot.toml`, and set `budget` to what you can afford to lose.
2. Install the extras with `pip install robin_stocks pyotp` (already in `requirements.txt`).
3. Provide your credentials as **environment variables**, never in files you commit:
   ```bash
   export RH_USERNAME="you@example.com"
   export RH_PASSWORD="..."
   export RH_TOTP_SECRET="..."   # optional: authenticator-app secret, so MFA codes are automatic
   ```
   The first login may ask you to approve it in the Robinhood app. After that the session is
   cached in `~/.tokens/robinhood.pickle`.
4. Check first, then trade:
   ```bash
   python -m algobot run --live --dry-run   # logs in and plans, but sends nothing
   python -m algobot run --live             # asks you to type YES before sending orders
   ```

Real money needs **both** `broker = "robinhood"` in the config **and** `--live` on the command line.
If either is missing, the bot refuses to trade.

---

## Running it every day

The strategies use **daily** bars, so run once per trading day, a few minutes before the 4:00 pm ET
close. Example crontab (`crontab -e`). Your computer has to be on at that time. Cron uses your
computer's time zone: 3:45 pm Eastern is 12:45 pm Pacific (`45 12 * * 1-5`).

```cron
45 15 * * 1-5  cd /path/to/TradeK/algobot && .venv/bin/python -m algobot run >> cron.log 2>&1
```

If it runs at the wrong time, the bot just reports "market is closed" and does nothing.

For unattended **live** trading, add `--live --yes` and put the `RH_*` variables in the cron
environment. Non-interactive live runs refuse to trade without `--yes`. On WSL, start cron with
`sudo service cron start`, or use Windows Task Scheduler instead.

---

## Safety features

| Feature | What it does |
|---|---|
| **Own ledger** | The bot tracks its own virtual cash (`budget`) and only the shares *it* bought. It will never sell shares you bought yourself, and it never spends more than `budget`. |
| **Kill switch** | `touch state/STOP` and the bot does nothing until you delete the file. |
| **Max order size** | `max_order_dollars` caps every buy. This protects you against a typo like `budget = 1500`. |
| **Daily order limit** | `max_orders_per_day`. Exits get priority over buys. |
| **Drawdown halt** | If bot equity falls `max_drawdown_pct` below its peak, buys stop until you run `python -m algobot reset-halt`. |
| **No same-day round trips** | `min_hold_days = 1` never sells shares bought the same day. This avoids day trades and cash-account settlement violations. |
| **Ambiguous orders halt** | If Robinhood doesn't answer while an order is being placed, the bot halts instead of risking a duplicate order. Check the app, then `reset-halt`. |
| **Bad data never trades** | Missing, short or stale price history (or a non-number signal) skips the symbol for that run. |
| **Market-hours check** | Live trading asks Robinhood for the real exchange calendar, including holidays. |
| **Pending orders** | A symbol with an unfilled order is skipped until the order resolves, so orders never pile up. |
| **Journal** | Every submit, fill and error is appended to `state/journal.csv`. |

---

## Rules worth knowing

- **Fractional shares:** Robinhood's minimum is $1 per order. The bot buys in dollars and sells in shares.
- **Pattern Day Trader rule:** the SEC approved removing the $25k PDT rule, effective June 4, 2026.
  Brokers have until October 2027 to implement it, so check your account's status. Either way the
  bot avoids same-day round trips.
- **Cash account settlement:** sale proceeds settle the next business day (T+1). The bot only buys
  with cash it already has, and on a daily schedule this avoids good-faith violations.
- **Taxes:** every sale is a taxable event, and anything held under a year is a short-term gain.
  Robinhood sends you the 1099. Watch out for wash sales if you also trade the same symbols by hand.

---

## Write your own strategy

A strategy is a small class: given closing prices, return a target weight for each day
(1.0 = fully invested, 0.0 = cash).

```python
# algobot/strategies.py
@dataclass(frozen=True)
class AboveLongAverage:
    """Hold while price is above its 200-day average."""
    window: int = 200

    @property
    def name(self): return f"above_sma_{self.window}"

    @property
    def warmup(self): return self.window

    def target_weights(self, close):
        return (close > sma(close, self.window)).astype(float)

STRATEGIES["above_sma"] = AboveLongAverage
```

Then backtest it with `python -m algobot backtest SPY --strategy above_sma`. To trade it, set
`name = "above_sma"` under `[strategy]` and add an `[strategy.above_sma]` section for its parameters.

Ideas to explore: a 200-day trend filter, dual momentum (SPY vs bonds), and volatility targeting.
To check for overfitting, tune the parameters on 2015–2020 data, then test them on 2021 onward.

---

## Project layout

```
algobot/
  strategies.py   indicators.py      the trading logic (pure functions of price)
  backtest.py     metrics.py         historical simulation + statistics
  engine.py                          one live/paper run, start to finish
  planner.py      risk.py            target weights → orders → risk checks
  ledger.py       journal.py         the bot's own books + audit log
  brokers/paper.py                   simulated broker
  brokers/robinhood.py               robin_stocks adapter
  data.py  config.py  cli.py         prices, settings, command line
tests/                               168 tests, ~99% coverage
```

Development:

```bash
pip install -e ".[dev,robinhood]"
pytest --cov=algobot       # all tests, no network or Robinhood account needed
ruff check algobot tests
```

*Nothing here is financial advice. This is a learning project: start on paper, keep it small, and
expect to lose the money you put in.*
