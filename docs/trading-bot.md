# Running the strategy for real: the trading bot

The bot runs the 50/200 moving-average crossover (or any strategy in
`wikiquant.trader.STRATEGIES`) on a Trading 212 account, once each weekday.

Before you put money in, remember what the backtest said: over 2020-26 this
rule earned **less than buying and holding** (9.6% vs 14.8% a year), mainly
because it sold after fast crashes and bought back higher. Running it live is
a good way to learn and a strong CV point, but it is not expected to beat an
index fund. Start on the practice account, then use only money you're happy
to treat as tuition.

## What one run does

1. Downloads about two years of SPY closing prices from Yahoo Finance and asks
   the strategy what it wants: **invested** (50-day average above 200-day) or **cash**.
2. Asks Trading 212 for the account value and how much of the fund you hold.
3. Works out the order needed to hold 95% of the account in the fund (or 0%).
   The 5% buffer leaves room for price moves and fees.
4. Skips orders under 20 (noise), waits if an earlier order is still pending,
   and refuses any order above a safety cap (2,000 by default).
5. Places a market order, unless it's a dry run.

Most days the answer is "hold": the backtest made only six trades in seven years.

**Which fund?** UK retail accounts can't buy SPY itself (US funds don't have
the UK key-information document). So the bot reads SPY's trend and trades
**VUAG**, the Vanguard S&P 500 UCITS ETF (accumulating, priced in pounds on
the London Stock Exchange). It follows the same index, but your returns also
move with the pound-dollar exchange rate. You can pick a different fund (see
step 3 below). Choose one priced in your account's currency.

**Timing.** The bot runs at 08:15 UTC on weekdays, after London opens. It
uses the previous US close, so trades happen about one day after the signal.
The backtest tested exactly this "trade one day later" case: 9.1% a year
instead of 9.6%.

## Safety switches

Each step towards real money needs its own switch, so you can't get there by accident:

| You want | What you need |
| --- | --- |
| See what it would do | nothing: the default is a dry run on a pretend account |
| Pretend trading | `--execute` (pretend account in `trades/paper_account.json`) |
| Trading 212 practice account | `--broker trading212` + practice API key |
| Practice orders actually placed | add `--execute` |
| Real money | live API key + `--live` + `BOT_CONFIRM_LIVE=yes` |

Also: the market-order endpoint can execute twice if a request is repeated, so
the bot never retries an order, and GitHub never runs two copies at once.

## Setup

### 1. On your computer (pretend money, no keys)

```bash
pip install -r requirements.txt && pip install -e .
python scripts/run_bot.py              # dry run
python scripts/run_bot.py --execute    # pretend trade; run it again tomorrow
```

### 2. A Trading 212 practice key

In the app, switch to **Practice**, then **Settings > API (Beta)** > generate a
key with permissions *account, portfolio, orders:read, orders:execute*. Copy
the key and the secret (the secret is shown only once). Keep them out of chat,
code and commits.

```bash
export T212_API_KEY="..."  T212_API_SECRET="..."      # Windows: set T212_API_KEY=...
python scripts/run_bot.py --broker trading212 --find "S&P 500"
```

### 3. Check the fund's code

`--find` lists matching instruments. Look for the Vanguard S&P 500
accumulating line in GBP. The bot assumes its code is `VUAGl_EQ`. If the list
shows something different, pass `--trade-ticker <code>` (and set
`BOT_TRADE_TICKER` on GitHub). Then:

```bash
python scripts/run_bot.py --broker trading212             # dry run against practice
python scripts/run_bot.py --broker trading212 --execute   # places a practice order
```

Check the order in the app. If the market is closed, it waits until 8:00 the next morning.

### 4. Run it automatically on GitHub

In the repository: **Settings > Secrets and variables > Actions**.

- **Secrets:** `T212_API_KEY`, `T212_API_SECRET` (the practice ones for now).
- **Variables:** `BOT_ENABLED` = `true`. Optional: `BOT_TRADE_TICKER`,
  `BOT_MAX_ORDER`.

Then open **Actions > Trading bot > Run workflow** and leave "execute" unticked
for a dry run. Once it's merged to `main`, it runs every weekday morning and
places orders. Set `BOT_ENABLED` to anything else to stop it.

Privacy: this repository is public, so its Actions logs are too. The bot
hides money amounts in those logs. Your trades are visible in the app.

### 5. Going live

**Start small with a budget.** The bot normally sizes trades from the whole
account value. Set the variable `BOT_BUDGET` (e.g. `50`) and it acts as if the
account were worth only that, leaving the rest alone. Also keep the fund the
bot trades (VUAG) *only* for the bot: when the signal turns off it sells
**all** of it, including any you bought yourself.


1. Switch the app to your real Invest or Stocks ISA account, generate a
   **live** key, and replace the two secrets with it.
2. Set the variables `T212_ENV` = `live` and `BOT_CONFIRM_LIVE` = `yes`.
3. Set `BOT_BUDGET` to the amount the bot may use, and `BOT_MAX_ORDER` a bit
   above it, so a bug can't buy more.

To stop live trading, delete `T212_ENV` or set `BOT_ENABLED` to `false`. Any
position stays open until you sell it in the app.

**Tax note:** in a Stocks ISA, gains are tax-free. In an Invest account, each
sell is a disposal for capital gains tax (you get a yearly allowance). This
isn't advice, just something to know.

## Code

- `src/wikiquant/trader.py`: the plan (signal, target, order) and the logging
- `src/wikiquant/brokers/paper.py`: the pretend broker
- `src/wikiquant/brokers/trading212.py`: the Trading 212 API client
- `scripts/run_bot.py`: the command-line entry point
- `.github/workflows/trading-bot.yml`: the weekday schedule
- `tests/test_trader.py`: tests with a fake broker and a fake Trading 212 server
