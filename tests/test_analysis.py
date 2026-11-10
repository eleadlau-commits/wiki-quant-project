"""Tests for the source-agnostic analysis, especially that it never peeks ahead."""

import numpy as np
import pandas as pd

from wikiquant import analysis as A


def _prices(n=120, start="2024-01-01"):
    days = pd.bdate_range(start, periods=n)
    return days


def test_first_trade_after_is_strictly_the_next_trading_day():
    trading = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"])   # Fri, Mon, Tue
    cal = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"])
    pos = A.first_trade_after(cal, trading)
    # Friday's attention -> trade Monday; the weekend -> Monday; Monday -> Tuesday
    assert list(trading[pos]) == list(pd.DatetimeIndex(
        ["2024-01-05", "2024-01-08", "2024-01-08", "2024-01-08", "2024-01-09"]))


def test_spike_ratio_matches_the_original_wikipedia_measure():
    from wikiquant.pageviews import abnormal_attention

    views = pd.Series(np.random.default_rng(0).integers(50, 150, 200).astype(float),
                      index=pd.date_range("2024-01-01", periods=200))
    pd.testing.assert_series_equal(A.spike_ratio(views), abnormal_attention(views))


def test_spike_ratio_smoothing_steadies_sparse_counts():
    counts = pd.Series([0.0, 1.0] * 40 + [5.0], index=pd.date_range("2024-01-01", periods=81))
    raw = A.spike_ratio(counts)                   # median of 0/1 days is 0.5
    smooth = A.spike_ratio(counts, smooth=7)      # 7-day totals: usual 3 or 4, now 8
    assert raw.iloc[-1] == 10.0
    assert 2 < smooth.iloc[-1] < 3


def test_spike_ratio_is_nan_when_usual_level_is_zero():
    counts = pd.Series([0.0] * 70 + [3.0], index=pd.date_range("2024-01-01", periods=71))
    assert np.isnan(A.spike_ratio(counts).iloc[-1])


def test_event_study_measures_from_the_entry_close():
    trading = pd.bdate_range("2024-01-01", periods=80)
    stock = pd.Series(100.0, index=trading)
    stock.iloc[41:] = 110.0                         # +10% jump on the day after the spike's entry
    bench = pd.Series(100.0, index=trading)
    cal = pd.date_range("2023-10-01", trading[-1], freq="D")
    att = pd.DataFrame({"X": 100.0}, index=cal)
    spike_day = trading[39] + pd.Timedelta(days=0)   # attention on trading day 39
    att.loc[spike_day, "X"] = 1000.0
    ev = A.event_study(att, pd.DataFrame({"X": stock}), bench, threshold=3, hold=20)
    assert len(ev) == 1
    assert ev.loc[0, "entry_date"] == trading[40]   # first trading day after the spike
    assert abs(ev.loc[0, "excess"] - 0.10) < 1e-12  # held from day 40's close
    assert ev.loc[0, "reaction"] == 0.0


def test_cooldown_counts_a_burst_once():
    trading = pd.bdate_range("2024-01-01", periods=120)
    cal = pd.date_range("2023-10-01", trading[-1], freq="D")
    ratio = pd.Series(1.0, index=cal)
    ratio.loc[trading[50]:trading[53]] = 5.0
    ev = A.find_spikes(ratio, trading, threshold=3, cooldown=10)
    assert len(ev) == 1


def test_portfolio_decisions_ignore_the_future():
    rng = np.random.default_rng(1)
    trading = pd.bdate_range("2024-01-01", periods=150)
    tickers = list("ABCDEFGHIJ")
    prices = pd.DataFrame(100 * np.exp(rng.normal(0, 0.01, (150, 10)).cumsum(0)), index=trading, columns=tickers)
    cal = pd.date_range("2023-10-01", trading[-1], freq="D")
    att = pd.DataFrame(rng.lognormal(5, 0.5, (len(cal), 10)), index=cal, columns=tickers)

    def run(a):
        return A.buzz_portfolios(A.weekly_scores(A.spike_ratio(a), trading), prices)

    base = run(att)
    cut = trading[100]
    changed = att.copy()
    changed.loc[cut - pd.Timedelta(days=1):] *= rng.lognormal(0, 2, changed.loc[cut - pd.Timedelta(days=1):].shape)
    after = run(changed)
    # Scrambling attention from the day before `cut` must not change any return up to `cut`
    pd.testing.assert_frame_equal(base.loc[:cut], after.loc[:cut])


def test_buzz_portfolio_earns_from_the_next_day():
    trading = pd.bdate_range("2024-01-01", periods=30)    # starts on a Monday
    tickers = list("ABCDEFGH")
    prices = pd.DataFrame(100.0, index=trading, columns=tickers)
    prices.loc[trading[5]:, "A"] = 200.0                  # A doubles on the 2nd Monday itself
    scores = pd.DataFrame({t: [float(i)] for i, t in enumerate(reversed(tickers))}, index=[trading[5]])
    daily = A.buzz_portfolios(scores, prices, k=4)        # A has the top score
    assert daily.index[0] == trading[6]                   # picks made at day 5's close
    assert (daily["buzz"] == 0).all()                     # so the day-5 jump is not captured


def test_summarise():
    s = A.summarise([0.01, 0.03, -0.01, 0.05])
    assert s["n"] == 4 and abs(s["mean"] - 0.02) < 1e-12 and s["hit"] == 0.75
    assert abs(s["t"] - 0.02 / (np.std([0.01, 0.03, -0.01, 0.05], ddof=1) / 2)) < 1e-12
    assert A.summarise([])["n"] == 0
