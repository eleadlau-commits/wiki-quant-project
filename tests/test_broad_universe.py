"""The point-in-time S&P 500 list parser, tested on a small copy of the 2016 table layout."""

from wikiquant import broad_universe as U

HTML = """
<div><table class="wikitable sortable">
<tr><th>Ticker symbol</th><th>Security</th><th>SEC filings</th><th>GICS Sector</th></tr>
<tr><td><a rel="nofollow" class="external text" href="https://www.nyse.com/quote/XNYS:MMM">MMM</a></td>
    <td><a href="/wiki/3M" title="3M">3M Company</a></td><td><a href="http://sec.gov">reports</a></td><td>Industrials</td></tr>
<tr><td><a class="external text" href="x">BRK.B</a></td>
    <td><a href="/wiki/Berkshire_Hathaway" title="Berkshire Hathaway">Berkshire Hathaway</a><sup><a href="#cite_note-1">[1]</a></sup></td>
    <td>reports</td><td>Financials</td></tr>
<tr><td><a class="external text" href="x">GOOGL</a></td><td><a href="/wiki/Alphabet_Inc.">Alphabet Inc Class A</a></td><td></td><td>Information Technology</td></tr>
<tr><td><a class="external text" href="x">GOOG</a></td><td><a href="/wiki/Alphabet_Inc.">Alphabet Inc Class C</a></td><td></td><td>Information Technology</td></tr>
<tr><td><a class="external text" href="x">FB</a></td><td><a href="/wiki/Facebook%2C_Inc.#History">Facebook</a></td><td></td><td>Information Technology</td></tr>
""" + "".join(
    f'<tr><td>X{i:02d}</td><td><a href="/wiki/Company_{i}">Company {i}</a></td><td></td><td>Energy</td></tr>'
    for i in range(60)
) + """
</table>
<table class="wikitable"><tr><th>Date</th><th>Added Ticker</th><th>Added Security</th></tr>
<tr><td>2015</td><td>ZZZ</td><td><a href="/wiki/Zzz">Zzz</a></td></tr></table></div>
"""


def test_parse_reads_first_constituents_table():
    rows = U.parse_constituents(HTML)
    assert len(rows) == 65
    assert rows[0] == {"ticker": "MMM", "name": "3M Company", "sector": "Industrials", "article": "3M"}
    # footnote text is dropped, anchors and %-escapes are cleaned from links
    assert rows[1]["name"] == "Berkshire Hathaway"
    assert rows[4]["article"] == "Facebook,_Inc."
    assert "ZZZ" not in {r["ticker"] for r in rows}


def test_share_classes_collapse_to_one_company():
    rows = U.dedupe_share_classes(U.parse_constituents(HTML))
    assert [r["ticker"] for r in rows[:4]] == ["MMM", "BRK.B", "GOOGL", "FB"]


def test_yahoo_tickers():
    assert U.yahoo_tickers("BRK.B") == ["BRK-B"]
    assert U.yahoo_tickers("FB") == ["META", "FB"]
    assert U.yahoo_tickers("AAPL") == ["AAPL"]


def test_member_keys_include_old_and_new_title():
    m = U.Member("FB", "META|FB", "Facebook", "IT", "Facebook,_Inc.", "Meta_Platforms")
    assert m.keys == ["Meta_Platforms", "Facebook,_Inc."]
    same = U.Member("MMM", "MMM", "3M", "Ind", "3M", "3M")
    assert same.keys == ["3M"]


def test_redirect_chains_are_followed(monkeypatch):
    hops = {"Netflix_Inc.": "Netflix,_Inc.", "Netflix,_Inc.": "Netflix"}
    monkeypatch.setattr(U, "_resolve_once", lambda ts: {t: hops.get(t, t) for t in ts})
    assert U.resolve_redirects(["Netflix_Inc.", "3M"]) == {"Netflix_Inc.": "Netflix", "3M": "3M"}
