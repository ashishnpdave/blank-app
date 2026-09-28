"""NSE Sector Rotation tool (Relative Rotation Graph style) built with Streamlit.

Shows which NSE sectors are gaining or losing relative strength versus a
benchmark, and lets you drill down to the stocks inside each sector.

Run:  streamlit run app.py

Educational tool only. Not investment advice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# =============================================================================
# CONFIG  (edit freely)
# =============================================================================
BENCHMARKS: Dict[str, str] = {
    "Nifty 500": "^CRSLDX",
    "Nifty 50": "^NSEI",
}

# Sector name -> Yahoo Finance index ticker. Each one is verified at runtime;
# failures are skipped with a warning (Yahoo occasionally renames/drops indices).
SECTOR_INDICES: Dict[str, str] = {
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^CNXIT",
    "Nifty Auto": "^CNXAUTO",
    "Nifty FMCG": "^CNXFMCG",
    "Nifty Pharma": "^CNXPHARMA",
    "Nifty Metal": "^CNXMETAL",
    "Nifty Realty": "^CNXREALTY",
    "Nifty Energy": "^CNXENERGY",
    "Nifty Financial Services": "NIFTY_FIN_SERVICE.NS",
    "Nifty Media": "^CNXMEDIA",
    "Nifty PSU Bank": "^CNXPSUBANK",
    "Nifty Infra": "^CNXINFRA",
}

# Sector name (must match SECTOR_INDICES keys) -> top constituents.
# Renamed/delisted tickers just produce a warning. Add or remove as you like.
CONSTITUENTS: Dict[str, List[str]] = {
    "Nifty Bank": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
                   "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "FEDERALBNK.NS",
                   "IDFCFIRSTB.NS", "AUBANK.NS", "BANDHANBNK.NS"],
    "Nifty IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS",
                 "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "OFSS.NS", "LTTS.NS"],
    # Tata Motors tickers changed after its demerger; add the current ones if you want them.
    "Nifty Auto": ["MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
                   "TVSMOTOR.NS", "ASHOKLEY.NS", "BOSCHLTD.NS", "MOTHERSON.NS",
                   "BHARATFORG.NS", "BALKRISIND.NS"],
    "Nifty FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS",
                   "DABUR.NS", "GODREJCP.NS", "MARICO.NS", "COLPAL.NS", "VBL.NS", "UBL.NS"],
    "Nifty Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS",
                     "AUROPHARMA.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS", "ALKEM.NS",
                     "MANKIND.NS", "BIOCON.NS", "GLENMARK.NS"],
    "Nifty Metal": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "JINDALSTEL.NS",
                    "SAIL.NS", "NMDC.NS", "APLAPOLLO.NS", "NATIONALUM.NS", "HINDCOPPER.NS"],
    "Nifty Realty": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "PHOENIXLTD.NS",
                     "LODHA.NS", "BRIGADE.NS", "SOBHA.NS", "ANANTRAJ.NS"],
    "Nifty Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "BPCL.NS", "IOC.NS",
                     "COALINDIA.NS", "GAIL.NS", "TATAPOWER.NS", "ADANIGREEN.NS",
                     "HINDPETRO.NS", "PETRONET.NS"],
    "Nifty Financial Services": ["HDFCBANK.NS", "ICICIBANK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
                                 "SBILIFE.NS", "HDFCLIFE.NS", "SHRIRAMFIN.NS", "CHOLAFIN.NS",
                                 "MUTHOOTFIN.NS", "SBICARD.NS", "PFC.NS", "RECLTD.NS"],
    "Nifty Media": ["SUNTV.NS", "ZEEL.NS", "PVRINOX.NS", "SAREGAMA.NS", "NAZARA.NS",
                    "NETWORK18.NS", "TV18BRDCST.NS", "DBCORP.NS", "HATHWAY.NS", "TIPSMUSIC.NS"],
    "Nifty PSU Bank": ["SBIN.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS",
                       "INDIANB.NS", "BANKINDIA.NS", "MAHABANK.NS", "IOB.NS", "UCOBANK.NS",
                       "CENTRALBK.NS", "PSB.NS"],
    "Nifty Infra": ["LT.NS", "ADANIPORTS.NS", "ULTRACEMCO.NS", "BHARTIARTL.NS", "NTPC.NS",
                    "POWERGRID.NS", "INDIGO.NS", "CONCOR.NS", "IRB.NS", "NCC.NS", "KEC.NS",
                    "GRASIM.NS"],
}

# --- Data & calculation constants -------------------------------------------
CACHE_TTL_SECONDS = 3600                                  # 1-hour cache
RETURN_WINDOWS: Dict[str, int] = {"1W": 5, "1M": 21, "3M": 63, "6M": 126}  # trading days
MIN_HISTORY_BARS = 60          # tickers with fewer daily bars are treated as failed
FFILL_LIMIT_BARS = 5           # bridge holidays / gaps of up to this many bars
WEEK_RULE = "W-FRI"            # weekly bars end on Friday
CENTER = 100.0                 # RRG centre line for both axes
TIMEFRAMES = ("Weekly", "Daily")

# --- UI defaults -------------------------------------------------------------
DEFAULT_RATIO_WINDOW = 14
DEFAULT_ROC_PERIOD = 4
DEFAULT_TAIL = 8
DEFAULT_LOOKBACK_YEARS = 3
LOOKBACK_YEARS_RANGE = (1, 5)
TAIL_RANGE = (2, 20)
TOP_N = 5                      # gainers / laggards shown in drill-down
DIRECTION_BARS = 3             # bars used to compute rotation direction

# --- Chart constants ---------------------------------------------------------
MIN_AXIS_HALF_RANGE = 1.5
AXIS_PADDING = 1.2
TAIL_MIN_OPACITY = 0.15
TAIL_MARKER_SIZES = (4, 9)
HEAD_MARKER_SIZE = 14
TAIL_LINE_ALPHA = 0.35
RRG_HEIGHT = 680

QUADRANT_COLORS: Dict[str, str] = {
    "Leading": "#2ea043",
    "Weakening": "#d4a72c",
    "Lagging": "#da3633",
    "Improving": "#388bfd",
}
# (x sign, y sign) of each quadrant relative to the (100, 100) centre.
QUADRANT_SIGNS: Dict[str, Tuple[int, int]] = {
    "Leading": (1, 1), "Weakening": (1, -1), "Lagging": (-1, -1), "Improving": (-1, 1),
}
QUADRANT_BG_ALPHA = 0.12

# Arrow for each 45-degree sector, starting at 0 deg (east) going counter-clockwise.
ARROW_SYMBOLS = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
GREEN, AMBER, RED = "#2ea043", "#d4a72c", "#da3633"
ARROW_COLORS = dict(zip(ARROW_SYMBOLS, [GREEN, GREEN, GREEN, AMBER, RED, RED, RED, AMBER]))

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
           "#17becf", "#bcbd22", "#7f7f7f", "#393b79", "#ad494a", "#637939", "#a55194", "#e6550d"]

DISCLAIMER = ("**Disclaimer:** This is an educational tool, not investment advice. Data comes "
              "from Yahoo Finance via yfinance and may be delayed, adjusted or incomplete. "
              "Do your own research and consult a SEBI-registered adviser before investing.")


@dataclass(frozen=True)
class Settings:
    """User-selected sidebar settings."""
    benchmark: str
    timeframe: str
    lookback_years: int
    ratio_window: int
    roc_period: int
    tail: int


# =============================================================================
# DATA LAYER
# =============================================================================
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def download_prices(tickers: Tuple[str, ...], years: int) -> pd.DataFrame:
    """Download daily adjusted closes for ``tickers`` (cached for one hour).

    Returns a DataFrame indexed by date with one column per ticker that returned at
    least ``MIN_HISTORY_BARS`` observations. Tickers with no usable data are dropped.
    """
    raw = yf.download(list(tickers), period=f"{years}y", interval="1d",
                      auto_adjust=True, progress=False, group_by="column", threads=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close = close.sort_index()
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    close = close.dropna(how="all")
    return close.loc[:, close.notna().sum() >= MIN_HISTORY_BARS]


def load_universe(name_to_ticker: Dict[str, str], years: int) -> Tuple[pd.DataFrame, List[str]]:
    """Download prices and relabel columns by display name.

    Returns ``(prices, failed_names)`` where ``failed_names`` are entries that returned
    no usable data.
    """
    tickers = tuple(sorted(set(name_to_ticker.values())))
    try:
        prices = download_prices(tickers, years)
    except Exception as exc:  # network errors, yfinance changes, etc.
        st.warning(f"Download failed: {exc}")
        return pd.DataFrame(), list(name_to_ticker)
    ticker_to_name = {t: n for n, t in name_to_ticker.items()}
    prices = prices.rename(columns=ticker_to_name)
    failed = [n for n in name_to_ticker if n not in prices.columns]
    return prices, failed


def align_to_benchmark(prices: pd.DataFrame, bench: pd.Series,
                       timeframe: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Align ``prices`` to the benchmark calendar, bridge holidays, optionally go weekly.

    Missing values are forward-filled up to ``FFILL_LIMIT_BARS``. For weekly data the last
    close of each week is used (the current, incomplete week uses the latest close).
    """
    bench = bench.dropna()
    aligned = prices.reindex(bench.index).ffill(limit=FFILL_LIMIT_BARS)
    if timeframe == "Weekly":
        aligned = aligned.resample(WEEK_RULE).last()
        bench = bench.resample(WEEK_RULE).last()
        bench = bench.dropna()
        aligned = aligned.reindex(bench.index)
    return aligned, bench


# =============================================================================
# CALCULATIONS
# =============================================================================
def rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Z-score of each column against its own rolling mean and standard deviation."""
    mean = df.rolling(window, min_periods=window).mean()
    std = df.rolling(window, min_periods=window).std(ddof=0)
    return ((df - mean) / std).replace([np.inf, -np.inf], np.nan)


def compute_rs_ratio(prices: pd.DataFrame, bench: pd.Series, window: int) -> pd.DataFrame:
    """RS-Ratio = 100 + rolling z-score of relative strength (price / benchmark)."""
    rs = prices.div(bench, axis=0)
    return CENTER + rolling_zscore(rs, window)


def compute_rs_momentum(ratio: pd.DataFrame, roc_period: int, window: int) -> pd.DataFrame:
    """RS-Momentum = 100 + ROC of RS-Ratio divided by its rolling standard deviation.

    Not mean-adjusted, so momentum > 100 always means RS-Ratio is rising.
    """
    roc = ratio.pct_change(roc_period, fill_method=None) * 100
    std = roc.rolling(window, min_periods=window).std(ddof=0)
    return (CENTER + roc / std).replace([np.inf, -np.inf], np.nan)


def compute_rrg(prices: pd.DataFrame, bench: pd.Series, window: int,
                roc_period: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(rs_ratio, rs_momentum)`` DataFrames aligned to ``prices``."""
    ratio = compute_rs_ratio(prices, bench, window)
    return ratio, compute_rs_momentum(ratio, roc_period, window)


def classify_quadrant(ratio: pd.Series, mom: pd.Series) -> pd.Series:
    """Label each name Leading / Weakening / Lagging / Improving (N/A if data missing)."""
    conditions = [
        (ratio > CENTER) & (mom > CENTER),
        (ratio > CENTER) & (mom <= CENTER),
        (ratio <= CENTER) & (mom <= CENTER),
        (ratio <= CENTER) & (mom > CENTER),
    ]
    valid = ratio.notna() & mom.notna()
    labels = np.select(conditions, list(QUADRANT_COLORS), default="N/A")
    return pd.Series(np.where(valid, labels, "N/A"), index=ratio.index)


def compute_returns(daily: pd.DataFrame, windows: Dict[str, int]) -> pd.DataFrame:
    """Percentage returns over each trading-day window (NaN if history is too short)."""
    filled = daily.ffill()
    last = filled.iloc[-1]
    nan_series = pd.Series(np.nan, index=filled.columns)
    out = {label: (last / filled.iloc[-1 - n] - 1 if len(filled) > n else nan_series)
           for label, n in windows.items()}
    return pd.DataFrame(out) * 100


def rs_change(ratio: pd.DataFrame, bars: int) -> pd.Series:
    """Change in RS-Ratio over the last ``bars`` observations."""
    if len(ratio) <= bars:
        return pd.Series(np.nan, index=ratio.columns)
    return ratio.iloc[-1] - ratio.iloc[-1 - bars]


def rotation_direction(ratio: pd.DataFrame, mom: pd.DataFrame, bars: int) -> pd.Series:
    """Arrow showing the direction the point moved on the RRG over the last ``bars``."""
    d_x = rs_change(ratio, bars).to_numpy(dtype=float)
    d_y = rs_change(mom, bars).to_numpy(dtype=float)
    angle = np.degrees(np.arctan2(d_y, d_x))
    sector = np.round(angle / 45) % len(ARROW_SYMBOLS)
    arrows = np.array(ARROW_SYMBOLS, dtype=object)
    idx = np.nan_to_num(sector).astype(int)
    return pd.Series(np.where(np.isnan(sector), "–", arrows[idx]), index=ratio.columns)


def build_ranking_table(daily: pd.DataFrame, ratio: pd.DataFrame,
                        mom: pd.DataFrame) -> pd.DataFrame:
    """Combine returns, composite rank, RS-Ratio/Momentum, quadrant and direction."""
    table = compute_returns(daily[ratio.columns], RETURN_WINDOWS)
    table.insert(0, "Rank", table.rank(ascending=False).mean(axis=1)
                 .rank(method="min").astype("Int64"))
    table["RS-Ratio"] = ratio.iloc[-1]
    table["RS-Momentum"] = mom.iloc[-1]
    table["Quadrant"] = classify_quadrant(table["RS-Ratio"], table["RS-Momentum"])
    table["Rotation"] = rotation_direction(ratio, mom, DIRECTION_BARS)
    return table.sort_values("Rank")


# =============================================================================
# CHARTS & STYLING
# =============================================================================
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' to a CSS rgba() string."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def build_rrg_figure(ratio: pd.DataFrame, mom: pd.DataFrame, tail: int, title: str) -> go.Figure:
    """Relative Rotation Graph with coloured quadrants and fading tails.

    Each name is drawn as a path of its last ``tail`` + 1 points: older points are smaller
    and more transparent; the newest point is large and labelled.
    """
    ratio_t, mom_t = ratio.iloc[-(tail + 1):], mom.iloc[-(tail + 1):]
    names = [c for c in ratio_t.columns if (ratio_t[c].notna() & mom_t[c].notna()).any()]
    fig = go.Figure()
    if not names:
        return fig

    half_x = max(MIN_AXIS_HALF_RANGE, (ratio_t[names] - CENTER).abs().max().max() * AXIS_PADDING)
    half_y = max(MIN_AXIS_HALF_RANGE, (mom_t[names] - CENTER).abs().max().max() * AXIS_PADDING)

    for quadrant, (sx, sy) in QUADRANT_SIGNS.items():
        color = QUADRANT_COLORS[quadrant]
        fig.add_shape(type="rect", layer="below", line_width=0,
                      x0=min(CENTER, CENTER + sx * half_x), x1=max(CENTER, CENTER + sx * half_x),
                      y0=min(CENTER, CENTER + sy * half_y), y1=max(CENTER, CENTER + sy * half_y),
                      fillcolor=hex_to_rgba(color, QUADRANT_BG_ALPHA))
        fig.add_annotation(x=CENTER + sx * half_x, y=CENTER + sy * half_y, text=f"<b>{quadrant}</b>",
                           showarrow=False, font=dict(color=color, size=16),
                           xanchor="right" if sx > 0 else "left",
                           yanchor="top" if sy > 0 else "bottom")

    for i, name in enumerate(names):
        pts = pd.concat([ratio_t[name], mom_t[name]], axis=1, keys=["x", "y"]).dropna()
        n, color = len(pts), PALETTE[i % len(PALETTE)]
        dates = pts.index.strftime("%d %b %Y")
        hover = ("<b>" + name + "</b><br>%{customdata}<br>RS-Ratio %{x:.2f}"
                 "<br>RS-Momentum %{y:.2f}<extra></extra>")
        fig.add_trace(go.Scatter(
            x=pts["x"], y=pts["y"], mode="lines+markers", customdata=dates, hovertemplate=hover,
            line=dict(color=hex_to_rgba(color, TAIL_LINE_ALPHA), width=1.5),
            marker=dict(color=color, size=np.linspace(*TAIL_MARKER_SIZES, n),
                        opacity=np.linspace(TAIL_MIN_OPACITY, 1, n)),
            showlegend=False))
        fig.add_trace(go.Scatter(
            x=pts["x"].iloc[-1:], y=pts["y"].iloc[-1:], mode="markers+text", text=[name],
            textposition="top center", customdata=dates[-1:], hovertemplate=hover,
            marker=dict(color=color, size=HEAD_MARKER_SIZE, line=dict(color="white", width=1.5)),
            showlegend=False))

    fig.update_layout(
        title=title, height=RRG_HEIGHT, margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(title="RS-Ratio (relative trend)", range=[CENTER - half_x, CENTER + half_x],
                   zeroline=False, showgrid=False),
        yaxis=dict(title="RS-Momentum (speed of change)", range=[CENTER - half_y, CENTER + half_y],
                   zeroline=False, showgrid=False))
    fig.add_vline(x=CENTER, line_width=1, line_color="gray")
    fig.add_hline(y=CENTER, line_width=1, line_color="gray")
    return fig


def build_heatmap(returns: pd.DataFrame, title: str) -> go.Figure:
    """Diverging heatmap (rows = names, columns = return horizons) centred on zero."""
    data = returns.sort_values(list(RETURN_WINDOWS)[-2], ascending=False)  # sort by 3M
    fig = go.Figure(go.Heatmap(
        z=data.to_numpy(), x=list(data.columns), y=list(data.index), colorscale="RdYlGn", zmid=0,
        texttemplate="%{z:+.1f}%", colorbar=dict(title="%"),
        hovertemplate="%{y}<br>%{x}: %{z:+.2f}%<extra></extra>"))
    fig.update_layout(title=title, height=max(350, 42 * len(data) + 120),
                      yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _color_return(v: float) -> str:
    return "" if pd.isna(v) else f"color: {GREEN if v >= 0 else RED}"


def _color_quadrant(q: str) -> str:
    c = QUADRANT_COLORS.get(q)
    return "" if c is None else f"background-color: {hex_to_rgba(c, 0.25)}; font-weight: 600"


def _color_arrow(a: str) -> str:
    return f"color: {ARROW_COLORS.get(a, 'gray')}; font-weight: 700; font-size: 1.3em"


def style_table(table: pd.DataFrame):
    """Apply number formats and colour coding to a ranking table."""
    fmt = {**{c: "{:+.1f}%" for c in RETURN_WINDOWS}, "RS-Ratio": "{:.2f}", "RS-Momentum": "{:.2f}"}
    return (table.style.format(fmt, na_rep="–")
            .map(_color_return, subset=list(RETURN_WINDOWS))
            .map(_color_quadrant, subset=["Quadrant"])
            .map(_color_arrow, subset=["Rotation"]))


# =============================================================================
# UI
# =============================================================================
def render_sidebar() -> Settings:
    """Draw the sidebar controls and return the chosen settings."""
    sb = st.sidebar
    sb.header("Controls")
    benchmark = sb.selectbox("Benchmark", list(BENCHMARKS))
    timeframe = sb.radio("Timeframe", TIMEFRAMES, horizontal=True)
    years = sb.slider("Lookback (years of data)", *LOOKBACK_YEARS_RANGE, DEFAULT_LOOKBACK_YEARS)
    window = sb.number_input("RS-Ratio window (bars)", 5, 60, DEFAULT_RATIO_WINDOW)
    roc = sb.number_input("RS-Momentum ROC period (bars)", 1, 20, DEFAULT_ROC_PERIOD)
    tail = sb.slider("Tail length (bars)", *TAIL_RANGE, DEFAULT_TAIL)
    if sb.button("Refresh data"):
        st.cache_data.clear()
    return Settings(benchmark, timeframe, years, int(window), int(roc), tail)


def render_how_to_read() -> None:
    """Short in-app guide to reading the RRG."""
    with st.expander("How to read this chart"):
        st.markdown(
            "- **X axis (RS-Ratio):** right of 100 = outperforming the benchmark; left = underperforming.\n"
            "- **Y axis (RS-Momentum):** above 100 = relative strength is rising; below = falling.\n"
            "- **Quadrants:** Leading (strong and rising), Weakening (strong but fading), "
            "Lagging (weak and falling), Improving (weak but recovering).\n"
            "- **Typical cycle is clockwise:** Improving → Leading → Weakening → Lagging → Improving.\n"
            "- **Tails** show the recent path; older points are fainter and smaller. Long tails mean "
            "fast rotation, short tails mean a slow move.")


def render_sector_tabs(cfg: Settings, sector_daily: pd.DataFrame, bench_daily: pd.Series,
                       ratio: pd.DataFrame, mom: pd.DataFrame, selected: List[str]) -> None:
    """Render the four tabs."""
    tab_rrg, tab_rank, tab_drill, tab_heat = st.tabs(
        ["📈 Sector RRG", "🏆 Ranking", "🔍 Drill-down", "🔥 Returns heatmap"])
    label = f"{cfg.timeframe} data · vs {cfg.benchmark}"

    with tab_rrg:
        st.plotly_chart(build_rrg_figure(ratio[selected], mom[selected], cfg.tail,
                                         f"Sector Rotation ({label})"), key="rrg_sectors")
        render_how_to_read()

    with tab_rank:
        table = build_ranking_table(sector_daily[selected], ratio[selected], mom[selected])
        st.dataframe(style_table(table), height=min(650, 40 * len(table) + 60))
        st.caption("Rank = average rank across 1W/1M/3M/6M returns. Rotation arrow = direction of "
                   f"movement on the RRG over the last {DIRECTION_BARS} bars "
                   "(green = strengthening, amber = mixed, red = weakening).")

    with tab_drill:
        render_drilldown(cfg, sector_daily, selected, label)

    with tab_heat:
        returns = compute_returns(sector_daily[selected], RETURN_WINDOWS)
        excess = st.checkbox("Show excess return vs benchmark")
        if excess:
            bench_ret = compute_returns(bench_daily.to_frame(cfg.benchmark), RETURN_WINDOWS)
            returns = returns - bench_ret.iloc[0]
        st.plotly_chart(build_heatmap(returns, "Excess return vs " + cfg.benchmark if excess
                                      else "Sector returns"), key="heatmap")


def render_drilldown(cfg: Settings, sector_daily: pd.DataFrame, selected: List[str],
                     label: str) -> None:
    """Stock-level RRG versus the chosen sector index, plus RS gainers/laggards."""
    sector = st.selectbox("Sector", selected, key="drill_sector")
    tickers = {t.replace(".NS", ""): t for t in CONSTITUENTS.get(sector, [])}
    if not tickers:
        st.info("No constituents configured for this sector.")
        return
    with st.spinner(f"Loading {sector} constituents…"):
        stocks, failed = load_universe(tickers, cfg.lookback_years)
    if failed:
        st.warning("Skipped (no data): " + ", ".join(failed))
    if stocks.empty:
        st.error("No constituent data available.")
        return

    stocks_tf, sector_tf = align_to_benchmark(stocks, sector_daily[sector], cfg.timeframe)
    ratio, mom = compute_rrg(stocks_tf, sector_tf, cfg.ratio_window, cfg.roc_period)
    if ratio.dropna(how="all").empty:
        st.error("Not enough history for the chosen windows. Increase lookback or reduce windows.")
        return

    st.plotly_chart(build_rrg_figure(ratio, mom, cfg.tail, f"{sector} stocks vs sector index ({label})"),
                    key="rrg_stocks")

    delta = rs_change(ratio, cfg.tail).dropna().sort_values(ascending=False)
    fmt = {"Δ RS-Ratio": "{:+.2f}"}
    left, right = st.columns(2)
    left.subheader(f"Top RS gainers ({cfg.tail} bars)")
    left.dataframe(delta.head(TOP_N).rename("Δ RS-Ratio").to_frame().style.format(fmt))
    right.subheader(f"Top RS laggards ({cfg.tail} bars)")
    right.dataframe(delta.tail(TOP_N).sort_values().rename("Δ RS-Ratio").to_frame().style.format(fmt))

    stocks_daily = stocks.reindex(sector_daily[sector].dropna().index).ffill(limit=FFILL_LIMIT_BARS)
    st.subheader("All constituents")
    st.dataframe(style_table(build_ranking_table(stocks_daily, ratio, mom)))


def run_dashboard(cfg: Settings) -> None:
    """Load data, compute the RRG and render the tabs (returns early on unusable data)."""
    with st.spinner("Downloading benchmark and sector data…"):
        bench_px, bench_failed = load_universe({cfg.benchmark: BENCHMARKS[cfg.benchmark]},
                                               cfg.lookback_years)
        sector_px, sector_failed = load_universe(SECTOR_INDICES, cfg.lookback_years)
    if bench_failed or bench_px.empty:
        st.error(f"Could not load benchmark {cfg.benchmark}. Try again or pick the other one.")
        return
    if sector_failed:
        st.sidebar.warning("Skipped (no data): " + ", ".join(sector_failed))
    if sector_px.empty:
        st.error("No sector data could be loaded.")
        return

    bench_daily = bench_px[cfg.benchmark].dropna()
    sector_daily = sector_px.reindex(bench_daily.index).ffill(limit=FFILL_LIMIT_BARS)
    sectors_tf, bench_tf = align_to_benchmark(sector_px, bench_daily, cfg.timeframe)
    ratio, mom = compute_rrg(sectors_tf, bench_tf, cfg.ratio_window, cfg.roc_period)
    if ratio.dropna(how="all").empty:
        st.error("Not enough history for the chosen windows. Increase lookback or reduce windows.")
        return

    selected = st.sidebar.multiselect("Sectors", list(sector_px.columns),
                                      default=list(sector_px.columns))
    if not selected:
        st.info("Select at least one sector in the sidebar.")
        return

    st.caption(f"Latest data: {bench_daily.index[-1]:%d %b %Y} · cached up to 1 hour")
    render_sector_tabs(cfg, sector_daily, bench_daily, ratio, mom, selected)


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="NSE Sector Rotation", page_icon="🔄", layout="wide")
    st.title("🔄 NSE Sector Rotation")
    st.caption("Relative Rotation Graph (RRG) style analysis of NSE sector indices. Educational only.")
    cfg = render_sidebar()
    try:
        run_dashboard(cfg)
    except Exception as exc:
        st.error(f"Something went wrong: {exc}")
    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
