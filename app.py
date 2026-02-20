from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Portfolio Management Dashboard", layout="wide")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
JOURNAL_FILE = DATA_DIR / "investment_journal.csv"

np.random.seed(11)


def load_portfolio() -> pd.DataFrame:
    rows = [
        ["AAPL", "Apple", "Aktie", 40, 145, 182, "USA", "Tech", "USD", "Large", "Growth", "Long", 210, 150, "Ökosystem + Free Cashflow", 5],
        ["MSFT", "Microsoft", "Aktie", 26, 280, 418, "USA", "Tech", "USD", "Large", "Growth", "Long", 460, 360, "Cloud + KI-Skalierung", 5],
        ["NOVN", "Novartis", "Aktie", 110, 82, 94, "Europa", "Healthcare", "CHF", "Large", "Value", "Long", 102, 84, "Defensiver Cashflow", 4],
        ["NESN", "Nestlé", "Aktie", 90, 111, 96, "Europa", "Consumer", "CHF", "Large", "Value", "Long", 108, 88, "Pricing-Power", 4],
        ["TSLA", "Tesla", "Aktie", 28, 240, 198, "USA", "Consumer", "USD", "Large", "Growth", "Mid", 260, 170, "Volatiler Growth-Titel", 2],
        ["VWCE", "Vanguard FTSE All-World", "ETF", 120, 98, 128, "Global", "Diversified", "USD", "Large", "Blend", "Long", 140, 108, "Breite Weltdiversifikation", 5],
        ["EIMI", "iShares Core EM IMI", "ETF", 210, 26, 30, "EM", "Diversified", "USD", "Mid", "Value", "Long", 33, 24, "EM-Beta", 3],
        ["SMH", "VanEck Semiconductor", "ETF", 35, 194, 225, "USA", "Tech", "USD", "Mid", "Growth", "Mid", 250, 185, "Halbleiter-Zyklus", 4],
        ["ORCL", "Oracle", "Aktie", 42, 95, 126, "USA", "Tech", "USD", "Large", "Value", "Mid", 136, 104, "Cloud-Migration", 3],
        ["SREN", "Swiss Re", "Aktie", 65, 86, 113, "Europa", "Financials", "CHF", "Large", "Value", "Long", 120, 98, "Zinshebel + Rückversicherung", 4],
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "Ticker",
            "Name",
            "Kategorie",
            "Stückzahl",
            "Kaufpreis",
            "Aktueller Preis",
            "Region",
            "Sektor",
            "Währung",
            "Market Cap",
            "Style",
            "Investment-Horizont",
            "Zielpreis",
            "Stop-Loss",
            "Investment-These",
            "Diamanten",
        ],
    )
    base_day = date.today() - timedelta(days=500)
    df["Kaufdatum"] = [base_day + timedelta(days=i * 27) for i in range(len(df))]
    return df


def generate_timeseries(df: pd.DataFrame, periods: int = 400):
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=periods)
    vol_map = {"ETF": 0.011, "Aktie": 0.015}
    drift_map = {"ETF": 0.00035, "Aktie": 0.0005}
    prices = {}
    for _, row in df.iterrows():
        start = row["Kaufpreis"]
        rets = np.random.normal(drift_map[row["Kategorie"]], vol_map[row["Kategorie"]], periods)
        series = start * np.exp(np.cumsum(rets))
        series *= row["Aktueller Preis"] / series[-1]
        prices[row["Ticker"]] = series
    price_df = pd.DataFrame(prices, index=dates)
    bench_rets = np.random.normal(0.00035, 0.01, periods)
    bench = 100 * np.exp(np.cumsum(bench_rets))
    bench *= (1 + portfolio_return_since_start(df))
    bench /= bench[-1]
    benchmark = pd.Series(bench, index=dates, name="MSCI World Proxy")
    return price_df, benchmark


def portfolio_return_since_start(df: pd.DataFrame):
    return (df["Aktueller Preis"].mul(df["Stückzahl"]).sum() - df["Kaufpreis"].mul(df["Stückzahl"]).sum()) / df[
        "Kaufpreis"
    ].mul(df["Stückzahl"]).sum()


def compute_metrics(df, price_df, benchmark, cash=22000, realized_pl=6800):
    df = df.copy()
    df["Positionswert"] = df["Stückzahl"] * df["Aktueller Preis"]
    df["Investiert"] = df["Stückzahl"] * df["Kaufpreis"]
    df["P/L absolut"] = df["Positionswert"] - df["Investiert"]
    df["P/L %"] = df["P/L absolut"] / df["Investiert"]
    total_value = df["Positionswert"].sum() + cash
    invested = df["Investiert"].sum()
    unrealized = df["P/L absolut"].sum()
    cash_quote = cash / total_value

    weights = df.set_index("Ticker")["Positionswert"] / df["Positionswert"].sum()
    port_returns = price_df.pct_change().dropna().mul(weights, axis=1).sum(axis=1)
    bench_returns = benchmark.pct_change().dropna()

    days = len(port_returns)
    years = days / 252
    cumulative = (1 + port_returns).prod() - 1
    cagr = (1 + cumulative) ** (1 / years) - 1
    vol = port_returns.std() * np.sqrt(252)
    sharpe = (cagr - 0.02) / vol if vol > 0 else np.nan

    ytd = (1 + port_returns[port_returns.index.year == port_returns.index.max().year]).prod() - 1
    since_start = (df["Positionswert"].sum() - invested) / invested
    beta = np.cov(port_returns, bench_returns.loc[port_returns.index])[0, 1] / np.var(bench_returns.loc[port_returns.index])

    drawdown = (1 + port_returns).cumprod() / (1 + port_returns).cumprod().cummax() - 1
    max_dd = drawdown.min()
    var95 = np.percentile(port_returns, 5)

    concentration = (df.nlargest(5, "Positionswert")["Positionswert"].sum() / df["Positionswert"].sum())

    return {
        "table": df,
        "total_value": total_value,
        "invested": invested,
        "unrealized": unrealized,
        "realized": realized_pl,
        "cash_quote": cash_quote,
        "ytd": ytd,
        "since_start": since_start,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "beta": beta,
        "max_dd": max_dd,
        "var95": var95,
        "port_returns": port_returns,
        "bench_returns": bench_returns,
        "drawdown": drawdown,
        "concentration": concentration,
    }


def render():
    st.title("📊 Interaktives Portfolio-Management-Dashboard")

    df = load_portfolio()
    price_df, benchmark = generate_timeseries(df)
    metrics = compute_metrics(df, price_df, benchmark)
    table = metrics["table"]

    tab_names = [
        "1) Portfolio Cockpit",
        "2) Asset Allocation",
        "3) Positionen",
        "4) Diamanten",
        "5) Performance",
        "6) Risiko",
        "7) Faktor-Exposure",
        "8) Szenarien",
        "9) Rebalancing",
        "10) Watchlist",
        "11) Investment Journal",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gesamtportfolio-Wert", f"CHF {metrics['total_value']:,.0f}")
        c2.metric("Investiertes Kapital", f"CHF {metrics['invested']:,.0f}")
        c3.metric("Unrealized P/L", f"CHF {metrics['unrealized']:,.0f}", f"{metrics['since_start']:.1%}")
        c4.metric("Realized P/L", f"CHF {metrics['realized']:,.0f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Cash-Quote", f"{metrics['cash_quote']:.1%}")
        c6.metric("Portfolio-Return YTD", f"{metrics['ytd']:.1%}")
        c7.metric("Portfolio-Return seit Start", f"{metrics['since_start']:.1%}")
        bench_total = (1 + metrics["bench_returns"]).prod() - 1
        c8.metric("Benchmark-Vergleich", f"{metrics['since_start'] - bench_total:.1%}")

        c9, c10, c11 = st.columns(3)
        c9.metric("CAGR", f"{metrics['cagr']:.1%}")
        c10.metric("Volatilität", f"{metrics['vol']:.1%}")
        c11.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")

        port_curve = (1 + metrics["port_returns"]).cumprod()
        perf_df = pd.DataFrame({"Portfolio": port_curve, "Benchmark": benchmark.loc[port_curve.index]})
        st.plotly_chart(px.line(perf_df, title="Portfolio Value Timeline"), use_container_width=True)

        relative_df = perf_df / perf_df.iloc[0]
        st.plotly_chart(px.line(relative_df, title="Performance vs Benchmark (rebased=1)"), use_container_width=True)

    with tabs[1]:
        table["Gewicht"] = table["Positionswert"] / table["Positionswert"].sum()
        for col in ["Kategorie", "Region", "Sektor", "Währung", "Market Cap", "Style"]:
            alloc = table.groupby(col, as_index=False)["Gewicht"].sum()
            st.plotly_chart(px.pie(alloc, values="Gewicht", names=col, title=f"Allokation nach {col}"), use_container_width=True)

    with tabs[2]:
        st.dataframe(
            table[
                [
                    "Ticker",
                    "Name",
                    "Kategorie",
                    "Stückzahl",
                    "Kaufpreis",
                    "Aktueller Preis",
                    "Positionswert",
                    "P/L absolut",
                    "P/L %",
                    "Gewicht",
                    "Kaufdatum",
                    "Investment-Horizont",
                    "Zielpreis",
                    "Stop-Loss",
                    "Investment-These",
                ]
            ].style.format(
                {
                    "Kaufpreis": "{:.2f}",
                    "Aktueller Preis": "{:.2f}",
                    "Positionswert": "{:.0f}",
                    "P/L absolut": "{:.0f}",
                    "P/L %": "{:.1%}",
                    "Gewicht": "{:.1%}",
                }
            ),
            use_container_width=True,
            height=480,
        )

    with tabs[3]:
        d_dist = table.groupby("Diamanten", as_index=False).size()
        d_weight = table.groupby("Diamanten", as_index=False)["Gewicht"].sum()
        st.plotly_chart(px.bar(d_dist, x="Diamanten", y="size", title="Verteilung der Diamanten-Klassifikation"), use_container_width=True)
        st.plotly_chart(px.bar(d_weight, x="Diamanten", y="Gewicht", title="Portfolio-Gewicht pro Diamanten-Kategorie"), use_container_width=True)

    with tabs[4]:
        top = table.sort_values("P/L %", ascending=False)
        c1, c2 = st.columns(2)
        c1.dataframe(top[["Ticker", "Name", "P/L %"]].head(5).style.format({"P/L %": "{:.1%}"}), use_container_width=True)
        c2.dataframe(top[["Ticker", "Name", "P/L %"]].tail(5).style.format({"P/L %": "{:.1%}"}), use_container_width=True)

        contrib = table[["Ticker", "P/L absolut"]].sort_values("P/L absolut", ascending=False)
        st.plotly_chart(px.bar(contrib, x="Ticker", y="P/L absolut", title="Contribution-Analyse pro Position"), use_container_width=True)

        rolling = (1 + metrics["port_returns"]).rolling(63).apply(np.prod, raw=True) - 1
        st.plotly_chart(px.line(rolling, title="Rolling Performance (63 Tage)"), use_container_width=True)
        st.plotly_chart(px.area(metrics["drawdown"], title="Drawdown-Chart"), use_container_width=True)

    with tabs[5]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Portfolio-Volatilität", f"{metrics['vol']:.1%}")
        c2.metric("Max Drawdown", f"{metrics['max_dd']:.1%}")
        c3.metric("Beta vs Benchmark", f"{metrics['beta']:.2f}")
        c4, c5 = st.columns(2)
        c4.metric("Konzentrationsrisiko Top-5", f"{metrics['concentration']:.1%}")
        c5.metric("Value at Risk (95%, 1d)", f"{metrics['var95']:.2%}")

        corr = price_df.pct_change().dropna().corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0))
        fig.update_layout(title="Korrelationsmatrix der Positionen")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[6]:
        factor_exposure = pd.DataFrame(
            {
                "Faktor": ["Growth", "Value", "Momentum", "Quality", "Size", "Low Volatility"],
                "Gewichtung": [0.31, 0.25, 0.14, 0.12, 0.10, 0.08],
            }
        )
        st.plotly_chart(px.bar(factor_exposure, x="Faktor", y="Gewichtung", title="Faktor-Gewichtung"), use_container_width=True)

    with tabs[7]:
        scenario_shocks = {
            "Zinsanstieg": -0.06,
            "Tech-Crash": -0.18,
            "USD-Schwäche": -0.04,
            "Rezession": -0.13,
            "Markt-Korrektur -20%": -0.20,
        }
        tech_weight = table.loc[table["Sektor"] == "Tech", "Gewicht"].sum()
        usd_weight = table.loc[table["Währung"] == "USD", "Gewicht"].sum()
        cyc_weight = table.loc[table["Sektor"].isin(["Consumer", "Financials", "Tech"]), "Gewicht"].sum()

        impacts = []
        for name, shock in scenario_shocks.items():
            if name == "Tech-Crash":
                impact = shock * tech_weight
            elif name == "USD-Schwäche":
                impact = shock * usd_weight
            elif name == "Rezession":
                impact = shock * cyc_weight
            else:
                impact = shock
            impacts.append([name, impact])
        scen_df = pd.DataFrame(impacts, columns=["Szenario", "Portfolio-Drawdown"])
        st.plotly_chart(px.bar(scen_df, x="Szenario", y="Portfolio-Drawdown", title="Szenario-Analyse"), use_container_width=True)

    with tabs[8]:
        st.subheader("Rebalancing-Panel")
        target = table.groupby("Kategorie", as_index=False)["Gewicht"].sum().copy()
        target["Zielgewicht"] = [0.55, 0.35] if len(target) == 2 else np.repeat(1 / len(target), len(target))
        target["Abweichung"] = target["Gewicht"] - target["Zielgewicht"]
        target["Signal"] = np.where(target["Abweichung"] > 0.03, "Reduzieren", np.where(target["Abweichung"] < -0.03, "Aufstocken", "Halten"))
        st.dataframe(target.style.format({"Gewicht": "{:.1%}", "Zielgewicht": "{:.1%}", "Abweichung": "{:.1%}"}), use_container_width=True)

    with tabs[9]:
        watchlist = pd.DataFrame(
            [
                ["ASML", 780, 900, "EUV-Monopol + KI-Nachfrage", "Auftragseingang > Konsens"],
                ["LVMH", 640, 760, "Luxus mit starker Marke", "China-Nachfrage stabilisiert"],
                ["ISRG", 300, 370, "Robotik in MedTech", "Neue Produktzulassung"],
            ],
            columns=["Aktie", "Ziel-Entry-Preis", "Fair-Value", "Investment-These", "Trigger-Event"],
        )
        st.data_editor(watchlist, use_container_width=True, num_rows="dynamic")

    with tabs[10]:
        if JOURNAL_FILE.exists():
            journal = pd.read_csv(JOURNAL_FILE)
        else:
            journal = pd.DataFrame(
                columns=["Datum", "Ticker", "Kaufentscheidung", "Erwartung", "Risiken", "Exit-Kriterien"]
            )

        edited = st.data_editor(journal, num_rows="dynamic", use_container_width=True)
        if st.button("Journal speichern"):
            edited.to_csv(JOURNAL_FILE, index=False)
            st.success("Investment Journal gespeichert.")

    st.caption("Hinweis: Demo-Daten mit modellhaften Risiko- und Faktorannahmen für Portfolio-Steuerung.")


if __name__ == "__main__":
    render()
