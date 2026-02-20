# Portfolio Management Dashboard

Interaktives Streamlit-Dashboard für Aktien- und ETF-Portfolios mit:

- Portfolio-Cockpit inkl. Benchmarkvergleich und KPIs (CAGR, Volatilität, Sharpe)
- Asset-Allocation nach Kategorie, Region, Sektor, Währung, Market Cap, Style
- Detaillierter Positionstabelle inkl. Zielpreis, Stop-Loss, Investment-These
- Diamanten-Klassifikation (1–5)
- Performance-, Risiko-, Faktor-, Szenario- und Rebalancing-Analyse
- Watchlist und Investment Journal (lokal in `data/investment_journal.csv`)

## Wo sehe ich das live?

### Option A: Lokal starten (am schnellsten)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Danach im Browser öffnen:

- `http://localhost:8501`

### Option B: Mit Docker starten

```bash
docker build -t portfolio-dashboard .
docker run --rm -p 8501:8501 portfolio-dashboard
```

Danach im Browser öffnen:

- `http://localhost:8501`

### Option C: Kostenlos deployen (Streamlit Community Cloud)

1. Repo nach GitHub pushen.
2. Auf https://share.streamlit.io/ anmelden.
3. „New app“ wählen und dieses Repo + `app.py` auswählen.
4. Deploy klicken.

Dann bekommst du einen öffentlichen Live-Link.
