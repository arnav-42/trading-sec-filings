"""
Very simple long/short back-test driven by sentiment scores.
Rules:
  sentiment >= +thr  → LONG
  sentiment <= -thr  → SHORT
  flat otherwise
Entry = next market open; exit after fixed holding period
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from utils import CFG, LOG

THR_POS = CFG["llm"]["sentiment_thresholds"]["positive"]
THR_NEG = CFG["llm"]["sentiment_thresholds"]["negative"]
HOLD_DAYS = CFG["backtest"]["holding_period_days"]

def load_sentiment() -> pd.DataFrame:
    rows = []
    with open(CFG["llm"]["output_file"], encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rows.append(
                {
                    "date": pd.to_datetime(d["date"]),
                    "ticker": d["ticker"],
                    "sentiment": d["sentiment"],
                }
            )
    return pd.DataFrame(rows)

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df["signal"] = 0
    df.loc[df.sentiment >= THR_POS, "signal"] = 1
    df.loc[df.sentiment <= THR_NEG, "signal"] = -1
    return df[df.signal != 0]

def simulate(tkr: str, signals: pd.DataFrame) -> Dict:
    price = yf.download(
        tkr,
        start=CFG["backtest"]["start_date"],
        end=CFG["backtest"]["end_date"],
        progress=False,
    )["Adj Close"]

    cash, pos, trades = CFG["backtest"]["initial_cash"], 0, []
    for _, row in signals.iterrows():
        entry_date = row["date"] + timedelta(days=1)
        if entry_date not in price.index:  # skip non-trading days
            continue
        entry_px = price.loc[entry_date]
        shares = cash * 0.1 / entry_px  # risk 10 % equity each trade
        pos += shares * row["signal"]
        cash -= shares * entry_px * row["signal"]
        exit_date = entry_date + timedelta(days=HOLD_DAYS)
        if exit_date in price.index:
            exit_px = price.loc[exit_date]
            cash += shares * exit_px * row["signal"]
            pos -= shares * row["signal"]
            trades.append(
                {
                    "entry": entry_date,
                    "exit": exit_date,
                    "direction": "LONG" if row["signal"] == 1 else "SHORT",
                    "pnl": (exit_px - entry_px) * shares * row["signal"],
                }
            )
    return {
        "ticker": tkr,
        "cash": cash,
        "trades": trades,
        "pnl_total": sum(t["pnl"] for t in trades),
    }

def main():
    df = load_sentiment()
    report = []
    for tkr in CFG["tickers"]:
        sigs = generate_signals(df[df.ticker == tkr])
        res = simulate(tkr, sigs)
        report.append(res)

    rpt_df = pd.DataFrame(
        {
            "Ticker": [r["ticker"] for r in report],
            "Trades": [len(r["trades"]) for r in report],
            "Total PnL": [r["pnl_total"] for r in report],
            "Final Cash": [r["cash"] for r in report],
        }
    )
    LOG.info("\n%s", rpt_df.to_string(index=False))

if __name__ == "__main__":
    main()
