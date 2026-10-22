# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:44:22 2026

@author: manan
"""

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime


start_date = "2025-07-01"
end_date = "2025-07-18"

silver = yf.download("SI=F", period="60d", interval="15m")
gold = yf.download("GC=F", period="60d", interval="15m")
print(silver, gold)

# Convert to IST (Indian Standard Time)
gold.index = gold.index.tz_convert("Asia/Kolkata")
silver.index = silver.index.tz_convert("Asia/Kolkata")


if gold.empty or silver.empty:
    raise ValueError("Gold or Silver data not available. Check the ticker and interval.")

data = pd.concat([gold["Close"], silver["Close"]], axis=1)
data.columns = ["Gold", "Silver"]
data = data.dropna()

model = sm.OLS(data["Gold"], sm.add_constant(data["Silver"])).fit()
hedge_ratio = model.params[1]
print(f"Hedge Ratio (Silver per Gold): {hedge_ratio:.4f}")

data["Spread"] = data["Gold"] - hedge_ratio * data["Silver"]
data["Z_Score"] = (data["Spread"] - data["Spread"].rolling(60).mean()) / data["Spread"].rolling(60).std()
data = data.dropna()


entry_z = 2.0   
exit_z = 0.5   

data["Position"] = 0
data.loc[data["Z_Score"] > entry_z, "Position"] = -2   # Short spread (sell gold, buy silver)
data.loc[data["Z_Score"] < -entry_z, "Position"] = 2   # Long spread (buy gold, sell silver)

data["Position"] = data["Position"].replace(0, np.nan).ffill()
data["Position"] = data["Position"].fillna(0)

data.loc[(abs(data["Z_Score"]) < exit_z), "Position"] = 0

# -----------------------------
# Backtest
# -----------------------------
trades = []
position_prev = 0
entry_price_gold = entry_price_silver = entry_time = None

for timestamp, row in data.iterrows():
    pos = row["Position"]
    gold_price = row["Gold"]
    silver_price = row["Silver"]

    # Entry
    if position_prev == 0 and pos != 0:
        entry_price_gold = gold_price
        entry_price_silver = silver_price
        entry_time = timestamp

        gold_units = 1
        silver_units = hedge_ratio
        trade_value = abs(gold_units * gold_price) + abs(silver_units * silver_price)

        trades.append([
            timestamp, "ENTER LONG" if pos == 1 else "ENTER SHORT",
            gold_price, silver_price, 0.0, trade_value
        ])

    # Exit
    elif pos == 0 and position_prev != 0:
        pnl = (gold_price - entry_price_gold) - hedge_ratio * (silver_price - entry_price_silver)
        if position_prev == -1:
            pnl *= -1

        gold_units = 1
        silver_units = hedge_ratio
        trade_value = abs(gold_units * gold_price) + abs(silver_units * silver_price)

        trades.append([
            timestamp, "EXIT", gold_price, silver_price, pnl, trade_value
        ])

    position_prev = pos


trade_log = pd.DataFrame(trades, columns=["Time (IST)", "Action", "Gold Price", "Silver Price", "PnL", "Trade Value"])
trade_log["PnL"] = trade_log["PnL"].fillna(0)
trade_log["Cumulative PnL"] = trade_log["PnL"].cumsum()


print("\n--- TRADE LOG ---")
print(trade_log.tail(10))

print(f"\nTotal Trades: {len(trade_log)//2}")
print(f"Net PnL: {trade_log['PnL'].sum():.4f}")

# Plot Spread & Z-Score
plt.figure(figsize=(12,6))
plt.plot(data["Spread"], label="Spread")
plt.axhline(data["Spread"].mean(), color="black", linestyle="--", label="Mean")
plt.axhline(data["Spread"].mean() + entry_z * data["Spread"].std(), color="red", linestyle="--", label="+2σ Entry")
plt.axhline(data["Spread"].mean() - entry_z * data["Spread"].std(), color="green", linestyle="--", label="-2σ Entry")
plt.title("Gold-Silver Spread (15-min, IST)")
plt.legend()
plt.show()

# Plot cumulative PnL
plt.figure(figsize=(10,5))
plt.plot(trade_log["Time (IST)"], trade_log["Cumulative PnL"], marker="o")
plt.title("Cumulative PnL Over Time")
plt.xlabel("Time (IST)")
plt.ylabel("PnL")
plt.grid(True)
plt.show()