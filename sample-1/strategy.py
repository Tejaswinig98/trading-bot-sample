 """Trading logic. Here the decision is always BUY - build from here to add your own strategy."""

 # Fixed USDT amount to spend on each run.
 TRADE_USD = 7.00


 def decide(price):
     """Return the action for this run. Always buys (price is unused, here for easy upgrades)."""
     action = "BUY"
     return action

# """
# Machine-learning + risk-controlled strategy for the ANDX competition.

# Scoring rules considered:
# - Keep average daily valid USD trading volume above $2,500 over 7 days.
# - Minimum order size is $5.
# - Avoid large drawdowns: daily stop and weekly stop are much safer than the
#   competition suspension/disqualification limits.

# ML is used only for signal confidence. Risk controls decide whether trading is allowed.
# This is educational sample code, not financial advice and not a guarantee of profit.
# """
# from dataclasses import dataclass
# import numpy as np
# import pandas as pd
# import ta

# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler

# WARMUP = 220

# PARAMS = {
#     "future_bars": 3,
#     "min_train_rows": 130,
#     "strong_buy_confidence": 0.65,
#     "volume_buy_confidence": 0.55,
#     "bearish_exit_confidence": 0.42,
#     "ema_fast": 9,
#     "ema_slow": 21,
#     "rsi_period": 14,
#     "atr_period": 14,
#     "tp_mult": 2.2,
#     "sl_mult": 1.2,
#     "trailing_atr_mult": 1.0,
#     "min_rsi_buy": 38,
#     "max_rsi_buy": 74,
#     "max_atr_pct": 0.050,
#     # To help volume, the bot can exit after this many runs if the day is behind target.
#     "volume_assist_hold_runs": 1,
# }

# @dataclass
# class Decision:
#     action: str
#     price: float
#     tp_price: float = 0.0
#     sl_price: float = 0.0
#     confidence: float = 0.0
#     note: str = ""

# FEATURE_COLS = [
#     "ret_1", "ret_3", "ret_6", "ret_12", "ema_gap", "rsi", "macd",
#     "atr_pct", "vol_chg", "volume_z", "hour_momentum", "day_momentum",
# ]

# def add_features(df: pd.DataFrame) -> pd.DataFrame:
#     d = df.copy()
#     close, high, low, volume = d["close"], d["high"], d["low"], d["volume"]
#     d["ret_1"] = close.pct_change(1)
#     d["ret_3"] = close.pct_change(3)
#     d["ret_6"] = close.pct_change(6)
#     d["ret_12"] = close.pct_change(12)
#     d["ema_fast"] = ta.trend.ema_indicator(close, window=PARAMS["ema_fast"])
#     d["ema_slow"] = ta.trend.ema_indicator(close, window=PARAMS["ema_slow"])
#     d["ema_gap"] = (d["ema_fast"] - d["ema_slow"]) / close
#     d["rsi"] = ta.momentum.rsi(close, window=PARAMS["rsi_period"])
#     d["macd"] = ta.trend.macd_diff(close)
#     d["atr"] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=PARAMS["atr_period"]).average_true_range()
#     d["atr_pct"] = d["atr"] / close
#     d["vol_chg"] = volume.pct_change(6).replace([np.inf, -np.inf], np.nan)
#     d["volume_z"] = (volume - volume.rolling(24).mean()) / volume.rolling(24).std()
#     d["hour_momentum"] = close / close.rolling(6).mean() - 1
#     d["day_momentum"] = close / close.rolling(24).mean() - 1
#     return d

# def ml_probability(df: pd.DataFrame):
#     d = add_features(df)
#     future_return = d["close"].shift(-PARAMS["future_bars"]) / d["close"] - 1
#     d["target"] = (future_return > 0.0015).astype(int)

#     train = d.dropna().iloc[:-PARAMS["future_bars"]]
#     latest = d.dropna().iloc[-1:]
#     if len(train) < PARAMS["min_train_rows"] or latest.empty or train["target"].nunique() < 2:
#         return 0.50, d

#     X, y = train[FEATURE_COLS], train["target"].astype(int)
#     X_latest = latest[FEATURE_COLS]
#     models = [
#         RandomForestClassifier(n_estimators=140, max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=7),
#         GradientBoostingClassifier(n_estimators=90, learning_rate=0.05, max_depth=2, random_state=7),
#         make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")),
#     ]
#     probs = []
#     for model in models:
#         try:
#             model.fit(X, y)
#             probs.append(float(model.predict_proba(X_latest)[0][1]))
#         except Exception:
#             pass
#     return (float(np.mean(probs)) if probs else 0.50), d

# def decide(df: pd.DataFrame, state: dict, daily_volume_target: float = 3200.0) -> Decision:
#     prob_up, d = ml_probability(df)
#     last = d.dropna().iloc[-1]
#     price = float(last["close"])
#     atr = float(last["atr"])
#     atr_pct = float(last["atr_pct"])
#     rsi = float(last["rsi"])
#     ema_gap = float(last["ema_gap"])
#     macd = float(last["macd"])
#     momentum = float(last["hour_momentum"])

#     daily_vol = float(state.get("daily", {}).get("volume_usd", 0) or 0)
#     behind_volume = daily_vol < daily_volume_target
#     hold_runs = int(state.get("hold_runs", 0) or 0)

#     if state.get("position") == "LONG":
#         tp_price = float(state.get("tp_price") or 0)
#         sl_price = float(state.get("sl_price") or 0)
#         trailing_sl = price - PARAMS["trailing_atr_mult"] * atr
#         effective_sl = max(sl_price, trailing_sl) if atr > 0 else sl_price

#         if price >= tp_price:
#             return Decision("SELL", price, confidence=1.0, note="take-profit hit")
#         if effective_sl > 0 and price <= effective_sl:
#             return Decision("SELL", price, confidence=1.0, note="stop-loss/trailing-stop hit")
#         if prob_up <= PARAMS["bearish_exit_confidence"] and rsi < 50:
#             return Decision("SELL", price, confidence=1 - prob_up, note=f"ML bearish exit prob_up={prob_up:.2f}")
#         if behind_volume and hold_runs >= PARAMS["volume_assist_hold_runs"] and prob_up < 0.70:
#             return Decision("SELL", price, confidence=prob_up, note=f"volume-assist exit after {hold_runs} run(s), prob_up={prob_up:.2f}")
#         return Decision("HOLD", price, confidence=prob_up, note=f"holding prob_up={prob_up:.2f}")

#     uptrend = ema_gap > 0
#     good_rsi = PARAMS["min_rsi_buy"] <= rsi <= PARAMS["max_rsi_buy"]
#     volatility_ok = atr_pct <= PARAMS["max_atr_pct"]
#     market_ok = volatility_ok and momentum > -0.006 and macd > -abs(price) * 0.0005

#     strong_entry = prob_up >= PARAMS["strong_buy_confidence"] and uptrend and good_rsi and market_ok
#     volume_entry = behind_volume and prob_up >= PARAMS["volume_buy_confidence"] and good_rsi and volatility_ok and momentum > -0.010

#     if atr > 0 and (strong_entry or volume_entry):
#         tp_price = price + PARAMS["tp_mult"] * atr
#         sl_price = price - PARAMS["sl_mult"] * atr
#         reason = "strong ML entry" if strong_entry else "volume-assist ML entry"
#         return Decision("BUY", price, tp_price, sl_price, prob_up, f"{reason}: prob_up={prob_up:.2f}, RSI={rsi:.1f}, ATR%={atr_pct:.2%}")

#     return Decision("HOLD", price, confidence=prob_up, note=f"waiting prob_up={prob_up:.2f}, RSI={rsi:.1f}, trend={uptrend}, volume_behind={behind_volume}")

