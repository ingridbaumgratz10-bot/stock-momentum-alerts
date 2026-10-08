#!/usr/bin/env python3
"""Simulacion rapida del motor actualizado (RS Rating + Trend Template +
base/ruptura de pivote + stop de precio) sobre el universo actual de
Nasdaq-100 + S&P500, caminando dia por dia desde hace ~21 meses. Reusa las
funciones reales de daily_check.py -- no reinventa la logica. Caveat de
sobreviviencia: usa la composicion ACTUAL de los indices para todo el
periodo (no la composicion historica real de cada mes)."""
import sys
sys.path.insert(0, '.')
from datetime import date, timedelta
import pandas as pd
import daily_check as dc

TREND_LOOKBACK_DAYS = dc.TREND_LOOKBACK_DAYS
MA200_SLOPE_LOOKBACK = dc.MA200_SLOPE_LOOKBACK

print("Bajando universo (Nasdaq-100 + S&P500)...")
nasdaq100 = dc.fetch_index_tickers("https://www.slickcharts.com/nasdaq100")
sp500 = dc.fetch_index_tickers("https://www.slickcharts.com/sp500")
universe = sorted(set(nasdaq100) | set(sp500))

today = date.today()
SIM_START = date(2025, 1, 1)
DATA_START = (SIM_START - timedelta(days=500)).strftime("%Y-%m-%d")  # margen para RS + trend template + base
end = (today + timedelta(days=1)).strftime("%Y-%m-%d")

print(f"Bajando precios de {len(universe)} simbolos desde {DATA_START}...")
closes = dc.download_closes(universe, DATA_START, end)
print(f"OK: {closes.shape[0]} dias x {closes.shape[1]} simbolos")

# --- precalculo vectorizado del Trend Template (causal, sin look-ahead) ---
ma50 = closes.rolling(50).mean()
ma150 = closes.rolling(150).mean()
ma200 = closes.rolling(200).mean()
roll_low_52w = closes.rolling(TREND_LOOKBACK_DAYS).min()
roll_high_52w = closes.rolling(TREND_LOOKBACK_DAYS).max()
ma200_shift = ma200.shift(MA200_SLOPE_LOOKBACK)

def trend_ok_at(sym, i, rs_rating=None):
    if rs_rating is not None and rs_rating < dc.RS_MIN_RATING:
        return False
    p = closes[sym].iloc[i]
    m50, m150, m200, m200p = ma50[sym].iloc[i], ma150[sym].iloc[i], ma200[sym].iloc[i], ma200_shift[sym].iloc[i]
    lo, hi = roll_low_52w[sym].iloc[i], roll_high_52w[sym].iloc[i]
    if pd.isna(p) or pd.isna(m50) or pd.isna(m150) or pd.isna(m200) or pd.isna(m200p) or pd.isna(lo) or pd.isna(hi):
        return False
    return (p > m150 and p > m200 and m150 > m200 and m200 > m200p
            and m50 > m150 and m50 > m200 and p > m50
            and p >= lo * 1.25 and p >= hi * 0.75)

sim_dates = [d for d in closes.index if d.date() >= SIM_START]

leaders = {}        # symbol -> {"rs_rating": int|None}
watching = {}        # symbol -> {"kind": "base"|"dip", "pivot"/"wait_days", "ready_idx"}
positions = {}       # symbol -> {"entry_idx", "entry_date", "entry_price", "source"}
trades = []           # (symbol, entry_date, entry_price, exit_date, exit_price, reason, ret_pct, source)
lead_times = {"base": [], "dip": []}  # dias habiles entre el aviso y la ruptura/confirmacion real, por sistema
near_miss_by_month = {}  # month_key -> list of (sym, rs, fails)
last_month = None

def top_n_by_rs(sub_tickers, rs_rating_today):
    cols = [t for t in sub_tickers if t in rs_rating_today.index]
    if not cols:
        return []
    return list(rs_rating_today[cols].sort_values(ascending=False).head(dc.TOPN_PER_UNIVERSE).index)

print(f"Simulando {len(sim_dates)} dias desde {SIM_START}...")
for d in sim_dates:
    i = closes.index.get_loc(d)
    month_key = d.strftime("%Y-%m")

    if last_month != month_key:
        raw_scores = dc.calc_rs_raw_score(closes.iloc[: i + 1])
        rs_rating_today = dc.rs_percentile_ranks(raw_scores)
        new_leaders = list(dict.fromkeys(top_n_by_rs(nasdaq100, rs_rating_today) + top_n_by_rs(sp500, rs_rating_today)))
        leaders = {s: {"rs_rating": int(rs_rating_today[s]) if s in rs_rating_today.index else None} for s in new_leaders}

        nm = dc.find_near_misses(closes.iloc[: i + 1], rs_rating_today)
        nm = [x for x in nm if x[0] not in leaders]
        near_miss_by_month[month_key] = nm

        last_month = month_key

    for sym in list(positions.keys()):
        if sym not in closes.columns:
            continue
        price = closes[sym].iloc[i]
        if pd.isna(price):
            continue
        pos = positions[sym]
        stop_level = pos["entry_price"] * (1 - dc.STOP_LOSS_PCT / 100.0)
        if price <= stop_level:
            ret_pct = (price / pos["entry_price"] - 1) * 100
            trades.append((sym, pos["entry_date"], pos["entry_price"], d.date(), price, "STOP", ret_pct, pos["source"]))
            del positions[sym]
            continue
        bars_held = i - pos["entry_idx"]
        if bars_held >= dc.MAX_HOLD_DAYS:
            ret_pct = (price / pos["entry_price"] - 1) * 100
            trades.append((sym, pos["entry_date"], pos["entry_price"], d.date(), price, "TIME", ret_pct, pos["source"]))
            del positions[sym]

    for sym, lstate in leaders.items():
        if sym in positions or sym not in closes.columns:
            continue
        rs_rating = lstate.get("rs_rating")
        if not trend_ok_at(sym, i, rs_rating):
            if sym in watching:
                del watching[sym]
            continue

        series = closes[sym].iloc[: i + 1].dropna()
        wstate = watching.get(sym)

        if wstate is not None and wstate["kind"] == "dip":
            wstate["wait_days"] = wstate.get("wait_days", 0) + 1
            ref_level = float(series.iloc[-2])
            confirmed_price = dc.dip_reversal_confirmed(series, ref_level)
            if confirmed_price is not None:
                positions[sym] = {"entry_idx": i, "entry_date": d.date(), "entry_price": float(confirmed_price), "source": "dip"}
                lead_times["dip"].append(i - wstate["ready_idx"])
                del watching[sym]
            elif wstate["wait_days"] >= dc.MAX_WAIT_CONFIRM_DAYS:
                del watching[sym]
            continue

        if wstate is not None and wstate["kind"] == "base":
            broke = dc.base_pivot_breakout(series)
            if broke is not None:
                positions[sym] = {"entry_idx": i, "entry_date": d.date(), "entry_price": float(broke), "source": "base"}
                lead_times["base"].append(i - wstate["ready_idx"])
                del watching[sym]
                continue
            ready = dc.base_ready_pivot(series)
            if ready is None:
                del watching[sym]
                continue
            wstate["pivot"] = float(ready)
            continue

        # sym no esta en observacion todavia
        broke = dc.base_pivot_breakout(series)
        if broke is not None:
            positions[sym] = {"entry_idx": i, "entry_date": d.date(), "entry_price": float(broke), "source": "base"}
            lead_times["base"].append(0)  # rompio sin haber pasado por "base lista" (caso raro)
            continue

        ready = dc.base_ready_pivot(series)
        if ready is not None:
            watching[sym] = {"kind": "base", "pivot": float(ready), "ready_idx": i}
            continue

        dip_ref = dc.dip_watch_level(series)
        if dip_ref is not None:
            watching[sym] = {"kind": "dip", "wait_days": 0, "ready_idx": i}

# posiciones que quedaron abiertas al final de la simulacion
still_open = []
last_i = closes.index.get_loc(sim_dates[-1])
for sym, pos in positions.items():
    price = closes[sym].iloc[last_i]
    if pd.isna(price):
        continue
    ret_pct = (price / pos["entry_price"] - 1) * 100
    still_open.append((sym, pos["entry_date"], pos["entry_price"], price, ret_pct))

def print_stats(label, ts):
    print(f"\n--- {label} ---")
    print(f"Operaciones cerradas: {len(ts)}")
    if not ts:
        return
    wins = [t for t in ts if t[6] > 0]
    print(f"% acierto: {len(wins)/len(ts)*100:.1f}%")
    print(f"Retorno promedio por operacion: {sum(t[6] for t in ts)/len(ts):.2f}%")
    print(f"Retorno promedio ganadoras: {sum(t[6] for t in wins)/len(wins):.2f}%" if wins else "n/a")
    losses = [t for t in ts if t[6] <= 0]
    print(f"Retorno promedio perdedoras: {sum(t[6] for t in losses)/len(losses):.2f}%" if losses else "n/a")
    by_reason = {}
    for t in ts:
        by_reason[t[5]] = by_reason.get(t[5], 0) + 1
    print(f"Salidas por motivo: {by_reason}")

print(f"\n=== RESULTADOS COMBINADOS ({SIM_START} a {sim_dates[-1].date()}) ===")
print_stats("TODAS (base + dip)", trades)
print_stats("Solo sistema de BASE+PIVOTE", [t for t in trades if t[7] == "base"])
print_stats("Solo sistema de REVERSION (dip)", [t for t in trades if t[7] == "dip"])

print("\nDetalle de cada operacion:")
for t in sorted(trades, key=lambda x: x[1]):
    print(f"  {t[0]:6s} {t[1]} ${t[2]:.2f} -> {t[3]} ${t[4]:.2f}  [{t[5]}]  {t[6]:+.2f}%  ({t[7]})")

print(f"\nPosiciones abiertas al final (sin cerrar en la simulacion): {len(still_open)}")
for sym, ed, ep, cp, r in still_open:
    print(f"  {sym:6s} entro {ed} a ${ep:.2f}, hoy ${cp:.2f}  {r:+.2f}% (no realizado)")

for kind_label, kind_key in [("BASE+PIVOTE", "base"), ("REVERSION (dip)", "dip")]:
    lts = lead_times[kind_key]
    print(f"\n=== TIEMPO DE ANTICIPACION -- {kind_label} ===")
    if not lts:
        print("sin datos")
        continue
    lt_sorted = sorted(lts)
    n = len(lt_sorted)
    same_day = sum(1 for x in lts if x == 0)
    print(f"n={n}  promedio={sum(lts)/n:.1f} dias  mediana={lt_sorted[n//2]} dias  min={lt_sorted[0]}  max={lt_sorted[-1]}")
    print(f"mismo dia (sin aviso previo): {same_day} ({same_day/n*100:.0f}%)")
    buckets = {"0 dias": 0, "1-2 dias": 0, "3-5 dias": 0, "6-10 dias": 0, ">10 dias": 0}
    for x in lts:
        if x == 0: buckets["0 dias"] += 1
        elif x <= 2: buckets["1-2 dias"] += 1
        elif x <= 5: buckets["3-5 dias"] += 1
        elif x <= 10: buckets["6-10 dias"] += 1
        else: buckets[">10 dias"] += 1
    for k, v in buckets.items():
        print(f"  {k}: {v} ({v/n*100:.0f}%)")

print("\n=== FRECUENCIA (operaciones por mes calendario) ===")
from collections import Counter
month_counts = Counter(t[1].strftime("%Y-%m") for t in trades)
for m in sorted(month_counts):
    print(f"  {m}: {month_counts[m]}")
n_months = len(month_counts) if month_counts else 1
print(f"promedio: {len(trades)/21.2:.1f} operaciones/mes (sobre ~21.2 meses simulados)")

print("\n=== CASI CALIFICAN (por mes, y si se graduaron a lider al mes siguiente) ===")
months_sorted = sorted(near_miss_by_month.keys())
total_nm = 0
graduated = 0
for idx, m in enumerate(months_sorted):
    nm_list = near_miss_by_month[m]
    total_nm += len(nm_list)
    next_leaders = set()
    if idx + 1 < len(months_sorted):
        # recalcular los lideres del mes siguiente para chequear graduacion
        next_month = months_sorted[idx + 1]
        # usamos la lista real de leaders guardada solo para el ULTIMO mes simulado;
        # para los anteriores, aproximamos re-derivando de near_miss_by_month no alcanza,
        # asi que marcamos graduacion solo si el simbolo aparece en los trades del mes siguiente
        next_month_syms = {t[0] for t in trades if t[1].strftime('%Y-%m') == next_month}
        graduated += sum(1 for sym, rs, fails in nm_list if sym in next_month_syms)
    print(f"  {m}: {len(nm_list)} casi-califican")
print(f"\nPromedio: {total_nm/len(months_sorted):.1f} casi-califican por mes")
print(f"De esos, cuantos aparecieron operando (base o dip) al mes siguiente: {graduated} de {total_nm} ({graduated/total_nm*100:.1f}%)" if total_nm else "sin datos")
