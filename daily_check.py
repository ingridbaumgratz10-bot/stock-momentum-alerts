#!/usr/bin/env python3
"""
Motor de senal diario, estilo Minervini: lideres por Relative Strength Rating
(Nasdaq-100 + S&P500), filtrados por Trend Template, que rompen el pivote de
una base comprimida (testeos repetidos de la misma resistencia, cada
retroceso mas angosto que el anterior). Sin ejecutar ordenes reales -- solo
manda un mail cuando hay que comprar o vender.

Se corre una vez por dia (tarea programada en la nube). Mantiene su estado
(lista de lideres del mes, posiciones "abiertas" presumidas) en state.json,
versionado en este mismo repo.
"""
import json, os, smtplib, ssl
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from io import StringIO

import requests
import pandas as pd
import yfinance as yf

STATE_PATH = "state.json"
CONFIG_PATH = "config.json"

LOOKBACK_MOMENTUM = 126      # ~6 meses habiles
TOPN_PER_UNIVERSE = 20
MAX_HOLD_DAYS = 5

# Trend Template (Minervini) y stop de precio
TREND_LOOKBACK_DAYS = 252        # ~52 semanas, para minimo/maximo y para tener suficiente historia de MA200
MA200_SLOPE_LOOKBACK = 21        # ~1 mes habil, para confirmar que la MA200 viene ascendiendo
STOP_LOSS_PCT = 8.0              # stop fijo desde el precio de entrada (regla clasica Minervini/O'Neil: nunca dejar correr una perdida mas alla de esto)

# Relative Strength Rating (formula de O'Neil/IBD): ultimo trimestre pesa el doble que cada uno de los 3 anteriores
RS_QUARTER_DAYS = 63             # ~1 trimestre habil
RS_MIN_RATING = 70               # percentil minimo (1-99) para contar como "lider" de verdad

# Base + ruptura de pivote (patron real de Minervini: varios testeos de la misma
# resistencia, cada retroceso mas angosto que el anterior, ruptura por encima)
BASE_LOOKBACK_DAYS = 40           # ventana de la base, ~8 semanas
BASE_TEST_TOLERANCE_PCT = 3.0     # un dia "testea" la resistencia si esta a <=3% de ella
BASE_MIN_TESTS = 2                # minimo de testeos distintos de la resistencia
BASE_PEAK_WINDOW = 3              # dias a cada lado para confirmar un maximo local
BASE_TIGHTENING_SLACK = 1.10      # cada retroceso puede ser hasta 10% mas profundo que el anterior y seguir contando como "se achica"
BASE_RECENCY_BUFFER_DAYS = 0      # colchon de dias desde el ultimo testeo antes de avisar "base lista" -- probado con 2 y 3 dias, EMPEORA el aviso temprano (ver CLAUDE.md / notas): muchas rupturas pasan 1-2 dias despues del ultimo testeo, exigir mas colchon hace que se pierdan del aviso en vez de ganarlo. Dejar en 0.

# Reversion de corto plazo (sistema paralelo, complementa al de base+pivote:
# pullback corto dentro de una tendencia ya validada, no una base lateral)
REV_LOOKBACK = 5                 # dias para medir la caida reciente
ENTRY_THRESHOLD_PCT = -5.0        # caida minima en REV_LOOKBACK dias para considerar el pullback
MAX_DROP_PCT = -35.0              # mas alla de esto ya no es "pullback", es otra cosa -- descartar
MAX_WAIT_CONFIRM_DAYS = 3         # dias maximos esperando que confirme el rebote antes de cancelar

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def send_email(cfg, subject, body):
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = cfg["gmail_address"]
    msg["To"] = cfg["destination_email"]
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
        server.login(cfg["gmail_address"], cfg["gmail_app_password"])
        server.sendmail(cfg["gmail_address"], [cfg["destination_email"]], msg.as_string())
    print(f"Mail enviado: {subject}")


def fetch_index_tickers(url, symbol_col="Symbol"):
    r = requests.get(url, headers=HEADERS, timeout=20)
    tables = pd.read_html(StringIO(r.text))
    for t in tables:
        if symbol_col in t.columns:
            return [str(s).replace(".", "-") for s in t[symbol_col].tolist()]
    raise ValueError(f"No se encontro una tabla con la columna '{symbol_col}' en {url}")


def download_closes(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, group_by="ticker",
                        threads=True, progress=False, auto_adjust=True)
    closes = {}
    for t in tickers:
        try:
            c = data[t]["Close"] if len(tickers) > 1 else data["Close"]
            c = c.dropna()
            if len(c) > 50:
                closes[t] = c
        except Exception:
            pass
    return pd.DataFrame(closes).sort_index()


def find_base_tests(values, resistance):
    """Devuelve la lista de testeos de la resistencia dentro de 'values': un
    dia 'testea' la resistencia si su cierre esta a <=BASE_TEST_TOLERANCE_PCT
    de ella; dias que tocan la resistencia cerca uno del otro (separados por
    BASE_PEAK_WINDOW dias o menos) se agrupan en un mismo testeo."""
    threshold = resistance * (1 - BASE_TEST_TOLERANCE_PCT / 100.0)
    touching = [i for i, v in enumerate(values) if v >= threshold]
    if not touching:
        return []
    tests = [[touching[0]]]
    for i in touching[1:]:
        if i - tests[-1][-1] <= BASE_PEAK_WINDOW:
            tests[-1].append(i)
        else:
            tests.append([i])
    return tests


def _valid_base_pattern(series):
    """Nucleo compartido: analiza la ventana hasta AYER (sin incluir hoy,
    el nivel de resistencia siempre se calcula con el dato mas actual
    posible, nunca atrasado). Si forma una base valida -- al menos
    BASE_MIN_TESTS testeos de la misma resistencia, cada retroceso entre
    testeos mas chico o igual que el anterior -- devuelve
    (resistencia, dias_habiles_desde_el_ultimo_testeo). Si no, (None, None)."""
    needed = BASE_LOOKBACK_DAYS + 2 * BASE_PEAK_WINDOW + 1
    if len(series) < needed:
        return None, None

    window = series.iloc[-BASE_LOOKBACK_DAYS - 1: -1]  # la base, sin el dia de hoy
    values = window.values
    resistance = values.max()
    if resistance <= 0:
        return None, None

    tests = find_base_tests(values, resistance)
    if len(tests) < BASE_MIN_TESTS:
        return None, None

    pullback_depths = []
    for k in range(len(tests) - 1):
        end_this = tests[k][-1]
        start_next = tests[k + 1][0]
        between = values[end_this: start_next + 1]
        if len(between) == 0:
            continue
        peak_price = values[tests[k][-1]]
        trough_price = between.min()
        if peak_price <= 0:
            continue
        pullback_depths.append((peak_price - trough_price) / peak_price * 100.0)

    if len(pullback_depths) < BASE_MIN_TESTS - 1:
        return None, None

    tightening = all(
        pullback_depths[i] <= pullback_depths[i - 1] * BASE_TIGHTENING_SLACK
        for i in range(1, len(pullback_depths))
    )
    if not tightening:
        return None, None

    last_test_idx = tests[-1][-1]
    days_since_last_test = (len(values) - 1) - last_test_idx  # dias habiles desde el ultimo testeo hasta ayer
    return resistance, days_since_last_test


def base_ready_pivot(series):
    """Base valida, el ultimo testeo ya se "asento" (paso al menos
    BASE_RECENCY_BUFFER_DAYS desde que toco la resistencia, para no avisar
    justo el dia que podria romper) y TODAVIA no rompio. Devuelve el pivote
    o None."""
    resistance, days_since_last_test = _valid_base_pattern(series)
    if resistance is None or days_since_last_test < BASE_RECENCY_BUFFER_DAYS:
        return None
    today_close = series.iloc[-1]
    if today_close > resistance:
        return None  # ya rompio -- eso lo marca base_pivot_breakout, no esta funcion
    return resistance


def base_pivot_breakout(series):
    """Patron de base + ruptura de pivote (Minervini): base valida y el cierre
    de HOY ya rompio por encima de la resistencia (sin exigir el colchon de
    dias -- una ruptura real vale aunque no hubieramos alcanzado a avisar
    antes). Devuelve el pivote (NO el cierre de hoy -- el pivote es el precio
    de entrada real si ya tenias la orden de stop-buy puesta) o None."""
    resistance, _ = _valid_base_pattern(series)
    if resistance is None:
        return None
    today_close = series.iloc[-1]
    if today_close <= resistance:
        return None
    return resistance


def dip_watch_level(series):
    """Sistema paralelo de reversion corta: si en los ultimos REV_LOOKBACK
    dias cayo al menos ENTRY_THRESHOLD_PCT (sin ser excesivo, MAX_DROP_PCT),
    devuelve el cierre de HOY como nivel de referencia para vigilar manana --
    si el precio lo supera, confirma el rebote. Se llama solo el primer dia
    del pullback; mientras se espera la confirmacion, el nivel se actualiza
    cada dia al cierre del dia anterior (ver main())."""
    if len(series) < REV_LOOKBACK + 1:
        return None
    ret_pct = (series.iloc[-1] / series.iloc[-1 - REV_LOOKBACK] - 1.0) * 100.0
    if ret_pct > ENTRY_THRESHOLD_PCT or ret_pct <= MAX_DROP_PCT:
        return None
    return float(series.iloc[-1])


def dip_reversal_confirmed(series, ref_level):
    """Si el cierre de HOY supera el nivel de referencia (el cierre de ayer,
    actualizado cada dia de espera), el rebote se confirma. Devuelve el
    precio de entrada (el nivel de referencia -- no el cierre de hoy, mismo
    criterio que el pivote: simula una orden puesta un poco arriba de ese
    nivel) o None."""
    today_close = series.iloc[-1]
    if today_close > ref_level:
        return ref_level
    return None


def calc_rs_raw_score(sub, idx=-1):
    """Puntaje crudo de fuerza relativa, formula O'Neil/IBD: el ultimo trimestre
    pesa el doble que cada uno de los 3 trimestres anteriores. Se aplica a todas
    las columnas de 'sub' (un DataFrame de cierres) de una, vectorizado."""
    q = RS_QUARTER_DAYS
    try:
        p0 = sub.iloc[idx]
        p1 = sub.iloc[idx - q]
        p2 = sub.iloc[idx - 2 * q]
        p3 = sub.iloc[idx - 3 * q]
        p4 = sub.iloc[idx - 4 * q]
    except IndexError:
        return pd.Series(dtype=float)
    valid = p0.notna() & p1.notna() & p2.notna() & p3.notna() & p4.notna() & (p1 > 0) & (p2 > 0) & (p3 > 0) & (p4 > 0)
    score = 2 * (p0[valid] / p1[valid]) + (p1[valid] / p2[valid]) + (p2[valid] / p3[valid]) + (p3[valid] / p4[valid])
    return score


def rs_percentile_ranks(raw_scores):
    """Convierte puntajes crudos en percentil 1-99, estilo IBD RS Rating."""
    if len(raw_scores) == 0:
        return pd.Series(dtype=float)
    pct = raw_scores.rank(pct=True) * 98 + 1
    return pct.round().astype(int)


def recompute_leaders(today):
    """Rankea Nasdaq-100 + S&P500 por Relative Strength Rating (O'Neil/IBD),
    union deduplicada top-20 c/u. Devuelve (lideres, rs_ratings, casi_califican)
    -- lo ultimo son simbolos con RS>=RS_MIN_RATING que no entraron al top-20
    pero estan a 1-2 criterios del Trend Template completo."""
    nasdaq100 = fetch_index_tickers("https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies", symbol_col="Ticker")
    sp500 = fetch_index_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", symbol_col="Symbol")
    universe = sorted(set(nasdaq100) | set(sp500))

    start = (today - timedelta(days=RS_QUARTER_DAYS * 4 * 2)).strftime("%Y-%m-%d")
    end = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    closes = download_closes(universe, start, end)

    raw = calc_rs_raw_score(closes)
    rs_rating = rs_percentile_ranks(raw)  # percentil contra TODO el universo (nasdaq100 + sp500 juntos)

    def top_n(tickers_subset):
        cols = [t for t in tickers_subset if t in rs_rating.index]
        if not cols:
            return []
        return list(rs_rating[cols].sort_values(ascending=False).head(TOPN_PER_UNIVERSE).index)

    combined = list(dict.fromkeys(top_n(nasdaq100) + top_n(sp500)))
    ratings = {s: int(rs_rating[s]) for s in combined if s in rs_rating.index}

    near_misses = find_near_misses(closes, rs_rating)
    near_misses = [nm for nm in near_misses if nm[0] not in ratings]  # no repetir los que ya son lideres

    return combined, ratings, near_misses


def trend_template_diagnostics(series):
    """Nucleo compartido del Trend Template: devuelve (paso_todo: bool,
    criterios_que_fallan: list[str]). (None, None) si no hay suficiente
    historia. No mira el RS Rating -- eso se aplica aparte."""
    needed = TREND_LOOKBACK_DAYS + MA200_SLOPE_LOOKBACK
    if len(series) < needed:
        return None, None

    ma50 = series.rolling(50).mean()
    ma150 = series.rolling(150).mean()
    ma200 = series.rolling(200).mean()

    price = series.iloc[-1]
    ma50_now, ma150_now, ma200_now = ma50.iloc[-1], ma150.iloc[-1], ma200.iloc[-1]
    ma200_past = ma200.iloc[-1 - MA200_SLOPE_LOOKBACK]
    if pd.isna(ma50_now) or pd.isna(ma150_now) or pd.isna(ma200_now) or pd.isna(ma200_past):
        return None, None

    window = series.iloc[-TREND_LOOKBACK_DAYS:]
    low_52w, high_52w = window.min(), window.max()

    checks = [
        ("precio > MA150", price > ma150_now),
        ("precio > MA200", price > ma200_now),
        ("MA150 > MA200", ma150_now > ma200_now),
        ("MA200 ascendente", ma200_now > ma200_past),
        ("MA50 > MA150", ma50_now > ma150_now),
        ("MA50 > MA200", ma50_now > ma200_now),
        ("precio > MA50", price > ma50_now),
        ("precio >= 25% sobre minimo 52 semanas", price >= low_52w * 1.25),
        ("precio dentro del 25% del maximo 52 semanas", price >= high_52w * 0.75),
    ]
    fails = [name for name, ok in checks if not ok]
    return (len(fails) == 0), fails


def trend_template_ok(series, rs_rating=None):
    """Trend Template de Minervini: los 9 criterios de precio/medias moviles
    + (si se provee) el Relative Strength Rating >= RS_MIN_RATING.
    Todos deben cumplirse -- no es un puntaje, es pasa/no pasa."""
    if rs_rating is not None and rs_rating < RS_MIN_RATING:
        return False
    passed, _ = trend_template_diagnostics(series)
    return bool(passed)


NEAR_MISS_MAX_FAILS = 2  # hasta esta cantidad de criterios fallando cuenta como "casi califica"


def find_near_misses(universe_closes, rs_rating):
    """Entre TODOS los simbolos con RS >= RS_MIN_RATING (no solo el top-20 por
    indice), busca los que fallan el Trend Template por poco (<=NEAR_MISS_MAX_FAILS
    criterios). Devuelve lista de (simbolo, rs, criterios_que_fallan), ordenada
    por menos fallas primero y despues por RS mas alto."""
    results = []
    for sym in rs_rating.index:
        rs = int(rs_rating[sym])
        if rs < RS_MIN_RATING or sym not in universe_closes.columns:
            continue
        series = universe_closes[sym].dropna()
        passed, fails = trend_template_diagnostics(series)
        if passed is None or passed:
            continue
        if len(fails) <= NEAR_MISS_MAX_FAILS:
            results.append((sym, rs, fails))
    return sorted(results, key=lambda x: (len(x[2]), -x[1]))


def main():
    # config.json es opcional: si existe y tiene credenciales de Gmail, manda el
    # mail el mismo por SMTP (uso local/manual). Si no, solo imprime el resultado
    # por stdout -- pensado para la rutina en la nube, que lee esta salida y
    # manda el mail usando el conector de Gmail ya autorizado (sin contraseñas).
    cfg = load_json(CONFIG_PATH, {})

    state = load_json(STATE_PATH, {
        "last_rebalanced_month": None,
        "leaders": {},   # symbol -> {"rs_rating": int|None}
        "watching": {},   # symbol -> {"pivot": float}  (base lista, esperando ruptura)
        "positions": {},  # symbol -> {"entry_date": "YYYY-MM-DD", "entry_price": float}
    })
    watching = state.setdefault("watching", {})

    today = datetime.utcnow().date()
    month_key = today.strftime("%Y-%m")

    actions = []  # lineas para el mail

    if state["last_rebalanced_month"] != month_key:
        print(f"Rebalanceo mensual: {month_key}")
        new_leaders, rs_ratings, near_misses = recompute_leaders(today)
        old_leaders = set(state["leaders"].keys())
        state["leaders"] = {s: {"rs_rating": rs_ratings.get(s)} for s in new_leaders}
        state["last_rebalanced_month"] = month_key
        dropped = old_leaders - set(new_leaders)
        added = set(new_leaders) - old_leaders
        if dropped or added:
            actions.append(f"Rebalanceo mensual ({month_key}): entran {sorted(added)}, salen de la lista {sorted(dropped)} (si tenes posicion abierta en una que sale, se mantiene hasta su salida normal).")
        if near_misses:
            lines = [f"  {sym} (RS={rs}): falta -> {', '.join(fails)}" for sym, rs, fails in near_misses[:15]]
            actions.append("CASI CALIFICAN (RS fuerte, Trend Template a 1-2 criterios de completarse, no estan en la lista de lideres):\n" + "\n".join(lines))

    all_symbols = sorted(set(state["leaders"].keys()) | set(state["positions"].keys()))
    start = (today - timedelta(days=450)).strftime("%Y-%m-%d")  # margen comodo arriba de los ~273 dias habiles que pide el Trend Template
    end = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    closes = download_closes(all_symbols, start, end)

    capital = cfg.get("capital_asignado", 0)
    n_leaders = max(len(state["leaders"]), 1)
    monto_por_posicion = capital / n_leaders if capital else None

    # 1) posiciones abiertas: stop de precio primero, despues salida por tiempo
    for sym, pos in list(state["positions"].items()):
        if sym not in closes.columns:
            continue
        price = closes[sym].iloc[-1]

        entry_price = pos.get("entry_price")
        if entry_price:
            stop_level = entry_price * (1 - STOP_LOSS_PCT / 100.0)
            if price <= stop_level:
                actions.append(f"VENDER {sym} -- STOP DE PRECIO: cayo {STOP_LOSS_PCT:.0f}% desde la entrada (${entry_price:.2f} -> ${price:.2f}).")
                del state["positions"][sym]
                continue

        entry_date = datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()
        bars_held = (closes.index.date >= entry_date).sum() - 1  # dias habiles desde la entrada (excluye el dia de entrada)
        if bars_held >= MAX_HOLD_DAYS:
            actions.append(f"VENDER {sym} -- cumplio {MAX_HOLD_DAYS} dias habiles desde la entrada ({pos['entry_date']}). Precio ref: ${price:.2f}")
            del state["positions"][sym]

    # 2) lideres sin posicion abierta: dos sistemas en paralelo --
    #    (a) base + ruptura de pivote (Minervini, aviso "BASE LISTA" antes de romper)
    #    (b) reversion corta de 5 dias (aviso "CAIDA DETECTADA", confirma si supera el cierre de ayer)
    for sym, lstate in state["leaders"].items():
        if sym in state["positions"] or sym not in closes.columns:
            continue
        series = closes[sym].dropna()
        if len(series) < TREND_LOOKBACK_DAYS + MA200_SLOPE_LOOKBACK:
            continue

        rs_rating = lstate.get("rs_rating")
        rs_txt = f", RS={rs_rating}" if rs_rating is not None else ""
        if not trend_template_ok(series, rs_rating):
            if sym in watching:
                actions.append(f"CANCELAR orden de {sym} -- ya no cumple el Trend Template.")
                del watching[sym]
            continue

        monto_txt = f" (~${monto_por_posicion:,.0f})" if monto_por_posicion else ""
        wstate = watching.get(sym)

        if wstate is not None and wstate["kind"] == "dip":
            wstate["wait_days"] = wstate.get("wait_days", 0) + 1
            ref_level = float(series.iloc[-2])
            confirmed_price = dip_reversal_confirmed(series, ref_level)
            if confirmed_price is not None:
                state["positions"][sym] = {"entry_date": today.strftime("%Y-%m-%d"), "entry_price": float(confirmed_price)}
                actions.append(f"RUPTURA CONFIRMADA (rebote) {sym} -- si pusiste el stop-buy en ${confirmed_price:.2f}, ya deberias estar adentro{monto_txt} (stop de perdida a ${confirmed_price * (1 - STOP_LOSS_PCT / 100.0):.2f}{rs_txt}).")
                del watching[sym]
            elif wstate["wait_days"] >= MAX_WAIT_CONFIRM_DAYS:
                actions.append(f"CANCELAR orden de {sym} -- paso el plazo de {MAX_WAIT_CONFIRM_DAYS} dias sin confirmar el rebote.")
                del watching[sym]
            continue

        if wstate is not None and wstate["kind"] == "base":
            broke = base_pivot_breakout(series)
            if broke is not None:
                price = broke
                state["positions"][sym] = {"entry_date": today.strftime("%Y-%m-%d"), "entry_price": float(price)}
                actions.append(f"RUPTURA CONFIRMADA (base) {sym} -- si pusiste el stop-buy en ${price:.2f}, ya deberias estar adentro{monto_txt} (stop de perdida a ${price * (1 - STOP_LOSS_PCT / 100.0):.2f}{rs_txt}).")
                del watching[sym]
                continue
            ready = base_ready_pivot(series)
            if ready is None:
                actions.append(f"CANCELAR orden de {sym} -- la base ya no es valida (se rompio el patron de achicamiento).")
                del watching[sym]
                continue
            old_pivot = wstate["pivot"]
            if old_pivot > 0 and abs(ready - old_pivot) / old_pivot > 0.01:
                actions.append(f"ACTUALIZAR orden de {sym} -- el pivote se movio de ${old_pivot:.2f} a ${ready:.2f}, ajusta tu stop-buy.")
                wstate["pivot"] = float(ready)
            continue

        # sym no esta en observacion todavia: evaluar las dos seniales desde cero
        broke = base_pivot_breakout(series)
        if broke is not None:
            price = broke
            state["positions"][sym] = {"entry_date": today.strftime("%Y-%m-%d"), "entry_price": float(price)}
            actions.append(f"COMPRAR {sym} -- rompio el pivote de una base sin aviso previo (lider nuevo este mes){monto_txt}. Poner stop-buy en ${price:.2f} (stop de perdida a ${price * (1 - STOP_LOSS_PCT / 100.0):.2f}{rs_txt})")
            continue

        ready = base_ready_pivot(series)
        if ready is not None:
            watching[sym] = {"kind": "base", "pivot": float(ready)}
            actions.append(f"BASE LISTA {sym} -- poner stop-buy en ${ready:.2f}{rs_txt} (stop de perdida planeado a ${ready * (1 - STOP_LOSS_PCT / 100.0):.2f} una vez adentro).")
            continue

        dip_ref = dip_watch_level(series)
        if dip_ref is not None:
            watching[sym] = {"kind": "dip", "wait_days": 0}
            actions.append(f"CAIDA DETECTADA {sym} -- posible pullback dentro de la tendencia{rs_txt}. Vigilando hasta {MAX_WAIT_CONFIRM_DAYS} dias: si el cierre de un dia supera el cierre del dia anterior, confirma el rebote y hay que comprar (stop de perdida al 8% desde ese precio).")

    save_json(STATE_PATH, state)

    if actions:
        body = "\n\n".join(actions) + "\n\n-- Motor de señal Momentum+Reversion (Nasdaq-100 + S&P500) --"
        subject = f"[Alertas Trading] {len(actions)} accion(es) para hoy {today}"
        if cfg.get("gmail_address") and cfg.get("gmail_app_password") and cfg.get("destination_email"):
            send_email(cfg, subject, body)
        else:
            print("===EMAIL_SUBJECT===")
            print(subject)
            print("===EMAIL_BODY===")
            print(body)
            print("===EMAIL_END===")
    else:
        print("Sin acciones hoy.")


if __name__ == "__main__":
    main()
