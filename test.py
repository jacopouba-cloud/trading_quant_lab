
#MODULO 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("🏆 REPORT FINALE MODULO 1: IL MODELLO 'DIAMANTE' (2021-2026) 🏆")

# Parametri Vincenti
PROX = 4.5; CAP = 50.0; PB = 0.15; SL_P = 0.60; VOL = 2.0; TTL = "08:45"
PIP_VALUE = 0.0001

df_final = df.copy()
df_final['Date_Only'] = df_final.index.date
trades_log = []

# --- ESECUZIONE BACKTEST DETTAGLIATO ---
for date, day_group in df_final.groupby('Date_Only'):
    asia = day_group.between_time('00:00', '07:59')
    if len(asia) < 120: continue
    ah, al = asia['high'].max(), asia['low'].min()
    ar = (ah - al) / PIP_VALUE
    if ar > CAP or ar < 10: continue
    
    l_open = day_group.between_time('08:00', '08:04')
    if l_open.empty: continue
    p08 = l_open['open'].iloc[0]
    
    pre_vol = abs(p08 - day_group.between_time('07:55', '07:59')['open'].iloc[0]) / PIP_VALUE
    if pre_vol > VOL: continue
    
    asia_v = "BULLISH" if asia['close'].iloc[-1] > asia['open'].iloc[0] else "BEARISH"
    pos = ((p08 - al) / (ah - al)) * 100
    bias = "LONG" if (pos >= 66.6 and asia_v == "BULLISH") else "SHORT" if (pos <= 33.3 and asia_v == "BEARISH") else None
    
    if bias:
        dist = (ah - p08)/PIP_VALUE if bias == "LONG" else (p08 - al)/PIP_VALUE
        if dist <= PROX:
            entry = p08 - (ar * PB * PIP_VALUE) if bias == "LONG" else p08 + (ar * PB * PIP_VALUE)
            sl_price = entry - (ar * SL_P * PIP_VALUE) if bias == "LONG" else entry + (ar * SL_P * PIP_VALUE)
            tp_price = ah if bias == "LONG" else al
            
            fill = day_group.between_time('08:00', TTL)
            is_filled = not (fill[fill['low'] <= entry] if bias == "LONG" else fill[fill['high'] >= entry]).empty
            
            if is_filled:
                f_idx = (fill[fill['low'] <= entry] if bias == "LONG" else fill[fill['high'] >= entry]).index[0]
                post_f = day_group.loc[f_idx:].between_time(f_idx.strftime('%H:%M'), '13:59')
                
                if bias == "LONG":
                    t_hit = post_f[post_f['high'] >= tp_price].index[0] if not post_f[post_f['high'] >= tp_price].empty else None
                    s_hit = post_f[post_f['low'] <= sl_price].index[0] if not post_f[post_f['low'] <= sl_price].empty else None
                else:
                    t_hit = post_f[post_f['low'] <= tp_price].index[0] if not post_f[post_f['low'] <= tp_price].empty else None
                    s_hit = post_f[post_f['high'] >= sl_price].index[0] if not post_f[post_f['high'] >= sl_price].empty else None
                
                if t_hit and (not s_hit or t_hit < s_hit):
                    res = abs(tp_price - entry)/PIP_VALUE
                elif s_hit:
                    res = -abs(entry - sl_price)/PIP_VALUE
                else: continue # Trade non concluso entro le 14:00
                
                trades_log.append({'Date': pd.to_datetime(date), 'PnL': res})

df_results = pd.DataFrame(trades_log).set_index('Date')
df_results['Cum_PnL'] = df_results['PnL'].cumsum()
df_results['Year'] = df_results.index.year
df_results['Semester'] = np.where(df_results.index.month <= 6, 'S1', 'S2')

# --- GRUPPO SEMESTRALE ---
sem_stats = df_results.groupby(['Year', 'Semester'])['PnL'].agg([
    ('Trades', 'count'),
    ('PnL_Pips', 'sum'),
    ('WR%', lambda x: (x > 0).mean() * 100)
])

# Calcolo Max DD per semestre
def get_max_dd(series):
    cum = series.cumsum()
    return (cum.cummax() - cum).max()

sem_stats['Max_DD'] = df_results.groupby(['Year', 'Semester'])['PnL'].apply(get_max_dd)

print("\n" + "="*85)
print(f"{'Periodo':<12} | {'Trades':<8} | {'WR%':<10} | {'PnL (Pips)':<12} | {'Max DD'}")
print("-" * 85)
for idx, row in sem_stats.iterrows():
    period = f"{idx[0]} {idx[1]}"
    print(f"{period:<12} | {int(row['Trades']):<8} | {row['WR%']:<10.1f} | {row['PnL_Pips']:<12.1f} | {row['Max_DD']:.1f}")
print("="*85)

# --- PLOT EQUITY CURVE ---
plt.figure(figsize=(12, 6))
plt.plot(df_results['Cum_PnL'], color='#00ff88', linewidth=2)
plt.title('EQUITY CURVE DEFINITIVA: EURUSD MODULO 1 (2021-2026)', fontsize=14, color='white')
plt.ylabel('Pips Cumulati', color='white')
plt.grid(alpha=0.2)
plt.gca().set_facecolor('#1e1e1e')
plt.gcf().set_facecolor('#1e1e1e')
plt.tick_params(colors='white')
plt.show()



# MODULO 2


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("🏆 FASE 35: FINAL GOLDEN SETUP - THE JUDAS REVERSAL (2021-2026) 🏆")

# Parametri d'oro
EXT_PCT = 0.075
SL_DIST = 8.5
TP_PCT = 0.10
MAX_MINS = 15
PIP_VALUE = 0.0001

df_final = df.copy()
df_final['Date_Only'] = df_final.index.date
trades_log = []

for date, day_group in df_final.groupby('Date_Only'):
    asia = day_group.between_time('00:00', '07:59')
    if len(asia) < 120: continue
    ah, al = asia['high'].max(), asia['low'].min()
    ar = (ah - al) / PIP_VALUE
    if ar < 10: continue

    london = day_group.between_time('08:00', '13:59')
    if london.empty: continue

    break_h = london[london['high'] > ah]
    break_l = london[london['low'] < al]

    side = None
    if not break_h.empty and (break_l.empty or break_h.index[0] < break_l.index[0]):
        side = "HIGH"; start_t = break_h.index[0]; level = ah
    elif not break_l.empty:
        side = "LOW"; start_t = break_l.index[0]; level = al

    if side is None: continue

    # Calcolo distanza entrata
    entry_dist_pips = ar * EXT_PCT
    if entry_dist_pips >= SL_DIST: continue

    post_break = london.loc[start_t:]

    if side == "HIGH":
        entry = level + (entry_dist_pips * PIP_VALUE)
        sl    = level + (SL_DIST * PIP_VALUE)
        tp    = level - (ar * TP_PCT * PIP_VALUE)

        fill_check = post_break[post_break['high'] >= entry]
        if fill_check.empty: continue
        fill_t = fill_check.index[0]

        # FILTRO TEMPORALE (TTL)
        mins_from_break = (fill_t - start_t).total_seconds() / 60
        if mins_from_break > MAX_MINS: continue

        active = post_break.loc[fill_t:]
        dummy = active.index[-1] + pd.Timedelta(days=1)

        t_sl = active[active['high'] >= sl].index[0] if not active[active['high'] >= sl].empty else dummy
        t_tp = active[active['low'] <= tp].index[0] if not active[active['low'] <= tp].empty else dummy

    else:
        entry = level - (entry_dist_pips * PIP_VALUE)
        sl    = level - (SL_DIST * PIP_VALUE)
        tp    = level + (ar * TP_PCT * PIP_VALUE)

        fill_check = post_break[post_break['low'] <= entry]
        if fill_check.empty: continue
        fill_t = fill_check.index[0]

        # FILTRO TEMPORALE (TTL)
        mins_from_break = (fill_t - start_t).total_seconds() / 60
        if mins_from_break > MAX_MINS: continue

        active = post_break.loc[fill_t:]
        dummy = active.index[-1] + pd.Timedelta(days=1)

        t_sl = active[active['low'] <= sl].index[0] if not active[active['low'] <= sl].empty else dummy
        t_tp = active[active['high'] >= tp].index[0] if not active[active['high'] >= tp].empty else dummy

    # Calcolo Risultati
    risk_pips = abs(entry - sl) / PIP_VALUE
    gain_pips = abs(entry - tp) / PIP_VALUE
    rr_trade = gain_pips / risk_pips if risk_pips > 0 else 0

    if t_tp < t_sl and t_tp != dummy:
        trades_log.append({'Date': pd.to_datetime(date), 'PnL': gain_pips, 'Type': 'Win', 'RR': rr_trade})
    elif t_sl < t_tp and t_sl != dummy:
        trades_log.append({'Date': pd.to_datetime(date), 'PnL': -risk_pips, 'Type': 'Loss', 'RR': rr_trade})

# --- ELABORAZIONE DATI E REPORT ---
df_res = pd.DataFrame(trades_log).set_index('Date')
df_res['Cum_PnL'] = df_res['PnL'].cumsum()
df_res['Quarter'] = df_res.index.to_period('Q')

def get_max_dd(series):
    cum = series.cumsum()
    return (cum.cummax() - cum).max()

q_stats = df_res.groupby('Quarter').agg(
    Trades=('PnL', 'count'),
    PnL_Pips=('PnL', 'sum'),
    Wins=('Type', lambda x: (x == 'Win').sum())
)
q_stats['WR%'] = (q_stats['Wins'] / q_stats['Trades']) * 100
q_stats['Max_DD'] = df_res.groupby('Quarter')['PnL'].apply(get_max_dd)

print("\n" + "="*85)
print(f"{'Trimestre':<10} | {'Trades':<8} | {'WR%':<8} | {'PnL (Pips)':<12} | {'Max DD'}")
print("-" * 85)
for q, row in q_stats.iterrows():
    print(f"{str(q):<10} | {int(row['Trades']):<8} | {row['WR%']:<8.1f} | {row['PnL_Pips']:<12.1f} | {row['Max_DD']:.1f}")
print("="*85)

total_pips = df_res['PnL'].sum()
global_max_dd = get_max_dd(df_res['PnL'])
global_wr = (len(df_res[df_res['Type'] == 'Win']) / len(df_res)) * 100
avg_rr = df_res['RR'].mean()

print(f"\n📊 STATISTICHE GLOBALI MODULO 2 (2021-2026):")
print(f"   Totale PnL:        {total_pips:.1f} Pips")
print(f"   Win Rate Globale:  {global_wr:.1f}%")
print(f"   Max DD Globale:    {global_max_dd:.1f} Pips")
print(f"   Avg RR:            {avg_rr:.2f}")

# --- PLOT EQUITY CURVE ---
plt.figure(figsize=(12, 6))
plt.plot(df_res.index, df_res['Cum_PnL'], color='#00ffcc', linewidth=2)
plt.title('EQUITY CURVE DEFINITIVA: EURUSD MODULO 2 (Golden Setup 2021-2026)', fontsize=14, color='white')
plt.ylabel('Pips Cumulati', color='white')
plt.grid(alpha=0.2)
plt.gca().set_facecolor('#1e1e1e')
plt.gcf().set_facecolor('#1e1e1e')
plt.tick_params(colors='white')
plt.show()


# MODULO 3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("🏆 FASE 56: IL MOTORE DEFINITIVO MODULO 3 (The Structural King) 🏆")

# Settings Finali e Robusti
MIN_DROP = 10.0
MAX_DROP = 40.0
TP_PIPS = 15.0
SL_PIPS = 20.0
TOUCH_END_HOUR = 15 # Accetta fino alle 15:59

df_m3 = df.copy()
df_m3['Date_Only'] = df_m3.index.date
PIP_VALUE = 0.0001
vol_col = 'volume' if 'volume' in df_m3.columns else ('tick_volume' if 'tick_volume' in df_m3.columns else None)

trades_log = []

for date, day_group in df_m3.groupby('Date_Only'):
    # Calcolo VWAP Giornaliero
    typical = (day_group['high'] + day_group['low'] + day_group['close']) / 3
    if vol_col:
        volume = day_group[vol_col]
        vwap = (typical * volume).cumsum() / volume.cumsum()
    else:
        vwap = typical.expanding().mean()
    day_group = day_group.assign(VWAP=vwap)
    
    # Snapshot ore 14:00
    pre_ny = day_group.between_time('13:55', '14:05')
    if pre_ny.empty: continue
    
    vwap_14 = pre_ny['VWAP'].iloc[-1]
    price_14 = pre_ny['close'].iloc[-1]
    dist_14 = (price_14 - vwap_14) / PIP_VALUE
    
    # FILTRO STRUTTURALE UNIVERSALE (10-40 pips)
    if not (MIN_DROP <= abs(dist_14) <= MAX_DROP):
        continue
        
    state = "ABOVE" if dist_14 > 0 else "BELOW"
    
    # FINESTRA DEL TOCCO (Golden Window)
    tw = day_group.between_time('14:00', f"{TOUCH_END_HOUR}:59")
    if tw.empty: continue
    
    t_idx = None
    if state == "ABOVE":
        touches = tw[tw['low'] <= tw['VWAP']]
        if not touches.empty: t_idx = touches.index[0]
    else:
        touches = tw[tw['high'] >= tw['VWAP']]
        if not touches.empty: t_idx = touches.index[0]
            
    if not t_idx: continue
        
    # ESECUZIONE TRADE (Senza gestione attiva per evitare Over-Management)
    entry_px = tw.loc[t_idx, 'VWAP']
    df_trade = day_group.loc[t_idx : f"{date} 19:59:59"]
    
    pnl = 0
    trade_closed = False
    
    for idx, row in df_trade.iterrows():
        h_dist = (row['high'] - entry_px) / PIP_VALUE if state == "ABOVE" else (entry_px - row['low']) / PIP_VALUE
        l_dist = (row['low'] - entry_px) / PIP_VALUE if state == "ABOVE" else (entry_px - row['high']) / PIP_VALUE
        
        if l_dist <= -SL_PIPS:
            pnl = -SL_PIPS
            trade_closed = True
            break
        if h_dist >= TP_PIPS:
            pnl = TP_PIPS
            trade_closed = True
            break
            
    if not trade_closed:
        # Chiusura a tempo alle 20:00
        close_px = df_trade['close'].iloc[-1]
        pnl = (close_px - entry_px) / PIP_VALUE if state == "ABOVE" else (entry_px - close_px) / PIP_VALUE
        
    trades_log.append({
        'Date': pd.to_datetime(date),
        'PnL': pnl,
        'Type': 'Win' if pnl > 0 else 'Loss' if pnl < 0 else 'Tie'
    })

# --- ELABORAZIONE E REPORTING ---
df_res = pd.DataFrame(trades_log).set_index('Date')
df_res['Cum_PnL'] = df_res['PnL'].cumsum()
df_res['Quarter'] = df_res.index.to_period('Q')

def get_max_dd(series):
    if series.empty: return 0
    cum = series.cumsum()
    return (cum.cummax() - cum).max()
    
q_stats = df_res.groupby('Quarter').agg(
    Trades=('PnL', 'count'),
    PnL_Pips=('PnL', 'sum'),
    Wins=('Type', lambda x: (x == 'Win').sum()),
)
q_stats['WR%'] = (q_stats['Wins'] / q_stats['Trades']) * 100
q_stats['Max_DD'] = df_res.groupby('Quarter')['PnL'].apply(get_max_dd)

print("\n" + "="*85)
print(f"{'Trimestre':<10} | {'Trades':<8} | {'WR% (>0)':<10} | {'PnL (Pips)':<12} | {'Max DD'}")
print("-" * 85)
for q, row in q_stats.iterrows():
    print(f"{str(q):<10} | {int(row['Trades']):<8} | {row['WR%']:<10.1f} | {row['PnL_Pips']:<12.1f} | {row['Max_DD']:.1f}")
print("="*85)

tot_pnl = df_res['PnL'].sum()
global_dd = get_max_dd(df_res['PnL'])
global_wr = (len(df_res[df_res['PnL'] > 0]) / len(df_res)) * 100

print(f"\n🌍 STATISTICHE GLOBALI MODULO 3 (The Structural King):")
print(f"   Operazioni Totali: {len(df_res)}")
print(f"   Win Rate Globale:  {global_wr:.1f}%")
print(f"   Profitto Totale:   {tot_pnl:.1f} Pips 💰")
print(f"   Drawdown Massimo:  {global_dd:.1f} Pips 🛡️")
print(f"   Recovery Factor:   {tot_pnl/global_dd:.2f} 👑")

# Grafico Equity Curve
plt.figure(figsize=(14, 7))
plt.plot(df_res.index, df_res['Cum_PnL'], color='#00d4ff', linewidth=3)
plt.fill_between(df_res.index, df_res['Cum_PnL'], df_res['Cum_PnL'].cummax(), color='#ff0055', alpha=0.2)
plt.title('EQUITY CURVE: MODULO 3 - THE STRUCTURAL KING (2021-2026)', fontsize=16, color='white', fontweight='bold')
plt.ylabel('Pips Cumulati', color='white')
plt.grid(alpha=0.1, color='white')
plt.gca().set_facecolor('#121212')
plt.gcf().set_facecolor('#121212')
plt.tick_params(colors='white')
plt.show()


# BACKTEST MODULO UNICO


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("🚀 BACKTEST DEFINITIVO: THE HOLY TRINITY (M1+M2+M3) 🚀")

# --- DATA PREP ---
df_master = df.copy()
df_master['Date_Only'] = df_master.index.date
PIP_VALUE = 0.0001
vol_col = 'volume' if 'volume' in df_master.columns else ('tick_volume' if 'tick_volume' in df_master.columns else None)

all_trades = []

# Funzione per il calcolo corretto del Max Drawdown
def get_real_max_dd(pnl_series):
    if pnl_series.empty: return 0
    cum_pnl = pnl_series.cumsum()
    max_cum = cum_pnl.cummax()
    dd = max_cum - cum_pnl
    return dd.max()

for date, day_group in df_master.groupby('Date_Only'):
    # --- CALCOLO PARAMETRI GIORNALIERI COMUNI ---
    asia = day_group.between_time('00:00', '07:59')
    if len(asia) < 120: continue
    ah, al = asia['high'].max(), asia['low'].min()
    ar_pips = (ah - al) / PIP_VALUE
    
    # Calcolo VWAP per M3
    typical = (day_group['high'] + day_group['low'] + day_group['close']) / 3
    vwap = (typical * day_group[vol_col]).cumsum() / day_group[vol_col].cumsum() if vol_col else typical.expanding().mean()
    day_group = day_group.assign(VWAP=vwap)
    dummy = day_group.index[-1] + pd.Timedelta(days=1)

# ---------------------------------------------------------
    # 💎 MODULO 1: DIAMOND SNIPER (Deep Limit Entry @ 08:00)
    # ---------------------------------------------------------
    if 10 <= ar_pips <= 50.0:
        l_open = day_group.between_time('08:00', '08:04')
        pre_8 = day_group.between_time('07:55', '07:59')
        if not l_open.empty and not pre_8.empty:
            p08 = l_open['open'].iloc[0]
            pre_vol = abs(p08 - pre_8['open'].iloc[0]) / PIP_VALUE
            if pre_vol <= 2.0:
                asia_v = "BULLISH" if asia['close'].iloc[-1] > asia['open'].iloc[0] else "BEARISH"
                pos = ((p08 - al) / (ah - al)) * 100
                bias = "LONG" if (pos >= 66.6 and asia_v == "BULLISH") else "SHORT" if (pos <= 33.3 and asia_v == "BEARISH") else None
                
                if bias:
                    dist_to_level = (ah - p08)/PIP_VALUE if bias == "LONG" else (p08 - al)/PIP_VALUE
                    if dist_to_level <= 4.5:
                        # Entry a sconto (Mantenuto al 15% come validato)
                        m1_entry = p08 - (ar_pips * 0.15 * PIP_VALUE) if bias == "LONG" else p08 + (ar_pips * 0.15 * PIP_VALUE)
                        
                        # Stop Loss invariato (60% AR)
                        m1_sl_px = m1_entry - (ar_pips * 0.60 * PIP_VALUE) if bias == "LONG" else m1_entry + (ar_pips * 0.60 * PIP_VALUE)
                        
                        # 🎯 NUOVO TAKE PROFIT (+10% Asian Range Extension)
                        m1_tp_px = ah + (ar_pips * 0.10 * PIP_VALUE) if bias == "LONG" else al - (ar_pips * 0.10 * PIP_VALUE)
                        
                        m1_fill_window = day_group.between_time('08:00', '08:45')
                        filled = m1_fill_window[m1_fill_window['low'] <= m1_entry] if bias == "LONG" else m1_fill_window[m1_fill_window['high'] >= m1_entry]
                        
                        if not filled.empty:
                            f_idx = filled.index[0]
                            post_f = day_group.loc[f_idx:].between_time(f_idx.strftime('%H:%M'), '13:59')
                            
                            t_tp = post_f[post_f['high'] >= m1_tp_px].index[0] if bias == "LONG" and not post_f[post_f['high'] >= m1_tp_px].empty else \
                                   post_f[post_f['low'] <= m1_tp_px].index[0] if bias == "SHORT" and not post_f[post_f['low'] <= m1_tp_px].empty else dummy
                            t_sl = post_f[post_f['low'] <= m1_sl_px].index[0] if bias == "LONG" and not post_f[post_f['low'] <= m1_sl_px].empty else \
                                   post_f[post_f['high'] >= m1_sl_px].index[0] if bias == "SHORT" and not post_f[post_f['high'] >= m1_sl_px].empty else dummy
                            
                            if t_tp < t_sl and t_tp != dummy:
                                all_trades.append({'Date': pd.to_datetime(date), 'PnL': abs(m1_tp_px - m1_entry)/PIP_VALUE, 'Module': 'M1_Diamond'})
                            elif t_sl < t_tp and t_sl != dummy:
                                all_trades.append({'Date': pd.to_datetime(date), 'PnL': -abs(m1_entry - m1_sl_px)/PIP_VALUE, 'Module': 'M1_Diamond'})

 
    # ---------------------------------------------------------
    # 🕵️ MODULO 2: MODERN JUDAS (Inversione Post-Breakout)
    # ---------------------------------------------------------
    if ar_pips >= 10:
        london = day_group.between_time('08:00', '13:59')
        break_h, break_l = london[london['high'] > ah], london[london['low'] < al]
        
        m2_side = None
        if not break_h.empty and (break_l.empty or break_h.index[0] < break_l.index[0]):
            m2_side = "HIGH"; m2_t0 = break_h.index[0]; m2_lvl = ah
        elif not break_l.empty:
            m2_side = "LOW"; m2_t0 = break_l.index[0]; m2_lvl = al
            
        if m2_side:
            m2_entry_dist = ar_pips * 0.075
            if m2_entry_dist < 8.5:
                m2_entry_px = m2_lvl + (m2_entry_dist * PIP_VALUE) if m2_side == "HIGH" else m2_lvl - (m2_entry_dist * PIP_VALUE)
                m2_test_win = london.loc[m2_t0 : m2_t0 + pd.Timedelta(minutes=15)]
                
                is_m2_triggered = (m2_test_win['high'] >= m2_entry_px).any() if m2_side == "HIGH" else (m2_test_win['low'] <= m2_entry_px).any()
                if is_m2_triggered:
                    m2_tp_px = m2_lvl - (ar_pips * 0.10 * PIP_VALUE) if m2_side == "HIGH" else m2_lvl + (ar_pips * 0.10 * PIP_VALUE)
                    m2_sl_px = m2_lvl + (8.5 * PIP_VALUE) if m2_side == "HIGH" else m2_lvl - (8.5 * PIP_VALUE)
                    
                    post_m2 = london.loc[m2_t0:]
                    t_tp = post_m2[post_m2['low'] <= m2_tp_px].index[0] if m2_side == "HIGH" and not post_m2[post_m2['low'] <= m2_tp_px].empty else \
                           post_m2[post_m2['high'] >= m2_tp_px].index[0] if m2_side == "LOW" and not post_m2[post_m2['high'] >= m2_tp_px].empty else dummy
                    t_sl = post_m2[post_m2['high'] >= m2_sl_px].index[0] if m2_side == "HIGH" and not post_m2[post_m2['high'] >= m2_sl_px].empty else \
                           post_m2[post_m2['low'] <= m2_sl_px].index[0] if m2_side == "LOW" and not post_m2[post_m2['low'] <= m2_sl_px].empty else dummy
                    
                    if t_tp < t_sl and t_tp != dummy:
                        all_trades.append({'Date': pd.to_datetime(date), 'PnL': abs(m2_entry_px - m2_tp_px)/PIP_VALUE, 'Module': 'M2_Judas'})
                    elif t_sl < t_tp and t_sl != dummy:
                        all_trades.append({'Date': pd.to_datetime(date), 'PnL': -abs(m2_entry_px - m2_sl_px)/PIP_VALUE, 'Module': 'M2_Judas'})

    # ---------------------------------------------------------
    # 🦘 MODULO 3: STRUCTURAL KING (New York VWAP)
    # ---------------------------------------------------------
    pre_ny = day_group.between_time('13:55', '14:05')
    if not pre_ny.empty:
        vwap_14 = pre_ny['VWAP'].iloc[-1]
        dist_14 = (pre_ny['close'].iloc[-1] - vwap_14) / PIP_VALUE
        if 10.0 <= abs(dist_14) <= 40.0:
            m3_state = "ABOVE" if dist_14 > 0 else "BELOW"
            tw = day_group.between_time('14:00', '15:59')
            touches = tw[tw['low'] <= tw['VWAP']] if m3_state == "ABOVE" else tw[tw['high'] >= tw['VWAP']]
            
            if not touches.empty:
                t_idx = touches.index[0]
                m3_entry_px = tw.loc[t_idx, 'VWAP']
                post_m3 = day_group.loc[t_idx : f"{date} 19:59:59"]
                m3_tp_px = m3_entry_px + (15.0 * PIP_VALUE) if m3_state == "ABOVE" else m3_entry_px - (15.0 * PIP_VALUE)
                m3_sl_px = m3_entry_px - (20.0 * PIP_VALUE) if m3_state == "ABOVE" else m3_entry_px + (20.0 * PIP_VALUE)
                
                t_tp = post_m3[post_m3['high'] >= m3_tp_px].index[0] if m3_state == "ABOVE" and not post_m3[post_m3['high'] >= m3_tp_px].empty else \
                       post_m3[post_m3['low'] <= m3_tp_px].index[0] if m3_state == "BELOW" and not post_m3[post_m3['low'] <= m3_tp_px].empty else dummy
                t_sl = post_m3[post_m3['low'] <= m3_sl_px].index[0] if m3_state == "ABOVE" and not post_m3[post_m3['low'] <= m3_sl_px].empty else \
                       post_m3[post_m3['high'] >= m3_sl_px].index[0] if m3_state == "BELOW" and not post_m3[post_m3['high'] >= m3_sl_px].empty else dummy
                
                if t_tp < t_sl and t_tp != dummy:
                    all_trades.append({'Date': pd.to_datetime(date), 'PnL': 15.0, 'Module': 'M3_VWAP'})
                elif t_sl < t_tp and t_sl != dummy:
                    all_trades.append({'Date': pd.to_datetime(date), 'PnL': -20.0, 'Module': 'M3_VWAP'})
                else: # Time Exit at market close
                    m3_c_px = post_m3['close'].iloc[-1]
                    all_trades.append({'Date': pd.to_datetime(date), 'PnL': (m3_c_px - m3_entry_px)/PIP_VALUE if m3_state=="ABOVE" else (m3_entry_px - m3_c_px)/PIP_VALUE, 'Module': 'M3_VWAP'})

# --- REPORTING FINALE E CALCOLI CORRETTI ---
df_res = pd.DataFrame(all_trades).set_index('Date').sort_index()
df_res['Cum_PnL'] = df_res['PnL'].cumsum()
df_res['Quarter'] = df_res.index.to_period('Q')

q_stats = df_res.groupby('Quarter').agg(
    Trades=('PnL', 'count'),
    PnL_Pips=('PnL', 'sum'),
    WR=('PnL', lambda x: (x > 0).mean() * 100)
)
# Calcolo Max Drawdown Corretto per ogni Quarter
q_stats['Max_DD'] = df_res.groupby('Quarter')['PnL'].apply(get_real_max_dd)

print("\n" + "="*85)
print(f"{'Trimestre':<12} | {'Trades':<8} | {'Win Rate %':<12} | {'PnL Pips':<12} | {'Max DD'}")
print("-" * 85)
for q, row in q_stats.iterrows():
    print(f"{str(q):<12} | {int(row['Trades']):<8} | {row['WR']:>10.1f}% | {row['PnL_Pips']:>10.1f} p | {row['Max_DD']:.1f}")
print("="*85)

# Metriche Globali Definitivamente Corrette
tot_pnl = df_res['PnL'].sum()
real_global_dd = get_real_max_dd(df_res['PnL'])
recovery_factor = tot_pnl / real_global_dd if real_global_dd > 0 else 0

print(f"\n🌍 PERFORMANCE MASTER PORTFOLIO (HOLY TRINITY):")
print(f"   Profitto Totale:   {tot_pnl:.1f} Pips 💰")
print(f"   Drawdown Massimo:  {real_global_dd:.1f} Pips 🛡️")
print(f"   Recovery Factor:   {recovery_factor:.2f} 👑")

# --- PLOT EQUITY CURVES ---
plt.figure(figsize=(15, 8))
colors = {'M1_Diamond': '#00ff88', 'M2_Judas': '#ff8800', 'M3_VWAP': '#00d4ff'}
labels = {'M1_Diamond': 'M1: Sniper 08:00', 'M2: Judas': 'M2: Judas Reversal', 'M3_VWAP': 'M3: NY VWAP'}

for mod in ['M1_Diamond', 'M2_Judas', 'M3_VWAP']:
    mod_data = df_res[df_res['Module'] == mod]
    if not mod_data.empty:
        plt.plot(mod_data.index, mod_data['PnL'].cumsum(), label=mod, color=colors.get(mod, 'gray'), alpha=0.5, linestyle='--')

plt.plot(df_res.index, df_res['Cum_PnL'], label='MASTER EQUITY (THE TRIAD)', color='#ff00ff', linewidth=4)
plt.fill_between(df_res.index, df_res['Cum_PnL'], df_res['Cum_PnL'].cummax(), color='#ff0055', alpha=0.15)

plt.title('THE HOLY TRINITY: MASTER PORTFOLIO EQUITY CURVE (2021-2026)', fontsize=16, color='white', fontweight='bold')
plt.grid(alpha=0.1, color='white'); plt.gca().set_facecolor('#121212'); plt.gcf().set_facecolor('#121212')
plt.tick_params(colors='white'); plt.legend(facecolor='#121212', labelcolor='white'); plt.tight_layout(); plt.show()




