#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Configure matplotlib backend safely
import matplotlib
if 'tkinter' not in sys.modules:
    try:
        matplotlib.use('Agg')
    except Exception as e:
        logger.warning(f"Could not set Agg backend: {e}")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.ticker as ticker_lib

def export_greenpink_chart(ticker, df_full, df_vn, save_path):
    """
    Generate and save the GreenPink & Octopus chart for a ticker (e.g. VNINDEX).
    """
    try:
        plt.style.use('dark_background')
        df_full = df_full.copy()
        
        # --- CALCULATE RS13 & RS52 ---
        if 'RS13' not in df_full.columns or 'RS52' not in df_full.columns:
            if df_vn is not None and not df_vn.empty:
                df_vn_indexed = df_vn.set_index('Date')
                bench_close = df_full['Date'].map(df_vn_indexed['Close']).ffill().bfill()
                rs_raw = df_full['Close'] / (bench_close + 1e-10)
                
                # RS52: 52 weeks = 260 bars
                rs52_min = rs_raw.rolling(window=260, min_periods=1).min()
                rs52_max = rs_raw.rolling(window=260, min_periods=1).max()
                df_full['RS52'] = 100 * (rs_raw - rs52_min) / (rs52_max - rs52_min + 0.0001)
                
                # RS13: 13 weeks = 65 bars
                rs13_min = rs_raw.rolling(window=65, min_periods=1).min()
                rs13_max = rs_raw.rolling(window=65, min_periods=1).max()
                df_full['RS13'] = 100 * (rs_raw - rs13_min) / (rs13_max - rs13_min + 0.0001)
            else:
                df_full['RS13'] = 50.0
                df_full['RS52'] = 50.0

        # 1. Prepare Data (Last 150 bars)
        count = 150
        df = df_full.tail(count).copy().reset_index(drop=True)
        x_idx = np.arange(len(df))
        
        # 2. Setup Figure with 3 subplots
        fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
        fig.patch.set_facecolor('black') 
        ax.set_facecolor('black')
        ax2.set_facecolor('black')
        ax3.set_facecolor('black')
        
        # --- TOP SUBPLOT: GREENPINK ---
        # 3. Plot Cloud (E14 vs E21)
        e14 = df['GP_E14']
        e21 = df['GP_E21']
        c = df['Close']
        green_mask = (c > e14) & (c > e21)
        pink_mask = ~green_mask
        ax.fill_between(x_idx, e14, e21, where=green_mask, color='#00FF00', alpha=0.3, interpolate=True, linewidth=0)
        ax.fill_between(x_idx, e14, e21, where=pink_mask, color='#FF69B4', alpha=0.3, interpolate=True, linewidth=0)

        # 4. Plot xFast and xSlow
        ax.plot(x_idx, df['GP_xFast'], color='lime', linewidth=2.5, label='xFast (Green)')
        ax.plot(x_idx, df['GP_xSlow'], color='red', linewidth=2.5, label='xSlow (Red)')

        # 5. Plot Bollinger Bands on xSlow
        ax.plot(x_idx, df['GP_BB_Top'], color='blue', linewidth=1.2, alpha=0.8, label='BB Top (xSlow)')
        ax.plot(x_idx, df['GP_BB_Bot'], color='blue', linewidth=1.2, alpha=0.8, label='BB Bot (xSlow)')
        ax.fill_between(x_idx, df['GP_BB_Bot'], df['GP_BB_Top'], color='blue', alpha=0.1)

        # 6. Plot Candlesticks
        close_val = df['Close']
        open_val = df['Open']
        high_val = df['High']
        low_val = df['Low']
        up_mask = close_val >= open_val
        down_mask = ~up_mask
        if up_mask.any():
            ax.vlines(x_idx[up_mask], low_val[up_mask], high_val[up_mask], color='#00FF00', linewidth=1.0)
            ax.bar(x_idx[up_mask], close_val[up_mask] - open_val[up_mask], bottom=open_val[up_mask], color='#00FF00', width=0.6)
        if down_mask.any():
            ax.vlines(x_idx[down_mask], low_val[down_mask], high_val[down_mask], color='#FF0000', linewidth=1.0)
            ax.bar(x_idx[down_mask], open_val[down_mask] - close_val[down_mask], bottom=close_val[down_mask], color='#FF0000', width=0.6)

        # --- BOTTOM SUBPLOT: OCTOPUS (MACD MCGINLEY) ---
        ax2.plot(x_idx, df['OCT_A1'], color='white', linewidth=0.8, alpha=0.3)
        
        # Plot A1 and B1 (Mirror) with dynamic color dots/line
        oct_colors = df['OCT_Color'].iloc[1:].tolist()
        
        a1_np = df['OCT_A1'].to_numpy()
        points_a1 = np.array([x_idx, a1_np]).T.reshape(-1, 1, 2)
        segments_a1 = np.concatenate([points_a1[:-1], points_a1[1:]], axis=1)
        lc_a1 = LineCollection(segments_a1, colors=oct_colors, linewidths=2.5)
        ax2.add_collection(lc_a1)
        
        b1_np = df['OCT_B1'].to_numpy()
        points_b1 = np.array([x_idx, b1_np]).T.reshape(-1, 1, 2)
        segments_b1 = np.concatenate([points_b1[:-1], points_b1[1:]], axis=1)
        lc_b1 = LineCollection(segments_b1, colors=oct_colors, linewidths=2.5)
        ax2.add_collection(lc_b1)

        # Plot Bollinger Bands Cloud on A1
        ax2.plot(x_idx, df['OCT_BB_Top'], color='#00008B', linewidth=1.0, linestyle='--', alpha=0.6)
        ax2.plot(x_idx, df['OCT_BB_Bot'], color='#00008B', linewidth=1.0, linestyle='--', alpha=0.6)
        ax2.fill_between(x_idx, df['OCT_BB_Bot'], df['OCT_BB_Top'], color='#ADD8E6', alpha=0.2, label='Octopus Band')
        ax2.axhline(0, color='white', linewidth=0.5, alpha=0.5)

        # --- THIRD SUBPLOT: RS CHART ---
        ax3.plot(x_idx, df['RS13'], color='white', linewidth=2.0, label='RS13')
        ax3.plot(x_idx, df['RS52'], color='yellow', linewidth=2.0, label='RS52')
        ax3.axhline(50, color='red', linewidth=0.8, linestyle='--', alpha=0.5)

        # 7. Formatting
        ax.set_title(f"GP & OCTOPUS CHART (HHV-LLV + McGinley): {ticker}", color='gold', fontsize=15, fontweight='bold', pad=12)
        ax.set_ylabel("Price", color='white', fontweight='bold')
        ax2.set_ylabel("Octopus MACD", color='white', fontweight='bold')
        ax3.set_ylabel("RS Rating", color='white', fontweight='bold')
        
        for axis in [ax, ax2, ax3]:
            axis.grid(True, color='#222222', linestyle=':', alpha=0.5)
            axis.tick_params(colors='white')
            for spine in axis.spines.values():
                spine.set_color('#444444')
        
        # Format X-axis dates
        df['Date'] = pd.to_datetime(df['Date'])
        date_labels = df['Date'].dt.strftime('%d/%m/%y').tolist()
        ax3.xaxis.set_major_formatter(ticker_lib.FuncFormatter(lambda x, pos: date_labels[int(round(x))] if 0 <= int(round(x)) < len(date_labels) else ""))
        
        ax.legend(loc='lower left', facecolor='black', edgecolor='#00FF00', labelcolor='white', fontsize=8)
        ax2.legend(loc='lower left', facecolor='black', edgecolor='#FF69B4', labelcolor='white', fontsize=8)
        ax3.legend(loc='lower left', facecolor='black', edgecolor='yellow', labelcolor='white', fontsize=8)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor='black', edgecolor='none', dpi=120)
        plt.close(fig)
        logger.info(f"✅ Exported GreenPink chart to {save_path}")
    except Exception as e:
        logger.error(f"Error exporting GreenPink chart for {ticker}: {e}")
        import traceback
        traceback.print_exc()

def export_heikin_chart(ticker, df_full, save_path):
    """
    Generate and save the Heikin-Ashi & 2Trend chart for a ticker (e.g. VNINDEX).
    """
    try:
        plt.style.use('dark_background')
        df = df_full.tail(150).copy().reset_index(drop=True)
        x_idx = np.arange(len(df))
        
        # Setup Figure with 2 subplots
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 11), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
        fig.patch.set_facecolor('black') 
        ax.set_facecolor('black')
        ax2.set_facecolor('black')
        
        # --- TOP SUBPLOT: HEIKIN & TREND COLOR ---
        # Plot Hull MA Cloud
        mh = df['HK_MHull']
        sh = df['HK_SHull']
        ax.fill_between(x_idx, mh, sh, where=(mh > sh), color='lime', alpha=0.1)
        ax.fill_between(x_idx, mh, sh, where=(mh <= sh), color='red', alpha=0.1)

        # Trend Color Line (EMA 13)
        tc_trend = df['TC_Trend']
        tc_t_color = df['TC_TrendColor'].fillna('#434651')
        tc_trend_np = tc_trend.to_numpy()
        points_tc = np.array([x_idx, tc_trend_np]).T.reshape(-1, 1, 2)
        segments_tc = np.concatenate([points_tc[:-1], points_tc[1:]], axis=1)
        tc_colors = tc_t_color.iloc[1:].tolist()
        lc_tc = LineCollection(segments_tc, colors=tc_colors, linewidths=2.5, alpha=0.9)
        ax.add_collection(lc_tc)
        
        # Stop Line (ATR Stop)
        tc_stop = df['TC_StopLine']
        tc_s_color = df['TC_StopColor'].fillna('#434651')
        ax.scatter(x_idx, tc_stop, c=tc_s_color, s=10, marker='_')

        # Plot NW Trailing Stop
        nw = df['HK_NW']
        trend = df['HK_Trend']
        nw_np = nw.to_numpy()
        points_nw = np.array([x_idx, nw_np]).T.reshape(-1, 1, 2)
        segments_nw = np.concatenate([points_nw[:-1], points_nw[1:]], axis=1)
        nw_colors = ['#00FF00' if trend.iloc[i] == 1 else '#FF0000' for i in range(1, len(df))]
        lc_nw = LineCollection(segments_nw, colors=nw_colors, linewidths=2)
        ax.add_collection(lc_nw)

        # Plot Smoothed Heikin Ashi Candles
        ho, hh, hl, hc = df['HK_Flower_Open'], df['HK_Flower_High'], df['HK_Flower_Low'], df['HK_Flower_Close']
        bar_colors = df['HK_BarColor']
        color_map = {'brightGreen': '#00FF00', 'red': '#FF0000', 'white': '#FFFFFF'}
        for color_name, color_hex in color_map.items():
            mask = bar_colors == color_name
            if mask.any():
                ax.vlines(x_idx[mask], hl[mask], hh[mask], color=color_hex, linewidth=1)
                ax.bar(x_idx[mask], abs(hc[mask] - ho[mask]) + 0.001, bottom=np.minimum(ho[mask], hc[mask]), color=color_hex, width=0.6, alpha=0.8)

        # Plot Signal Shapes
        buys = df[df['HK_BuySignal'] | df['HK_BuyManh']]
        sells = df[df['HK_SellSignal'] | df['HK_SellManh']]
        if not buys.empty:
            ax.plot(buys.index, buys['HK_Flower_Low'] * 0.985, '^', markersize=10, color='lime', markeredgecolor='white')
        if not sells.empty:
            ax.plot(sells.index, sells['HK_Flower_High'] * 1.015, 'v', markersize=10, color='red', markeredgecolor='white')

        # --- BOTTOM SUBPLOT: NORMAL CANDLES & 2TREND ---
        # Plot Normal Candlesticks
        o, h, l, c_val = df['Open'], df['High'], df['Low'], df['Close']
        up_mask = c_val >= o
        down_mask = ~up_mask
        if up_mask.any():
            ax2.vlines(x_idx[up_mask], l[up_mask], h[up_mask], color='#00FF00', linewidth=1)
            ax2.bar(x_idx[up_mask], abs(c_val[up_mask] - o[up_mask]) + 0.001, bottom=np.minimum(o[up_mask], c_val[up_mask]), color='#00FF00', width=0.6)
        if down_mask.any():
            ax2.vlines(x_idx[down_mask], l[down_mask], h[down_mask], color='#FF0000', linewidth=1)
            ax2.bar(x_idx[down_mask], abs(c_val[down_mask] - o[down_mask]) + 0.001, bottom=np.minimum(o[down_mask], c_val[down_mask]), color='#FF0000', width=0.6)

        # Plot 2Trend SMA
        t2_sma = df['T2_SMA']
        t2_trend = df['T2_SMA_Trend']
        t2_sma_np = t2_sma.to_numpy()
        points_t2 = np.array([x_idx, t2_sma_np]).T.reshape(-1, 1, 2)
        segments_t2 = np.concatenate([points_t2[:-1], points_t2[1:]], axis=1)
        t2_colors = ['#00ffaa' if t2_trend.iloc[i] == 1 else '#ff0000' for i in range(1, len(df))]
        lc_t2 = LineCollection(segments_t2, colors=t2_colors, linewidths=3)
        ax2.add_collection(lc_t2)

        # Plot 2Trend Supertrend Bands
        st_upper = df['T2_ST_Upper']
        st_lower = df['T2_ST_Lower']
        st_trend = df['T2_ST_Trend']
        mid = (o + c_val) / 2
        ax2.fill_between(x_idx, mid, st_lower, where=(st_trend == 1), color='#00ffaa', alpha=0.2)
        ax2.fill_between(x_idx, mid, st_upper, where=(st_trend == -1), color='#ff0000', alpha=0.2)
        
        # Signals for 2Trend
        t2_sma_shift = df['T2_SMA_Trend'].shift(1).fillna(0)
        buys2 = df[(df['T2_SMA_Trend'] == 1) & (t2_sma_shift <= 0)]
        sells2 = df[(df['T2_SMA_Trend'] == -1) & (t2_sma_shift >= 0)]
        
        if not buys2.empty:
            for idx in buys2.index:
                ax2.text(idx, df['Low'].iloc[idx]*0.97, "𝑳", color='#00ffaa', fontsize=12, fontweight='bold', ha='center')
        if not sells2.empty:
            for idx in sells2.index:
                ax2.text(idx, df['High'].iloc[idx]*1.03, "𝑺", color='#ff0000', fontsize=12, fontweight='bold', ha='center')

        # Formatting
        df['Date'] = pd.to_datetime(df['Date'])
        last_date = df['Date'].iloc[-1].strftime('%d/%m/%Y') if 'Date' in df.columns else "N/A"
        ax.set_title(f"Chart trend color - {ticker} - {last_date}", color='gold', fontsize=16, fontweight='bold', pad=15)
        ax2.set_title(f"Normal Candles & 2Trend Logic", color='gold', fontsize=14, fontweight='bold')
        
        for a in [ax, ax2]:
            a.set_ylabel("Price", color='white', fontweight='bold')
            a.grid(True, color='#222222', linestyle=':', alpha=0.5)
            a.tick_params(colors='white')
            for spine in a.spines.values():
                spine.set_color('#444444')
        
        date_labels = df['Date'].dt.strftime('%d/%m/%y').tolist()
        ax2.xaxis.set_major_formatter(ticker_lib.FuncFormatter(lambda x, pos: date_labels[int(round(x))] if 0 <= int(round(x)) < len(date_labels) else ""))
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor='black', edgecolor='none', dpi=120)
        plt.close(fig)
        logger.info(f"✅ Exported Heikin chart to {save_path}")
    except Exception as e:
        logger.error(f"Error exporting Heikin chart for {ticker}: {e}")
        import traceback
        traceback.print_exc()
