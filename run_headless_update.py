#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json

# Reconfigure stdout/stderr to UTF-8 on Windows to prevent UnicodeEncodeError
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding='utf-8')
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set base path to import local modules
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)

from tinvest.storage_manager import StorageManager
from tinvest.vietstock_client import VietstockClient
from tinvest.data_loader import enrich_dataframe
from AICcode import (
    analyze_ticker_worker,
    analyze_batch_worker,
    CUSTOM_RULES,
    check_rsi_bullish_divergence,
    check_macd_bullish_divergence,
    check_macd_hist_bullish_divergence,
    check_accumulation_breakout
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(base_path, "headless_update.log"), encoding='utf-8')
    ]
)
logger = logging.getLogger("HeadlessUpdate")

def run_sync_and_update():
    logger.info("==================================================")
    logger.info("🚀 BẮT ĐẦU CẬP NHẬT DỮ LIỆU TỰ ĐỘNG (HEADLESS)")
    logger.info("==================================================")
    
    # 1. Initialize Storage & Client
    storage = StorageManager()
    client = VietstockClient()
    
    # 2. Check Session Status
    logger.info("[*] Đang kiểm tra phiên làm việc với Vietstock...")
    status = client.check_session_status()
    logger.info(f"[*] Trạng thái phiên làm việc: {status}")
    
    if status in ["LIMITED", "ERROR", "NO_DATA"]:
        logger.warning("⚠️ Phiên làm việc bị hạn chế hoặc lỗi. Đang kích hoạt Selenium để làm mới...")
        refreshed = client.config_mgr.refresh_token()
        if refreshed:
            logger.info("✅ Đã làm mới token thành công! Thiết lập lại client...")
            client.refresh_from_config()
        else:
            logger.error("❌ Không thể làm mới token bằng Selenium.")
            # We still proceed with the existing credentials/bypass small paging as a fallback
    
    # 3. Detect Missing Dates (SSoT)
    last_date = storage.get_last_date()
    logger.info(f"[*] Ngày cuối cùng có dữ liệu trong storage: {last_date}")
    
    missing_dates = client.get_missing_dates(last_date)
    
    # Integrity check: Last 3 trading days
    check_dates = []
    current = last_date or datetime.now()
    while len(check_dates) < 3 and current is not None:
        if current.weekday() < 5:
            check_dates.append(current.strftime("%Y-%m-%d"))
        current -= timedelta(days=1)
        
    if check_dates:
        logger.info(f"[*] Đang quét tính toàn vẹn 3 ngày gần nhất: {', '.join(check_dates)}...")
        ticker_counts = storage.get_ticker_counts_for_dates(check_dates)
        bad_dates = [d for d, count in ticker_counts.items() if count > 0 and count < 1200]
        if bad_dates:
            logger.warning(f"⚠️ Phát hiện {len(bad_dates)} ngày bị thiếu mã (< 1200 mã): {', '.join(bad_dates)}")
            logger.info("[*] Đang xóa dữ liệu lỗi để tải lại...")
            storage.delete_specific_dates(bad_dates)
            missing_dates = sorted(list(set(missing_dates) | set(bad_dates)))

    # Force update current trading day
    now = datetime.now()
    effective_today = now.date()
    if now.weekday() == 5: effective_today -= timedelta(days=1)
    elif now.weekday() == 6: effective_today -= timedelta(days=2)
    eff_today_str = effective_today.strftime("%Y-%m-%d")
    
    if eff_today_str not in missing_dates:
        missing_dates.append(eff_today_str)
        missing_dates = sorted(missing_dates)
        
    logger.info(f"[*] Danh sách các ngày cần tải dữ liệu: {', '.join(missing_dates)}")
    
    affected_tickers = set()
    
    # 4. Ingest Price Data
    for i, d in enumerate(missing_dates):
        day_total = []
        logger.info(f"\n--- [Ngày {i+1}/{len(missing_dates)}] TẢI DỮ LIỆU: {d} ---")
        
        # HOSE=1, HNX=2, UPCOM=3
        for cat_id, cat_name in [(1, "HSX"), (2, "HNX"), (3, "UPCOM")]:
            try:
                logger.info(f"   [+] Đang nạp sàn {cat_name}...")
                raw, is_limited = client.fetch_market_day(cat_id, d)
                if raw:
                    day_total.extend(raw)
                    logger.info(f"   ---> ✅ Đã tải: {len(raw)} mã {cat_name}")
            except Exception as e:
                logger.error(f"   ! Lỗi tải sàn {cat_name} ngày {d}: {e}")
                
        if day_total:
            total_raw = len(day_total)
            df_day = client.format_to_df(day_total)
            
            # Skip if total rows is too low
            if total_raw < 1200:
                logger.error(f"❌ HỦY BỎ ngày {d}: Chỉ có {total_raw} mã (Yêu cầu >= 1200)")
                continue
                
            # Stagnant Bluechips check
            if 'MarketCap' in df_day.columns:
                top50 = df_day.sort_values('MarketCap', ascending=False).head(50)
                is_stagnant_top50 = (top50['Open'] == top50['High']) & \
                                    (top50['Open'] == top50['Low']) & \
                                    (top50['Open'] == top50['Close'])
                if is_stagnant_top50.all():
                    logger.error(f"❌ HỦY BỎ ngày {d}: Phát hiện 50 mã Bluechips đều đứng im.")
                    continue
                    
            logger.info(f"   [DONE] Kiểm tra toàn vẹn OK. Lưu dữ liệu...")
            
            # Update Active Registry on the last day
            if d == missing_dates[-1]:
                all_tickers = df_day['Ticker'].unique().tolist()
                storage.save_active_registry(all_tickers)
                logger.info(f"   [*] Đã cập nhật Registry: {len(all_tickers)} mã niêm yết.")
                
            # Sync stock prices
            for idx, (ticker, group) in enumerate(df_day.groupby("Ticker")):
                try:
                    t_min = storage.sync_prices(ticker, group, source='API')
                    if t_min is not None:
                        affected_tickers.add(ticker)
                except Exception as ex:
                    pass
                    
        # Fetch Indices (VNINDEX=1, HNX-INDEX=2)
        indices = [("VNINDEX", 1, -19), ("HNX-INDEX", 2, -18)]
        for ticker, tid, sid in indices:
            try:
                idx_raw = client.fetch_index_day(ticker, tid, sid, d)
                if idx_raw:
                    day_idx = client.format_to_df(idx_raw)
                    storage.sync_prices(ticker, day_idx, source='API')
                    affected_tickers.add(ticker)
                    logger.info(f"   ---> Xong Index: {ticker} ({d})")
            except Exception as e:
                logger.error(f"   ! Lỗi Index {ticker}: {e}")
                
    # 5. Determine which tickers need recalculation
    if not affected_tickers:
        logger.info("ℹ️ Dữ liệu giá hiện tại đã khớp 100%. Đang tính toán cho toàn bộ mã trong Registry...")
        current_reg = storage.get_active_registry() or []
        affected_tickers = set(current_reg)
        
    logger.info(f"--- ĐANG TÍNH TOÁN CHỈ BÁO VÀ PHÂN TÍCH CHO {len(affected_tickers)} MÃ ---")
    
    data_dict = {}
    analysis_cache = {}
    items_to_recompute = []
    
    # Load historical data for affected tickers
    for t in affected_tickers:
        df_full = storage.load_ticker_data(t)
        if df_full is not None:
            data_dict[t] = df_full
            items_to_recompute.append((t, df_full))
            
    total = len(items_to_recompute)
    if total > 0:
        batch_size = 10
        batches = [items_to_recompute[i:i + batch_size] for i in range(0, total, batch_size)]
        num_workers = min((os.cpu_count() or 4) * 2, 16)
        
        logger.info(f"[*] Khởi chạy ThreadPoolExecutor với {num_workers} workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(analyze_batch_worker, b) for b in batches]
            completed_count = 0
            for future in as_completed(futures):
                batch_results = future.result()
                for ticker, res in batch_results:
                    if res:
                        analysis_cache[ticker] = res
                        if 'df' in res:
                            data_dict[ticker] = res['df']
                        storage.save_indicators(ticker, res['df'])
                        storage.save_analysis(ticker, res)
                completed_count += len(batch_results)
                if completed_count % 200 == 0 or completed_count == total:
                    logger.info(f"      ... Tiến độ: {completed_count}/{total} mã...")
                    
    # 6. Post-Integrity Check (Hậu Kiểm)
    missing_after = []
    for t in affected_tickers:
        df_check = data_dict.get(t)
        if df_check is not None and ('HK_NW' not in df_check.columns or 'T2_SMA' not in df_check.columns):
            missing_after.append(t)
            
    if missing_after:
        logger.warning(f"⚠️ HẬU KIỂM: Phát hiện {len(missing_after)} mã thiếu Trending. Đang xử lý bù...")
        for t in missing_after:
            try:
                df_final = enrich_dataframe(data_dict[t])
                data_dict[t] = df_final
                storage.save_indicators(t, df_final)
            except Exception as ex:
                pass
        logger.info("✅ Hậu kiểm hoàn tất.")
    else:
        logger.info("✅ Tuyệt vời! 100% mã đã đầy đủ chỉ số Trending.")

    # 7. Calculate Market Breadth (Time-series)
    logger.info("📊 Đang tính toán dữ liệu Độ rộng Thị trường...")
    market_breadth_data = compute_market_breadth(data_dict)
    
    # 8. Filter Tickers & Build Output Structure
    logger.info("🔍 Đang tổng hợp các bộ lọc và luật tùy chỉnh...")
    tickers_analysis = []
    
    categories_meta = {
        "ACCUMULATION": "Tích lũy",
        "PERFECT_MA": "Perfect MA (Xu hướng tăng mạnh)",
        "HEIKIN_BUY": "Heikin Buy (Tín hiệu mua Heikin Ashi)",
        "UPCLOUD": "UpCloud (Xu hướng tăng trên mây)",
        "WHITE_ADX": "ADX Trắng (Đầu chu kỳ xu hướng)",
        "EARLY": "Điểm mua EARLY (Mua sớm)",
        "ADD_1": "Điểm mua gia tăng 1 (ADD_1)",
        "ADD_2": "Điểm mua gia tăng 2 (ADD_2)",
        "STRONG": "Điểm mua MẠNH (STRONG)"
    }
    
    rules_meta = {k: v["label"] for k, v in CUSTOM_RULES.items()}
    
    # Pre-compiled list of tickers for simple category views
    filtered_results = {cat: [] for cat in categories_meta.keys()}
    for rule_key in rules_meta.keys():
        filtered_results[rule_key] = []
        
    for ticker, data in list(analysis_cache.items()):
        df = data.get("df")
        if df is None or df.empty:
            continue
            
        # Get price indicators
        current_vol = int(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0
        if current_vol < 20000:  # Allow low volumes in JSON but filter inside dashboard
            continue
            
        res = data.get("adv") or {}
        accum = data.get("accum") or {}
        ma_trend = data.get("ma_trend") or {}
        val = data.get("valuation") or {}
        
        current_p = float(df['Close'].iloc[-1]) * 1000
        ep = val.get("price", 0)
        tp = val.get("tp1", 0)
        sl = val.get("cutloss_partial", 0)
        rr_ratio = val.get("rr_ratio", 0)
        val_score = val.get("risk_score", 0)
        risk_pct = val.get("risk_pct", 0)
        action = val.get("action", "WAIT")
        
        # Evaluate matched categories
        matched_categories = []
        
        if accum.get("is_accumulation", False):
            matched_categories.append("ACCUMULATION")
            
        if ma_trend.get("is_perfect_uptrend", False):
            matched_categories.append("PERFECT_MA")
            
        # Heikin Buy
        buy_2 = False
        if 'HK_BuySignal' in df.columns or 'HK_BuyManh' in df.columns:
            buy_2 = df.get('HK_BuySignal', pd.Series([False])).tail(2).any() or df.get('HK_BuyManh', pd.Series([False])).tail(2).any()
        if buy_2:
            matched_categories.append("HEIKIN_BUY")
            
        # UpCloud
        if len(df) > 0 and 'High' in df.columns and 'Low' in df.columns:
            last = df.iloc[-1]
            current_price = last['Close']
            span_a = last.get('SpanA', 0)
            span_b = last.get('SpanB', 0)
            tenkan = last.get('Tenkan', 0)
            kijun = last.get('Kijun', 0)
            ma10 = last.get('MA10', 0)
            ma20 = last.get('MA20', 0)
            
            future_span_a = (tenkan + kijun) / 2
            h52 = df['High'].iloc[-52:].max() if len(df) >= 52 else df['High'].max()
            l52 = df['Low'].iloc[-52:].min() if len(df) >= 52 else df['Low'].min()
            future_span_b = (h52 + l52) / 2
            
            c1 = (current_price > span_a) and (current_price > span_b) if span_a > 0 else False
            c2 = (future_span_a > future_span_b)
            c3 = (tenkan > kijun)
            c4 = (ma10 > ma20)
            if c1 and c2 and c3 and c4:
                matched_categories.append("UPCLOUD")
                
        # White ADX
        adx_color = str(df['ADX_Color'].iloc[-1]).upper() if 'ADX_Color' in df.columns else "N/A"
        if adx_color == "WHITE":
            matched_categories.append("WHITE_ADX")
            
        # Entry Type (EARLY, ADD_1, ADD_2, STRONG)
        entry_type = res.get("entry_type")
        if entry_type in ["EARLY", "ADD_1", "ADD_2", "STRONG"]:
            matched_categories.append(entry_type)
            
        # Evaluate matched custom rules
        matched_rules = []
        for rule_key, r_def in CUSTOM_RULES.items():
            try:
                if r_def["func"](df):
                    matched_rules.append(rule_key)
            except Exception:
                pass
                
        # Create ticker record
        ticker_record = {
            "Ticker": ticker,
            "Price": int(current_p),
            "Volume": int(current_vol),
            "Entry": int(ep * 1000) if ep > 0 else None,
            "Target": int(tp * 1000) if tp > 0 else None,
            "StopLoss": int(sl * 1000) if sl > 0 else None,
            "RR": f"{round(rr_ratio, 1)}/1" if rr_ratio > 0 else "N/A",
            "RiskScore": int(val_score),
            "RiskPct": float(risk_pct),
            "Action": action,
            "Categories": matched_categories,
            "Rules": matched_rules
        }
        
        tickers_analysis.append(ticker_record)
        
        # Populate pre-compiled categories/rules lists
        for cat in matched_categories:
            filtered_results[cat].append(ticker)
        for rule_key in matched_rules:
            filtered_results[rule_key].append(ticker)

    # 9. Output to JSON file
    output_dir = os.path.join(base_path, "Output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "analysis_results.json")
    
    # ICT current time format
    from datetime import timezone
    ict_time = datetime.now(timezone.utc) + timedelta(hours=7)
    last_update_str = ict_time.strftime("%Y-%m-%d %H:%M:%S")
    
    final_output = {
        "last_update": last_update_str,
        "market_breadth": market_breadth_data,
        "categories_meta": categories_meta,
        "rules_meta": rules_meta,
        "tickers_analysis": tickers_analysis,
        "filtered_results": filtered_results
    }
    
    logger.info(f"[*] Đang xuất file kết quả ra: {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
        
    logger.info(f"✅ HOÀN TẤT CẬP NHẬT! Đã xuất {len(tickers_analysis)} mã cổ phiếu.")
    logger.info("==================================================")

def compute_market_breadth(data_dict):
    """Ported market breadth computation from TinvestApp._update_breadth_from_cache."""
    if len(data_dict) < 5:
        logger.warning("⚠️ Không đủ dữ liệu để tính độ rộng (cần ít nhất 5 mã).")
        return {}
        
    # Get reference dates from VNINDEX
    vn_key = next((k for k in data_dict.keys() if "VNINDEX" in k), "VNINDEX")
    idx_df = data_dict.get(vn_key)
    if idx_df is None or idx_df.empty:
        logger.warning("⚠️ Không tìm thấy VNINDEX để làm mốc thời gian.")
        return {}
        
    all_dates = pd.to_datetime(idx_df['Date']).sort_values().unique()
    ref_date = all_dates[-1]
    
    breadth_dfs = []
    processed_count = 0
    
    for ticker, df_sub in data_dict.items():
        if ticker in ["VNINDEX", "HNXINDEX", "UPCOM", "VN30", "HNX30", "HAINDEX", "UPCOM-INDEX", "HNX-INDEX"]:
            continue
        if len(df_sub) < 50:
            continue
            
        try:
            if df_sub is None or df_sub.empty:
                continue
                
            # Skip if delisted/suspended > 30 days
            last_ticker_date = pd.to_datetime(df_sub['Date'].iloc[-1])
            if (ref_date - last_ticker_date).days > 30:
                continue
                
            temp = pd.DataFrame()
            temp['Date'] = pd.to_datetime(df_sub['Date'])
            temp = temp.drop_duplicates(subset=['Date'])
            
            active_mask = (df_sub['Volume'] > 0) & (df_sub['Close'] > 0)
            ma10 = df_sub['MA10'] if 'MA10' in df_sub.columns else df_sub['Close'].rolling(10).mean()
            ma20 = df_sub['MA20'] if 'MA20' in df_sub.columns else df_sub['Close'].rolling(20).mean()
            ma50 = df_sub['MA50'] if 'MA50' in df_sub.columns else df_sub['Close'].rolling(50).mean()
            
            temp['Valid'] = active_mask.astype(int)
            temp['>MA10'] = ((df_sub['Close'] > ma10) & active_mask).astype(int)
            temp['>MA20'] = ((df_sub['Close'] > ma20) & active_mask).astype(int)
            temp['>MA50'] = ((df_sub['Close'] > ma50) & active_mask).astype(int)
            
            temp = temp.set_index('Date')
            temp = temp[temp['Valid'] == 1]
            temp = temp.reindex(all_dates).ffill(limit=30).reset_index()
            temp['Valid'] = temp['Valid'].fillna(0)
            
            breadth_dfs.append(temp)
            processed_count += 1
        except Exception as e:
            pass
            
    if breadth_dfs:
        all_breadth = pd.concat(breadth_dfs)
        grouped = all_breadth.groupby('Date').sum()
        valid_counts = grouped['Valid'].replace(0, 1)
        
        mb = pd.DataFrame()
        mb['%MA10'] = (grouped['>MA10'] / valid_counts) * 100
        mb['%MA20'] = (grouped['>MA20'] / valid_counts) * 100
        mb['%MA50'] = (grouped['>MA50'] / valid_counts) * 100
        mb = mb.sort_index()
        
        logger.info(f"✅ Tính xong Độ rộng Độ rộng từ {processed_count} mã cổ phiếu.")
        return {
            "dates": [d.strftime("%Y-%m-%d") for d in mb.index],
            "MA10": mb['%MA10'].round(2).tolist(),
            "MA20": mb['%MA20'].round(2).tolist(),
            "MA50": mb['%MA50'].round(2).tolist()
        }
    return {}

if __name__ == "__main__":
    run_sync_and_update()
