import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def analyze_portfolio(portfolio_params: dict, tickers_data: list, storage) -> str:
    """
    Phân tích danh mục đầu tư dựa trên bộ công thức quản trị rủi ro.
    
    portfolio_params: {
        'nav_total': float,
        'w_target': float, # % (ví dụ: 100 cho 100%)
        'n_tickers': int,
        'r_cl': float # % cutloss mặc định (ví dụ: 7)
    }
    
    tickers_data: list of dict, each dict: {
        'ticker': str,
        'quantity': float,
        'avg_price': float
    }
    
    storage: StorageManager instance to load data
    """
    
    try:
        cash_on_hand = float(portfolio_params['nav_total'])
        w_target = float(portfolio_params['w_target']) / 100.0
        n_tickers = int(portfolio_params['n_tickers'])
        r_cl = float(portfolio_params['r_cl']) / 100.0
    except Exception as e:
        return f"Lỗi tham số đầu vào danh mục: {e}"

    if n_tickers <= 0:
        return "Số lượng mã phải lớn hơn 0."

    report_lines = []
    report_lines.append("BÁO CÁO ĐÁNH GIÁ DANH MỤC ĐẦU TƯ\n" + "="*50)
    
    total_market_value = 0
    total_cost_value = 0
    
    pre_results = []

    from tinvest.analyzer import analyze_stock
    
    for item in tickers_data:
        ticker = item['ticker'].upper().strip()
        q_i = float(item['quantity'])
        p_avg_input = float(item['avg_price'])
        
        # Load data
        df = storage.load_ticker_data(ticker)
        if df is None or len(df) < 20:
            pre_results.append({'ticker': ticker, 'valid': False, 'msg': f"Không đủ dữ liệu cho {ticker}. Vui lòng update."})
            continue
            
        # Chạy phân tích on-the-fly để luôn có dữ liệu mới nhất
        analysis = analyze_stock(ticker, df)
        if not analysis:
            pre_results.append({'ticker': ticker, 'valid': False, 'msg': f"Không thể phân tích {ticker}."})
            continue
            
        val = analysis.get('valuation', {})
        if not val or not val.get('is_valid'):
            pre_results.append({'ticker': ticker, 'valid': False, 'msg': f"Lỗi Valuation cho {ticker}."})
            continue
            
        # Prices from system are in thousands (e.g. 27.5). Convert to VND.
        p_now_vnd = float(val.get('price', 0)) * 1000
        p_sup_vnd = float(val.get('s1', 0)) * 1000
        if p_sup_vnd == 0: p_sup_vnd = p_now_vnd * 0.95
        p_res_vnd = float(val.get('r1', 0)) * 1000
        if p_res_vnd == 0: p_res_vnd = p_now_vnd * 1.05
        p_ts_vnd = float(val.get('trailing_stop', p_sup_vnd/1000)) * 1000
        
        # Normalize p_avg_input to VND
        p_avg_vnd = p_avg_input * 1000 if p_avg_input < 1000 else p_avg_input
        
        # Determine Trend
        ma_trend = analysis.get('ma_trend', {})
        trend_status = ma_trend.get('trend_status', 'Sideway')
        trend_i = 0
        if "Uptrend" in trend_status: trend_i = 1
        elif "Downtrend" in trend_status: trend_i = -1
        
        market_val = q_i * p_now_vnd
        cost_val = q_i * p_avg_vnd
        
        total_market_value += market_val
        total_cost_value += cost_val

        pre_results.append({
            'ticker': ticker, 'valid': True, 'q': q_i, 'p_avg_vnd': p_avg_vnd, 'p_now_vnd': p_now_vnd,
            'p_sup_vnd': p_sup_vnd, 'p_res_vnd': p_res_vnd, 'p_ts_vnd': p_ts_vnd, 'trend': trend_i, 'trend_desc': trend_status,
            'market_val': market_val, 'cost_val': cost_val, 'df': df, 'val': val, 'analysis': analysis
        })

    # Calculate True NAV
    nav_current = cash_on_hand + total_market_value
    nav_cost = cash_on_hand + total_cost_value
    
    if nav_current <= 0:
        return "Tổng tài sản (Tiền mặt + Cổ phiếu) phải lớn hơn 0."
        
    v_max_i = (nav_current * w_target) / n_tickers
    risk_max_nav = 0.03 * nav_current
    portfolio_risk_total = 0
    results = []

    # 2. Second Pass: Evaluate metrics using correct NAV
    for item in pre_results:
        if not item['valid']:
            results.append(item)
            continue
            
        ticker = item['ticker']
        q_i = item['q']
        p_avg_vnd = item['p_avg_vnd']
        p_now_vnd = item['p_now_vnd']
        p_sup_vnd = item['p_sup_vnd']
        p_res_vnd = item['p_res_vnd']
        p_ts_vnd = item['p_ts_vnd']
        trend_i = item['trend']
        df = item['df']
        
        pl_pct = (p_now_vnd - p_avg_vnd) / p_avg_vnd if p_avg_vnd > 0 else 0
        w_curr = item['market_val'] / nav_current
        
        last_row = df.iloc[-1]
        mcdx_banker = float(last_row.get('Banker', 0)) if 'Banker' in df.columns else 10
        prev_mcdx_banker = float(df.iloc[-2].get('Banker', mcdx_banker)) if len(df) > 1 and 'Banker' in df.columns else mcdx_banker
        adx = float(last_row.get('ADX', 20))
        ha_color = str(last_row.get('HA_Color', 'Green')) if 'HA_Color' in df.columns else 'Green'
        ma20 = float(last_row.get('MA20', p_now_vnd/1000))
        vol = float(last_row.get('Volume', 0))
        vol_avg = float(last_row.get('AvgVolume20', vol))
        
        mcdx_weak = (mcdx_banker < prev_mcdx_banker) and (mcdx_banker < 15)
        adx_low = adx < 20
        heikin_red = (ha_color.lower() == 'red')
        price_below_ma20 = (p_now_vnd/1000) < ma20
        tech_weak = mcdx_weak or adx_low or heikin_red or price_below_ma20
        
        sideways_near_res = False
        if len(df) >= 4:
            recent_highs = df['High'].iloc[-4:].max() * 1000
            recent_lows = df['Low'].iloc[-4:].min() * 1000
            recent_vols = df['Volume'].iloc[-4:].mean()
            if recent_highs >= p_res_vnd * 0.98 and (recent_highs - recent_lows)/recent_lows < 0.05 and recent_vols > vol_avg:
                sideways_near_res = True
                
        # Stoploss logic: System calculated based on technical support
        p_sl_vnd = p_ts_vnd if p_ts_vnd > 0 else p_sup_vnd * 0.97
        sl_source = "Hệ thống tư vấn"

        current_risk_amt = q_i * (p_avg_vnd - p_sl_vnd) if p_avg_vnd > p_sl_vnd else 0
        portfolio_risk_total += current_risk_amt

        results.append({
            'ticker': ticker, 'valid': True, 'q': q_i, 'p_avg_vnd': p_avg_vnd, 'p_now_vnd': p_now_vnd,
            'p_sup_vnd': p_sup_vnd, 'p_res_vnd': p_res_vnd, 'p_ts_vnd': p_ts_vnd, 'trend': trend_i, 'trend_desc': item['trend_desc'],
            'pl_pct': pl_pct, 'w_curr': w_curr, 'm_val': item['market_val'],
            'tech_weak': tech_weak, 'sideways_near_res': sideways_near_res,
            'current_risk': current_risk_amt, 'p_sl_vnd': p_sl_vnd, 'sl_source': sl_source,
            'val': item['val'], 'analysis': item['analysis']
        })

    # 3. Portfolio Level Assessment Output
    report_lines.append("1. ĐÁNH GIÁ CHẤT LƯỢNG TÀI SẢN (TỔNG QUAN)")
    report_lines.append(f"- Tổng Tài Sản Hiện Tại (NAV): {nav_current:,.0f} VND")
    report_lines.append(f"  + Tiền mặt đang có: {cash_on_hand:,.0f} VND ({(cash_on_hand/nav_current)*100:.1f}%)")
    report_lines.append(f"  + Giá trị cổ phiếu: {total_market_value:,.0f} VND ({(total_market_value/nav_current)*100:.1f}%)")
    report_lines.append(f"- Tổng Giá Vốn Cổ Phiếu: {total_cost_value:,.0f} VND")
    
    total_profit = total_market_value - total_cost_value
    total_profit_pct = (total_profit / total_cost_value * 100) if total_cost_value > 0 else 0
    sign = "+" if total_profit > 0 else ""
    report_lines.append(f"- Lợi Nhuận Trạng Thái Cổ Phiếu: {sign}{total_profit:,.0f} VND ({sign}{total_profit_pct:.1f}%)")
    report_lines.append(f"- Tổng rủi ro tiềm ẩn (Số tiền mất nếu hit SL): {portfolio_risk_total:,.0f} VND ({(portfolio_risk_total/nav_current)*100:.1f}% NAV)")
    
    report_lines.append("\nNhận xét cân đối:")
    if (total_market_value/nav_current) > w_target:
        report_lines.append(f"  [!] QUÁ TỶ TRỌNG CỔ PHIẾU: Tỷ trọng hiện tại ({(total_market_value/nav_current)*100:.1f}%) vượt mức khuyến cáo ({w_target*100:.0f}%). Cần chốt lời/hạ tỷ trọng.")
    else:
        report_lines.append(f"  [v] TỶ TRỌNG AN TOÀN: Tỷ lệ phân bổ cổ phiếu ({(total_market_value/nav_current)*100:.1f}%) đang nằm trong mức khuyến cáo ({w_target*100:.0f}%).")
        
    if portfolio_risk_total > risk_max_nav:
        report_lines.append("  [!] CẢNH BÁO RỦI RO: Rủi ro tổng đang VƯỢT QUÁ 3% NAV. Ưu tiên số 1 là giảm tỷ trọng các mã vi phạm hoặc cắt lỗ ngay lập tức để bảo vệ vốn.")
    else:
        report_lines.append("  [v] RỦI RO KIỂM SOÁT TỐT: Rủi ro tổng trong tầm kiểm soát (< 3% NAV).")
        
    valid_tickers_count = len([r for r in results if r['valid']])
    if valid_tickers_count > n_tickers:
        report_lines.append(f"  [!] DANH MỤC DÀN TRẢI: Bạn đang cầm {valid_tickers_count} mã, vượt quá số lượng tối ưu là {n_tickers} mã. Khuyên dùng: Tỉa cỏ trồng hoa, bán bớt các mã gãy trend/yếu.")
    
    report_lines.append("\n2. ĐÁNH GIÁ CHI TIẾT TỪNG MÃ (Tư duy xử lý)")
    for res in results:
        if not res['valid']: continue
        t = res['ticker']
        pl_sign = "+" if res['pl_pct'] > 0 else ""
        val_res = res['val']
        h_rating = val_res.get('tech_health', {}).get('health_rating', 'N/A')
        h_score = val_res.get('tech_health', {}).get('health_score', 0)
        
        an_core = res['analysis']
        val_core = an_core.get('valuation', {})
        state_rules = an_core.get('state_rules', {})
        m = state_rules.get("metrics", {})
        
        # Mô phỏng chính xác logic Chiến lược cốt lõi của analyzer
        state = val_core.get("state", "NONE")
        sig_map = {
            "STRONG": "Mua mạnh (Trend Leader)", "ADD_2": "Gia tăng vị thế 2 (Confirm)",
            "ADD_1": "Gia tăng vị thế 1 (Pullback)", "EARLY": "Mua sớm (Thăm dò)", "NONE": "Chưa có tín hiệu dứt khoát"
        }
        holding_sig = sig_map.get(state, "Chưa có tín hiệu dứt khoát")
        
        rt_sig_map = {
            "BREAKOUT_BUY": "MUA BREAKOUT (Tiền tấn công)", "PULLBACK_BUY": "MUA PULLBACK (Tiền gốc)",
            "RETEST_BUY": "MUA RETEST (Điểm Giàu)", "CONTINUATION_BUY": "GIA TĂNG (Trend Confirm)",
            "TREND_FOLLOW": "ÔM TIẾP (Theo sóng)", "TAKE_PROFIT": "CHỐT LÃI (Canh nhả hàng)",
            "EXIT_OR_SHORT": "THOÁT HÀNG (Rủi ro)", "EXIT_FAST": "CHẠY NGAY (Bẫy giá)", "SHORT": "Đứng ngoài hoàn toàn"
        }
        realtime_sig = rt_sig_map.get(state_rules.get("signal", ""), "")
        sr_signal = realtime_sig if realtime_sig else holding_sig
        
        avoid_entry = state_rules.get("avoid_entry", False)
        if avoid_entry and (sr_signal.upper().startswith("MUA") or sr_signal.upper().startswith("GIA TĂNG")):
            if m.get("anti_trap_block"):
                sr_signal = "BLOCK (Rủi ro Fomo: Đợi chỉnh)"
                
        # Ghi nhận lại vào state_sig để Bảng tư vấn bên dưới xài chung
        res['state_sig'] = sr_signal.upper()
        
        # --- BẮT ĐẦU TÍNH TOÁN ACTION SỚM ĐỂ ĐỒNG BỘ ---
        q_i = res['q']
        p_avg_vnd = res['p_avg_vnd']
        p_now_vnd = res['p_now_vnd']
        p_sup_vnd = res['p_sup_vnd']
        p_res_vnd = res['p_res_vnd']
        w_curr = res['w_curr']
        pl_pct = res['pl_pct']
        
        status_desc = val_res.get('tech_health', {}).get('health_rating', 'BT')
        action = "HOLD"
        q_action = 0
        p_action_vnd = 0
        reason = ""
        
        if w_curr > w_target/n_tickers:
            status_desc = "Quá Tỷ Trọng"
        elif pl_pct < -0.05:
            status_desc = "Đang Lỗ/Yếu"
            
        sig_upper = sr_signal.upper()
        _rs = val_core.get("risk_score", 50)
        anti_trap = m.get("anti_trap_block", False)
        
        # Quy tắc 1: Cắt lỗ & Chặn lãi tuyệt đối (Bảo vệ vốn)
        if p_now_vnd < res['p_sl_vnd']:
            action = "BÁN HẾT (100%)"
            q_action = q_i
            p_action_vnd = p_now_vnd
            reason = f"Vi phạm điểm cắt lỗ/chặn lãi ({res['sl_source']})."
            
        # Quy tắc 2: Đồng bộ Tín hiệu Chốt Lời / Rủi Ro từ AI Lõi
        # Ưu tiên bỏ qua nếu phân tích đơn lẻ cho thấy vị thế đang khỏe (STRONG/ADD_1/ADD_2) và không quá rủi ro
        elif ("CHỐT" in sig_upper or "THOÁT" in sig_upper or "CHẠY" in sig_upper or "BLOCK" in sig_upper) and not (state in ("STRONG", "ADD_2", "ADD_1") and _rs <= 75 and not anti_trap):
            if "BLOCK" in sig_upper or "50%" in sig_upper:
                action = "CHỐT LỜI (50%)"
                q_action = q_i * 0.5
            else:
                action = "CHỐT LỜI/THOÁT"
                q_action = q_i
            p_action_vnd = p_now_vnd
            reason = f"Đồng bộ AI lõi: {sr_signal}."
            
        # Quy tắc 3: Xử lý khi gãy Trend/Downtrend
        elif h_score <= 20 or "DOWNTREND" in sig_upper or "ĐỨNG NGOÀI" in sig_upper:
            if pl_pct < 0:
                action = "CẮT LỖ (Gãy Trend)"
                q_action = q_i
                p_action_vnd = p_now_vnd
                reason = "Cổ phiếu gãy trend/Downtrend. Cắt bỏ dứt khoát."
            else:
                action = "CHỐT LỜI/THOÁT"
                q_action = q_i
                p_action_vnd = p_now_vnd
                reason = "Trend đảo chiều xấu, ưu tiên chốt lãi bảo vệ vốn."
                
        # Quy tắc 4: Quản trị tỷ trọng (Chặn lãi tỷ trọng)
        elif w_curr > w_target/n_tickers + 0.05: # Vượt dung sai 5%
            action = "HẠ TỶ TRỌNG"
            excess_value = (q_i * p_now_vnd) - v_max_i
            q_action = excess_value / p_now_vnd
            p_action_vnd = p_now_vnd
            if state in ("STRONG", "ADD_2", "ADD_1"):
                reason = f"Cổ phiếu khỏe ({state}) nhưng tỷ trọng ({w_curr*100:.1f}%) quá lớn, ưu tiên hạ bớt."
            else:
                reason = f"Tỷ trọng hiện tại ({w_curr*100:.1f}%) vượt quá mức an toàn."

        # Quy tắc 5: Trung bình giá xuống (Khi đang lỗ)
        elif pl_pct < -0.04:
            if pl_pct < -0.10 or (p_now_vnd - p_sup_vnd)/p_now_vnd > 0.10:
                action = "CẮT LỖ BỚT"
                q_action = q_i * 0.5
                p_action_vnd = p_now_vnd
                reason = "Lỗ > 10% hoặc xa hỗ trợ > 10%. Tuyệt đối không TBG."
            elif p_now_vnd <= p_sup_vnd * 1.02: # Ở vùng hỗ trợ
                current_loss_abs = q_i * (p_avg_vnd - p_now_vnd) if p_now_vnd < p_avg_vnd else 0
                X_max = (risk_max_nav - current_loss_abs) / 0.07
                
                if X_max <= 0:
                    action = "KHÔNG TBG"
                    reason = "Lỗ hiện tại đã chiếm hết hạn mức rủi ro 3% NAV."
                elif h_score >= 40: # Có cơ sở bật tăng
                    q_add_max = X_max / p_sup_vnd
                    q_action = min(q_add_max, max(0, (v_max_i - q_i * p_now_vnd) / p_sup_vnd))
                    if q_action > 0:
                        action = "MUA TBG XUỐNG"
                        p_action_vnd = p_sup_vnd
                        reason = f"Về hỗ trợ ({p_sup_vnd/1000:.1f}), Sức khỏe tốt. Mua giảm giá vốn."
                    else:
                        action = "CHỜ ĐỢI"
                        reason = "Đã hết mức tỷ trọng cho phép để trung bình giá."
                else:
                    action = "CHỜ ĐỢI"
                    reason = "Ở hỗ trợ nhưng cổ phiếu yếu (<40đ), rủi ro thủng nền cao."
            else:
                action = "CHỜ VỀ HỖ TRỢ"
                reason = f"Đang lơ lửng, đợi nhúng về vùng {p_sup_vnd/1000:.1f}."

        # Quy tắc 6: Mua gia tăng (Ưu tiên theo đánh giá Vị thế đang cầm cổ từ Phân tích đơn lẻ)
        elif state in ("STRONG", "ADD_2", "ADD_1") and _rs <= 75 and not anti_trap:
             if w_curr < (w_target/n_tickers)*0.8:
                  action = "MUA GIA TĂNG"
                  q_action = (v_max_i - q_i * p_now_vnd) / p_now_vnd
                  p_action_vnd = p_now_vnd
                  reason = f"Phân tích đơn lẻ: Vị thế {state} khỏe. Gia tăng tỷ trọng."
             else:
                  action = "HOLD"
                  reason = "Vị thế đang khỏe nhưng đã đủ tỷ trọng. Gồng lãi."
                  
        # Quy tắc 7: Mua gia tăng theo Tín hiệu Lõi (Fallback)
        elif "MUA" in sig_upper or "GIA TĂNG" in sig_upper or ("TREND" in sig_upper and pl_pct > 0.03 and h_score >= 60):
             if w_curr < (w_target/n_tickers)*0.8:
                  action = "MUA GIA TĂNG"
                  q_action = (v_max_i - q_i * p_now_vnd) / p_now_vnd
                  p_action_vnd = p_sup_vnd if p_now_vnd > p_sup_vnd * 1.05 else p_now_vnd
                  reason = f"Đồng bộ AI lõi: {sr_signal}. Nhặt thêm tại {p_action_vnd/1000:.1f}."
             else:
                  action = "HOLD"
                  reason = "Trend khỏe nhưng đã đủ tỷ trọng. Tiếp tục gồng lãi."

        if not reason:
            reason = "Duy trì vị thế hiện tại, theo dõi thêm."

        # Lưu lại kết quả vào res
        res['action'] = action
        res['q_action'] = q_action
        res['p_action_vnd'] = p_action_vnd
        res['reason'] = reason
        res['status_desc'] = status_desc
        
        # --- KẾT THÚC TÍNH TOÁN ACTION ---

        desc = f"- {t}: Đang chiếm {res['w_curr']*100:.1f}% NAV. Lãi/lỗ: {pl_sign}{res['pl_pct']*100:.1f}%. "
        desc += f"Chất lượng KT: {h_rating} ({h_score}đ). Tín hiệu AI: {sr_signal.upper()}. "
        desc += f"Hướng xử lý chiến lược: {action} ({reason})."
        
        report_lines.append(desc)

    report_lines.append("\n3. BẢNG TƯ VẤN KIẾN NGHỊ XỬ LÝ (ACTION)")
    report_lines.append("-" * 125)
    header = f"| {'Mã':<6} | {'Lãi/Lỗ %':<9} | {'Trạng Thái':<12} | {'Khuyến Nghị':<16} | {'KL Hiện Tại':<12} | {'KL Khuyến Nghị':<15} | {'Giá Bán/Mua':<12} | {'Lý do Kỹ thuật'}"
    report_lines.append(header)
    report_lines.append("-" * 125)

    # 3. Action Logic for each ticker
    for res in results:
        if not res['valid']:
            row = f"| {res['ticker']:<6} | {'-':<9} | {'Lỗi Dữ Liệu':<16} | {'Bỏ qua':<18} | {'-':<15} | {'-':<12} | {res['msg']}"
            report_lines.append(row)
            continue
            
        ticker = res['ticker']
        q_i = res['q']
        p_avg_vnd = res['p_avg_vnd']
        p_now_vnd = res['p_now_vnd']
        p_sup_vnd = res['p_sup_vnd']
        p_res_vnd = res['p_res_vnd']
        w_curr = res['w_curr']
        pl_pct = res['pl_pct']
        trend_i = res['trend']
        tech_weak = res['tech_weak']
        sideways_near_res = res['sideways_near_res']
        
        # Đọc kết quả đã tính toán ở Bước 2
        status_desc = res['status_desc']
        action = res['action']
        q_action = res['q_action']
        p_action_vnd = res['p_action_vnd']
        reason = res['reason']





                






                  
        # Quy tắc 7: Mua gia tăng theo Tín hiệu Lõi (Fallback)





        pl_str = f"{pl_pct*100:+.1f}%"
        q_curr_str = f"{int(q_i):,}"
        
        # Làm tròn khối lượng khuyến nghị đến hàng trăm theo yêu cầu người dùng
        q_action_rounded = round(q_action / 100) * 100 if q_action else 0
        q_str = f"{int(q_action_rounded):,}" if q_action_rounded else "-"
        
        p_str = f"{p_action_vnd/1000:.1f}" if p_action_vnd else "-"
        
        row = f"| {ticker:<6} | {pl_str:<9} | {status_desc:<12} | {action:<16} | {q_curr_str:<12} | {q_str:<15} | {p_str:<12} | {reason}"
        report_lines.append(row)

    report_lines.append("-" * 125)
    report_lines.append("\nQUY TẮC BẢO VỆ (Vô hiệu hóa khuyến nghị)")
    report_lines.append("- Nếu thị trường chung (VN-INDEX) xác nhận gãy Trend hoặc rủi ro vĩ mô đột biến, HỦY TOÀN BỘ LỆNH MUA.")
    report_lines.append("- Các mức hỗ trợ/kháng cự có thể thay đổi sau phiên giao dịch. Không mua mù quáng nếu cổ phiếu thủng hỗ trợ với Vol lớn.")
    
    return "\n".join(report_lines)

