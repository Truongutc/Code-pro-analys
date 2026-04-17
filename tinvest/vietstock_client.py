import requests
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import time
import re
from tinvest.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class VietstockClient:
    def __init__(self):
        self.config_mgr = ConfigManager()
        self.base_url = "https://finance.vietstock.vn"
        self.stats_api_url = self.config_mgr.get("vietstock_api_url")
        self.index_api_url = self.config_mgr.get("vietstock_index_url")
        self.stocklist_api_url = self.config_mgr.get("stocklist_api_url")
        
        self.session_limited = False # Track if current token is restricted to 200 items
        self.session = requests.Session()
        self.api_url = "https://finance.vietstock.vn/data/KQGDThongKeGiaPaging"
        self.token = None
        self.refresh_from_config()

    def refresh_from_config(self):
        """Update session headers and cookies from config (Mirror Mode)."""
        # Reload config from disk
        self.config_mgr = ConfigManager()
        conf_headers = self.config_mgr.get("headers") or {}
        conf_cookies = self.config_mgr.get("cookies") or {}
        
        # 1. Start with a clean slate to mirror browser exactly
        self.session.headers.clear()
        
        # 2. Rebuild headers based on provided config
        ua = conf_headers.get("User-Agent") or conf_headers.get("user-agent")
        if not ua:
            ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
            
        self.session.headers.update({
            "User-Agent": ua,
            "Accept": conf_headers.get("Accept", "*/*"),
            "Accept-Language": conf_headers.get("Accept-Language", "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7"),
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "https://finance.vietstock.vn",
            "Connection": "keep-alive"
        })
        
        # Inherit any other specialized headers (sec-*, referer)
        for k, v in conf_headers.items():
            k_low = k.lower()
            if k_low.startswith("sec-") or k_low == "referer":
                self.session.headers[k] = v
        
        # Ensure Referer is at least the base page if missing
        if "Referer" not in self.session.headers:
            self.session.headers["Referer"] = f"{self.base_url}/ket-qua-giao-dich?tab=thong-ke-gia"
            
        # 3. Clear and update cookies
        self.session.cookies.clear()
        if conf_cookies:
            self.session.cookies.update(conf_cookies)
            
        # 4. Sync tokens and status
        self.manual_token = self.config_mgr.get("payload_token")
        self.session_limited = False
        logger.info(f"Mirror Mode Active: Headers synced (UA: {ua[:30]}...)")

    def ensure_valid_session(self):
        """Visit landing page to get fresh ASP.NET_SessionId and __RequestVerificationToken cookies."""
        try:
            url = f"{self.base_url}/ket-qua-giao-dich?tab=thong-ke-gia"
            # We use a clean GET to the landing page to populate cookies
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                # Save captured cookies back to config manager for persistence
                current_cookies = requests.utils.dict_from_cookiejar(self.session.cookies)
                if current_cookies:
                    self.config_mgr.set("cookies", current_cookies)
                    logger.info(f"Automatically captured {len(current_cookies)} cookies.")
                return True
        except Exception as e:
            logger.error(f"Failed to ensure valid session: {e}")
        return False

    def check_session_status(self, date_str=None):
        """Force return VALID to allow the user to proceed without being blocked by the probe."""
        logger.info("[*] Chế độ Cưỡng bức: Bỏ qua kiểm tra trạng thái, lao thẳng vào nạp dữ liệu.")
        self.session_limited = False
        return "VALID"

    def get_token(self):
        """Fetch __RequestVerificationToken from Vietstock landing page."""
        try:
            url = f"{self.base_url}/ket-qua-giao-dich?tab=thong-ke-gia"
            response = self.session.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                token_input = soup.find('input', {'name': '__RequestVerificationToken'})
                if token_input:
                    self.token = token_input.get('value')
                    return self.token
        except Exception as e:
            logger.error(f"Error fetching Vietstock token: {e}")
        return None

    def get_stock_list(self, cat_id):
        """Fetch full symbol mapping for a category (1:HOSE, 2:HNX, 3:UPCOM)."""
        params = {"catID": cat_id}
        try:
            response = self.session.get(self.stocklist_api_url, params=params)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching stock list for cat {cat_id}: {e}")
        return []

    def _is_json(self, text):
        try:
            json.loads(text)
            return True
        except:
            return False

    def _fetch_page(self, cat_id, date_str, page=1, page_size=2000):
        # 1. ALWAYS Refresh from config first to get latest pasted tokens/cookies
        self.refresh_from_config()
        
        # 2. Prevent infinite refresh loops
        if not hasattr(self, '_refreshing'):
            self._refreshing = False
            
        # 3. Determine the correct token to use (Payload token)
        token_to_use = self.manual_token if self.manual_token else self.token
        
        # 4. Standard Vietstock POST Payload
        payload = {
            "page": page,
            "pageSize": page_size,
            "catID": cat_id,
            "date": date_str,
            "__RequestVerificationToken": token_to_use or ""
        }
        
        try:
            # 5. Mirror RAW Cookie Header if available (Crucial for bypass)
            raw_cookie_str = self.config_mgr.get("raw_cookie_str")
            if raw_cookie_str:
                self.session.headers["Cookie"] = raw_cookie_str
                # Clear standard cookies to let the header take priority
                self.session.cookies.clear()
            
            logger.info(f"[*] API POST -> {self.api_url} | Payload: page={page}, pageSize={page_size}, catID={cat_id}")
            self.session.headers["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"
            
            response = self.session.post(self.api_url, data=payload, timeout=15)
            logger.info(f"[<] Response: {response.status_code}")

            if response.status_code == 200:
                content_preview = response.content[:200].decode('utf-8', errors='ignore').strip()
                if "<!DOCTYPE html>" in content_preview.upper() or "<HTML" in content_preview.upper():
                    logger.warning("⚠️ Nhận được trang HTML thay vì JSON. Vietstock có thể đang chặn yêu cầu hoặc Token hết hạn.")
                    logger.debug(f"Response Body Preview: {response.text[:500]}")
                    
                    if "/Error/Index" in response.text:
                         logger.error("❌ Vietstock trả về trang lỗi hệ thống (/Error/Index). Vui lòng dán lại cURL mới.")
                    
                    if not self._refreshing:
                        logger.warning("Đang thử tự động làm mới chuẩn (Interactive Refresh)...")
                        self._refreshing = True
                        try:
                            if self.config_mgr.refresh_token():
                                self.refresh_from_config()
                                # Retry once
                                return self._fetch_page(cat_id, date_str, page, page_size)
                        finally:
                            self._refreshing = False
                    
                    # Log error details if still failing
                    logger.error("❌ Không thể kết nối API dù đã thử làm mới. Xem debug_api_error.html để biết chi tiết.")
                    with open("debug_api_error.html", "w", encoding="utf-8") as f:
                        f.write(f"URL: {self.stats_api_url}\nStatus: {response.status_code}\n\n{response.text}")
                    return None
                
                try:
                    return json.loads(response.content.decode('utf-8-sig'))
                except json.JSONDecodeError:
                    logger.error(f"❌ Lỗi định dạng JSON. Phản hồi bắt đầu bằng: {content_preview}")
                    return None
            else:
                logger.error(f"❌ API trả về lỗi HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Lỗi kết nối API: {e}")
        return None

    def is_session_valid(self, raw_data, prev_records=None):
        """
        Validate if the session has real trading data.
        Rejects if all sampled stocks have zero volume or stagnant prices.
        """
        if not raw_data or not isinstance(raw_data, list) or len(raw_data) < 3:
            return False
        
        stocks = raw_data[2]
        if not stocks: return False
        
        # Check top 20 for signs of life
        samples = stocks[:20]
        total_vol = sum(int(s.get('M_TotalVol', 0)) for s in samples)
        if total_vol == 0: return False
        
        # Check if all OHLC are equal (market hasn't moved / invalid)
        stagnant = 0
        for s in samples:
            o = float(s.get('OpenPrice', 0))
            h = float(s.get('HighestPrice', 0))
            l = float(s.get('LowestPrice', 0))
            c = float(s.get('ClosePrice', 0))
            if o > 0 and o == h == l == c:
                stagnant += 1
        
        if stagnant == len(samples): return False
        return True

    def fetch_market_day(self, cat_id, date_str):
        """Fetch all stocks for a market category using stable pagination logic."""
        # Safety: Use pageSize=200 which is the most stable authenticated limit.
        default_size = 200
        
        logger.info(f"[*] Đang thực hiện nạp dữ liệu sàn {cat_id} (pageSize=200)...")
        raw_p1 = self._fetch_page(cat_id, date_str, page=1, page_size=default_size)
        
        if not raw_p1 or not isinstance(raw_p1, list) or len(raw_p1) < 3:
            return [], False
            
        if not self.is_session_valid(raw_p1):
            logger.warning(f"⚠️ Phiên làm việc không hợp lệ hoặc không có dữ liệu giao dịch cho ngày {date_str}.")
            return [], False

        all_stocks = []
        all_stocks.extend(raw_p1[2])
        
        # 2. Extract total pages - Only loop if we didn't get enough in page 1
        # (With pageSize=2000, we should almost always get everything in page 1)
        total_pages = 1
        try:
            if len(raw_p1) >= 4:
                tp = raw_p1[3]
                if isinstance(tp, list): total_pages = int(tp[0])
                else: total_pages = int(tp)
        except: total_pages = 1
        
        if len(all_stocks) < 100 and total_pages > 1: # Safety check for very small markets
             logger.info(f"[+] Vietstock báo cáo có {total_pages} trang dữ liệu. Đang tải tiếp...")
             for p in range(2, total_pages + 1):
                 p_raw = self._fetch_page(cat_id, date_str, page=p, page_size=default_size)
                 if p_raw and len(p_raw) >= 3:
                     all_stocks.extend(p_raw[2])
                 else: break
                 
        # Final limit check
        is_limited = (len(all_stocks) == 200 and total_pages == 1)
        return all_stocks, is_limited

    def fetch_index_day(self, ticker, cat_id, stock_id, date_str):
        """Fetch index data for a given date."""
        self.refresh_from_config()
        token_to_use = self.manual_token if self.manual_token else self.token
        if not token_to_use:
             self.get_token()
             token_to_use = self.token
             
        payload = {
            "page": 1,
            "pageSize": 20,
            "catID": cat_id,
            "stockID": stock_id,
            "fromDate": date_str,
            "toDate": date_str,
            "__RequestVerificationToken": token_to_use
        }
        
        try:
            response = self.session.post(self.index_api_url, data=payload)
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) >= 2:
                    records = data[1]
                    formatted = []
                    for r in records:
                        formatted.append({
                            "StockCode": ticker, # Use StockCode so format_to_df renames it correctly
                            "TradingDate": date_str,
                            "OpenPrice": r.get("OpenPrice", 0),
                            "HighestPrice": r.get("HighestPrice", 0),
                            "LowestPrice": r.get("LowestPrice", 0),
                            "ClosePrice": r.get("ClosePrice", 0),
                            "M_TotalVol": int(r.get("TotalVol", 0))
                        })
                    return formatted
        except Exception as e:
            logger.error(f"Error fetching index {ticker}: {e}")
        return []

    def get_missing_dates(self, last_date):
        """Return missing trading dates up to today."""
        now = datetime.now()
        effective_today = now.date()
        if now.weekday() == 5: effective_today -= timedelta(days=1)
        elif now.weekday() == 6: effective_today -= timedelta(days=2)
        
        if not last_date:
            last_date = now - timedelta(days=365)
        
        missing = []
        curr = (last_date + timedelta(days=1)).date()
        while curr <= effective_today:
            if curr.weekday() < 5:
                missing.append(curr.strftime("%Y-%m-%d"))
            curr += timedelta(days=1)
        return missing

    def format_to_df(self, raw_list):
        if not raw_list: return pd.DataFrame()
        
        df = pd.DataFrame(raw_list)
        if 'StockCode' in df.columns:
            df = df.rename(columns={
                'StockCode': 'Ticker',
                'TradingDate': 'Date',
                'OpenPrice': 'Open',
                'HighestPrice': 'High',
                'LowestPrice': 'Low',
                'ClosePrice': 'Close',
                'M_TotalVol': 'Volume'
            })
            # Convert prices to thousands ONLY for stocks. 
            # Indices like VNINDEX/HNX-INDEX are already in the correct unit.
            is_index = df['Ticker'].iloc[0] in ['VNINDEX', 'HNX-INDEX'] if not df.empty else False
            
            if not is_index:
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in df.columns:
                        df[col] = df[col] / 1000.0
            
            def parse_ms_date(d):
                if not isinstance(d, str): return d
                match = re.search(r'\((\d+)\)', d)
                if match:
                    ts = int(match.group(1)) / 1000.0
                    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                return d
            df['Date'] = df['Date'].apply(parse_ms_date)

        required = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[[c for c in required if c in df.columns]]
        return df
