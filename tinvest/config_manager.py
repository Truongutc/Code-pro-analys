import json
import os
import re
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path="config.json"):
        self.config_path = Path(config_path)
        self.default_config = {
            "vietstock_api_url": "https://finance.vietstock.vn/data/KQGDThongKeGiaPaging",
            "vietstock_index_url": "https://finance.vietstock.vn/data/KQGDThongKeGiaStockPaging",
            "stocklist_api_url": "https://finance.vietstock.vn/data/stocklist",
            "bypass_pageSize": 50,
            "cookies": {},
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Accept-Language": "en-US,en;q=0.9",
                "X-Requested-With": "XMLHttpRequest",
                "Connection": "keep-alive"
            }
        }
        self.config = self._load()

    def _load(self):
        if not self.config_path.exists():
            self._save(self.default_config)
            return self.default_config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Merge with defaults to ensure all keys exist
                for k, v in self.default_config.items():
                    if k not in data:
                        data[k] = v
                return data
        except Exception:
            return self.default_config

    def _save(self, data):
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def get(self, key):
        return self.config.get(key, self.default_config.get(key))

    def set(self, key, value):
        self.config[key] = value
        self._save(self.config)

    def update_url(self, url):
        self.set("vietstock_api_url", url)

    def _sanitize_curl(self, text):
        """Remove shell escapes and line continuations to make parsing easier."""
        if not text: return ""
        # 1. Remove line continuations (backslash or caret)
        text = text.replace("\\\n", " ").replace(" ^\n", " ").replace("^\n", " ")
        # 2. General shell cleanup
        text = text.replace("^$", "$").replace("^\"", "\"")
        text = text.replace("^\\^\"", "\"").replace("\\^\"", "\"").replace("\\\"", "\"")
        # 3. Last resort: remove any remaining ^ before symbols
        text = re.sub(r"\^([=:\s\$])", r"\1", text)
        return text

    def _is_tracked_header(self, name):
        tracked = {
            "user-agent", "referer", "origin", "accept-language", "accept",
            "sec-fetch-dest", "sec-fetch-mode", "sec-fetch-site",
            "sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform",
            "x-requested-with", "content-type", "connection"
        }
        return name.lower() in tracked

    def _should_preserve_case(self, name):
        """Headers starting with sec- should stay lowercase."""
        return name.lower().startswith("sec-")

    def parse_input(self, text):
        """
        Extract cookies, tokens and headers from raw text (cURL or Browser Headers).
        Always returns True if anything was updated.
        """
        if not text: return False
        
        # Cleanup input (remove helper markers if present)
        text = re.sub(r'---.*---', '', text).strip()
        
        # 1. Handle Multi-cURL paste: Pick the right one
        target_text = text
        if "curl" in text.lower():
            # Split by 'curl ' but handle the first one
            chunks = re.split(r'curl\s+', text, flags=re.IGNORECASE)
            chunks = [c.strip() for c in chunks if c.strip()]
            
            # Prioritize Vietstock Data Paging or StockList
            priority_keywords = ["KQGDThongKeGiaPaging", "GetTemplateByName", "stocklist", "finance.vietstock.vn"]
            
            best_chunk = ""
            for kw in priority_keywords:
                for c in chunks:
                    if kw in c:
                        best_chunk = c
                        break
                if best_chunk: break
            
            target_text = "curl " + (best_chunk or chunks[0])

        # Detect if we are processing a cURL command
        is_curl = "curl" in target_text.lower()[:50]
        if is_curl:
            target_text = self._sanitize_curl(target_text)
        
        updates = {
            "cookies": {},
            "headers": {}, 
            "vietstock_api_url": None,
            "payload_token": "",
            "bypass_pageSize": None
        }
        
        # Initialize headers with existing ones from config
        old_headers = self.config.get("headers", {})
        for k, v in old_headers.items():
            updates["headers"][k] = v

        # 2. Extract Data from cURL
        if is_curl:
            # A. Extract URL from curl
            url_match = re.search(r"curl\s+['\"]?(https?://[^'\"]+)['\"]?", target_text, re.IGNORECASE)
            # B. Extract Headers (Cookie)
            raw_headers = re.findall(r"(?:-H|--header)\s+(?:'([^']+)'|\"((?:\\\\\"|[^\"])+)\")", target_text, re.IGNORECASE)
            for h_tuple in raw_headers:
                h = h_tuple[0] or h_tuple[1]
                if ":" in h:
                    k, v = h.split(":", 1)
                    k, v = k.strip(), v.strip()
                    if k.lower() == "cookie":
                        updates["cookies"].update(self._parse_cookie_str(v))
                    elif self._is_tracked_header(k):
                        updates["headers"][k] = v
            
            # C. Extract Payload (Token)
            data_match = re.search(r"--data(?:-raw|-binary|-ascii)?\s+(?:'([^']+)'|\"((?:\\\\\"|[^\"])+)\")", target_text, re.IGNORECASE)
            if data_match:
                data_val = data_match.group(1) or data_match.group(2)
                params = data_val.split("&")
                for p in params:
                    if "=" in p:
                        tk_key, tv_val = p.split("=", 1)
                        tk_key, tv_val = tk_key.strip(), tv_val.strip()
                        if tk_key == "__RequestVerificationToken":
                            updates["payload_token"] = tv_val
                        elif tk_key.lower() == "pagesize":
                            try:
                                updates["bypass_pageSize"] = int(tv_val)
                                logger.info(f"Extracted pageSize from cURL: {tv_val}")
                            except: pass
        else:
            # Format 2: Universal Multi-line / Raw Header Parser
            # We use a state-machine approach to handle alternating lines (Key \n Value)
            lines = [l.strip() for l in target_text.split("\n") if l.strip()]
            
            # List of keys we are looking for in multi-line format
            header_keys = {k.lower() for k in self._get_tracked_header_list()}
            payload_keys = {"__requestverificationtoken", "pagesize", "catid", "page", "date"}
            
            i = 0
            while i < len(lines):
                line = lines[i]
                line_lower = line.lower().rstrip(":")
                
                # Format 3: Tab-Separated (Application Tab Copy)
                if "\t" in line:
                    parts = [p.strip() for p in line.split("\t") if p.strip()]
                    if len(parts) >= 2:
                        k_tab, v_tab = parts[0], parts[1]
                        self._extract_header_field(k_tab, v_tab, updates)
                        # Special case for payload token in cookie list
                        if k_tab == "__RequestVerificationToken":
                             updates["payload_token"] = v_tab
                    i += 1
                    continue

                # Format 2 (Continued): Alternating lines or Colon-separated
                if ":" in line and not line.startswith("http"):
                    k, v = line.split(":", 1)
                    self._extract_header_field(k.strip(), v.strip(), updates)
                elif line_lower in header_keys or line_lower == "cookie":
                    # Key on this line, Value might be on NEXT line
                    k = line.strip()
                    if i + 1 < len(lines):
                        v = lines[i+1]
                        # If next line looks like another key, then this was a key-only or malformed
                        if v.lower().rstrip(":") not in header_keys and v.lower() != "cookie":
                            self._extract_header_field(k, v, updates)
                            i += 1 # Skip value line
                elif line_lower in payload_keys:
                    # Payload param on this line
                    pk = line.strip()
                    if i + 1 < len(lines):
                        pv = lines[i+1]
                        if pk == "__RequestVerificationToken":
                            updates["payload_token"] = pv
                        elif pk.lower() == "pagesize":
                            try: updates["bypass_pageSize"] = int(pv)
                            except: pass
                        i += 1 # Skip value line
                i += 1
            
            # Format 4: Lone Token Detection
            if not updates["payload_token"] and not updates["cookies"] and not updates["headers"]:
                clean_input = target_text.strip()
                if len(clean_input) > 30 and " " not in clean_input and "\n" not in clean_input:
                    updates["payload_token"] = clean_input
                    logger.info("Detected lone string as payload_token.")

        # D. Dual-token synchronization:
        # DO NOT overwrite payload_token with cookie_token if they are both found!
        # Vietstock often requires them to be different but valid for the same session.
        if not updates["payload_token"]:
             # Fallback: if only cookie has token, use it
             if "__RequestVerificationToken" in updates["cookies"]:
                 updates["payload_token"] = updates["cookies"]["__RequestVerificationToken"]

        # Commit updates
        updated = False
        
        if updates["vietstock_api_url"]:
            self.set("vietstock_api_url", updates["vietstock_api_url"])
            updated = True

        if updates["cookies"]:
            # If we found NEW cookies, we replace the whole cookie dict to avoid mixing old/new sessions
            self.set("cookies", updates["cookies"])
            updated = True
        elif is_curl and not updates["cookies"]:
            # If it was a curl paste but no cookies found (unlikely), something is wrong.
            pass
            
        if updates["payload_token"]:
            self.set("payload_token", updates["payload_token"])
            updated = True
            
        if updates["bypass_pageSize"] is not None:
             self.set("bypass_pageSize", updates["bypass_pageSize"])
             updated = True

        if updates["headers"]:
             current_headers = self.config.get("headers", {})
             # Normalize keys before updating
             new_headers = {}
             
             # If this is a NEW cURL paste, we should clear old session state entirely
             # to avoid mixing cookies or fingerprints from different sessions.
             if is_curl:
                 # WIPE cookies - always use exactly what's in the cURL
                 self.config["cookies"] = {}
                 
                 # CLEAR tracked headers
                 keys_to_clear = [k for k in current_headers.keys() if self._is_tracked_header(k)]
                 for k in keys_to_clear:
                     del current_headers[k]
                 
                 # CLEAR old tokens
                 self.config["payload_token"] = ""

             for k, v in updates["headers"].items():
                 # Filter out restricted headers that requests handles itself
                 if k.lower() not in ["content-length", "host", "connection"]:
                     new_headers[k] = v
             
             # Also ensure we remove these if they were previously in config
             for k in ["Host", "Content-Length", "host", "content-length"]:
                 if k in current_headers:
                     del current_headers[k]

             current_headers.update(new_headers)
             self.set("headers", current_headers)
             updated = True
             
        return updated

    def _is_tracked_header(self, k):
        return k.lower() in self._get_tracked_header_list()

    def _get_tracked_header_list(self):
        return ["user-agent", "origin", "referer", "x-requested-with", 
                "sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform",
                "sec-fetch-dest", "sec-fetch-mode", "sec-fetch-site", "accept-language", "accept"]

    def _extract_header_field(self, k, v, updates):
        k_lower = k.lower().rstrip(":")
        
        # If it's a known header, store it in headers
        if self._is_tracked_header(k_lower):
            updates["headers"][k] = v
            return

        # Otherwise, treat it as a potential cookie
        # (We want to keep ALL cookies provided by the user now)
        if k_lower == "cookie":
            updates["cookies"].update(self._parse_cookie_str(v))
        else:
            val = v.strip().strip(";").strip()
            updates["cookies"][k] = val
            logger.info(f"[*] Đã nhận diện Cookie: {k}")

    def refresh_token(self):
        """Force a fresh token acquisition using Selenium."""
        from tinvest.token_refresher import fetch_fresh_token
        result = fetch_fresh_token()
        if result:
            self.config = self._load() # Reload from disk
            return True
        return False

    def _parse_cookie_str(self, cookie_str):
        cookies = {}
        pairs = cookie_str.split(";")
        for p in pairs:
            if "=" in p:
                k, v = p.strip().split("=", 1)
                cookies[k] = v
        return cookies
