import unittest
import pandas as pd
import numpy as np
import sys
import os

# Append the project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AICcode import CUSTOM_RULES

class TestCustomRules(unittest.TestCase):
    def test_basic_ma_crossovers(self):
        # Create a mock dataframe with Close, MA20, MA50
        dates = pd.date_range(start="2026-01-01", periods=10)
        df = pd.DataFrame({
            "Date": dates,
            "Close": [10, 11, 12, 13, 9, 8, 11, 12, 13, 14], # cross MA20 at index 6 (8 -> 11)
            "MA20": [10.0] * 10,
            "MA50": [9.0] * 10,
        })
        
        # Test Price cuts above MA20
        rule = CUSTOM_RULES["PRICE_CROSS_MA20"]
        
        # Case 1: Ticker is above MA20 (14) and yesterday was above (13) -> No cross
        self.assertFalse(rule["func"](df))
        
        # Case 2: Ticker just crossed (let's truncate to index 7: date 7, close 11, prev close 8)
        df_crossed = df.iloc[:7]
        self.assertTrue(rule["func"](df_crossed))

    def test_macd_signals(self):
        df = pd.DataFrame({
            "Close": [10, 11, 12],
            "MACD": [1.0, 1.2, 1.5],
            "MACD_Signal": [1.1, 1.3, 1.4] # MACD crossed above Signal at index 2 (1.2/1.3 -> 1.5/1.4)
        })
        
        rule_cross = CUSTOM_RULES["MACD_CROSS_SIGNAL"]
        rule_above = CUSTOM_RULES["MACD_ABOVE_SIGNAL"]
        
        self.assertTrue(rule_cross["func"](df))
        self.assertTrue(rule_above["func"](df))

    def test_week_high_breakout(self):
        rule = CUSTOM_RULES["WEEK_HIGH_BREAKOUT"]
        # High of past 5 days (indexes -6 to -2) is 10.
        # Close at index -1 is 11.
        df = pd.DataFrame({
            "Close": [5, 5, 5, 5, 5, 5, 11],
            "High":  [6, 7, 8, 9, 10, 8, 12]
        })
        self.assertTrue(rule["func"](df))

        # Close at index -1 is 9 (not breakout).
        df_no = pd.DataFrame({
            "Close": [5, 5, 5, 5, 5, 5, 9],
            "High":  [6, 7, 8, 9, 10, 8, 12]
        })
        self.assertFalse(rule["func"](df_no))

    def test_accumulation_breakout(self):
        rule = CUSTOM_RULES["ACCUMULATION_BREAKOUT"]
        
        # Scenario 1: Sideways for 20 days with High=10.5, Low=9.5 (range = (10.5-9.5)/9.5 = 10.5% <= 15%)
        # Then breaks out with Close=11.0 today (index -1)
        closes = [10.0] * 20 + [11.0]
        highs = [10.5] * 20 + [11.5]
        lows = [9.5] * 20 + [9.5]
        df = pd.DataFrame({"Close": closes, "High": highs, "Low": lows})
        self.assertTrue(rule["func"](df))

        # Scenario 2: Breakout was 2 days ago (index -3), and it should still trigger.
        # Let's add 2 days of post-breakout data:
        closes_2 = [10.0] * 20 + [11.0, 11.2, 11.5]
        highs_2 = [10.5] * 20 + [11.5, 11.8, 12.0]
        lows_2 = [9.5] * 20 + [9.5, 11.0, 11.2]
        df_2 = pd.DataFrame({"Close": closes_2, "High": highs_2, "Low": lows_2})
        self.assertTrue(rule["func"](df_2))

        # Scenario 3: Consolidation range is too wide (High=12, Low=9, range = 33% > 15%) -> Should be False
        closes_wide = [10.0] * 20 + [13.0]
        highs_wide = [12.0] * 20 + [13.5]
        lows_wide = [9.0] * 20 + [9.0]
        df_wide = pd.DataFrame({"Close": closes_wide, "High": highs_wide, "Low": lows_wide})
        self.assertFalse(rule["func"](df_wide))

if __name__ == "__main__":
    unittest.main()
