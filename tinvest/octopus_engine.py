import pandas as pd
import numpy as np

def mcginley_dynamic(series, period):
    """
    McGinley Dynamic calculation.
    Formula: MD[i] = MD[i-1] + (Price[i] - MD[i-1]) / (N * (Price[i] / MD[i-1])^4)
    """
    md = np.zeros(len(series))
    md[0] = series.iloc[0]
    prices = series.values
    for i in range(1, len(series)):
        prev_md = md[i-1]
        if prev_md <= 0: 
            md[i] = prices[i]
            continue
            
        ratio = prices[i] / prev_md
        # Tránh mẫu số bằng 0 khi prices[i] = 0
        denom = period * (ratio**4)
        if denom < 0.01:
            # Nếu mẫu số quá nhỏ, tiến dần về EMA bình thường hoặc giữ nguyên giá trị
            md[i] = prev_md + (prices[i] - prev_md) / period
        else:
            md[i] = prev_md + (prices[i] - prev_md) / denom
    return pd.Series(md, index=series.index)

def analyze_octopus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements Rule Octopus (MACD Band with McGinley Dynamic)
    AFL conversion:
    A1 = MCGin(C, 12) - MCGin(C, 25)
    B1 = MCGin(C, 25) - MCGin(C, 12)
    BBands on A1 (20, 1)
    """
    df = df.copy()
    
    # 1. McGinley Dynamic averages
    mc12 = mcginley_dynamic(df['Close'], 12)
    mc25 = mcginley_dynamic(df['Close'], 25)
    
    # 2. MACD values
    df['OCT_A1'] = mc12 - mc25
    df['OCT_B1'] = mc25 - mc12
    
    # 3. Bollinger Bands on A1
    periods = 20
    width = 1
    df['OCT_BB_Mid'] = df['OCT_A1'].rolling(window=periods).mean()
    df['OCT_BB_Std'] = df['OCT_A1'].rolling(window=periods).std()
    df['OCT_BB_Top'] = df['OCT_BB_Mid'] + (width * df['OCT_BB_Std'])
    df['OCT_BB_Bot'] = df['OCT_BB_Mid'] - (width * df['OCT_BB_Std'])
    
    # 4. Color Logic
    # Color=IIf(a1<0 AND a1>Ref(a1,-1), colorGreen,IIf(a1>0 AND a1>Ref(a1,-1) ,colorBrightGreen,IIf(a1>0 AND a1<Ref(a1,-1),colorCustom12,colorRed)));
    a1 = df['OCT_A1']
    a1_prev = a1.shift(1)
    
    conditions = [
        (a1 < 0) & (a1 > a1_prev),
        (a1 > 0) & (a1 > a1_prev),
        (a1 > 0) & (a1 < a1_prev),
        (a1 < 0) & (a1 <= a1_prev)
    ]
    # Sky Blue - Cover, Light Green - Buy, Pink - Sell, Orange- Short
    # But AFL uses: Green, BrightGreen, Custom12 (Pink), Red
    choices = ['#008000', '#00FF00', '#FF69B4', '#FF0000'] 
    df['OCT_Color'] = np.select(conditions, choices, default='#808080')
    
    return df
