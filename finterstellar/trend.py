import pandas as pd


def rsi(df, w=5):
    df.fillna(method='ffill', inplace=True)  # 들어온 데이터의 구멍을 메꿔준다
    if len(df) > w+1:
        df['diff'] = df.iloc[:,0].diff()   # 일별 가격차이 계산
        df['diff_abs'] = df.iloc[:,0].diff().abs()   # 일별 가격차이의 절대값 계산
        df['diff_positive'] = df['diff'].mask(df['diff']<0, 0)   # 가격차이가 +인 경우만 발라내기
        df['diff_abs_rolling_sum'] = df['diff_abs'].rolling(w).sum()   # RSI 분모
        df['diff_positive_rolling_sum'] = df['diff_positive'].rolling(w).sum()   # RSI 분자
        df['rsi'] = df['diff_positive_rolling_sum'].truediv(df['diff_abs_rolling_sum'], fill_value=.5)*100   # RSI
        df.drop(columns=['diff','diff_abs','diff_positive','diff_abs_rolling_sum','diff_positive_rolling_sum'], inplace=True)
        return df['rsi'].round(2)
    else:
        return None


def wrsi(price_df, w=5):
    df.fillna(method='ffill', inplace=True)  # 들어온 데이터의 구멍을 메꿔준다
    if len(df) > w+1:
        df['diff'] = df.iloc[:,0].diff()   # 일별 가격차이 계산
        df['diff_abs'] = df.iloc[:,0].diff().abs()   # 일별 가격차이의 절대값 계산
        df['diff_positive'] = df['diff'].mask(df['diff']<0, 0)   # 가격차이가 +인 경우만 발라내기
        weight = pd.array(range(1, w+1))
        df['weighted_diff_abs_rolling_sum'] = df['diff_abs'].rolling(w).apply(lambda x: np.dot(x,weight))   # WRSI 분모
        df['weighted_diff_positive_rolling_sum'] = df['diff_positive'].rolling(w).apply(lambda x: np.dot(x,weight))   # WRSI 분자
        df['wrsi'] = df['weighted_diff_positive_rolling_sum'].truediv(df['weighted_diff_abs_rolling_sum'], fill_value=.5)*100   # WRSI
        df.drop(columns=['diff', 'diff_abs','diff_positive','weighted_diff_abs_rolling_sum', 'weighted_diff_positive_rolling_sum'], inplace=True)
        return df['wrsi'].round(2)
    else:
        return None


class Trend():

    def RSI(self, df, period, base_date):
        rsi_df = df.dropna()
        rsi_df['diff'] = rsi_df[cd] - rsi_df[cd].shift(1)
        for p in rsi_df.iloc[period:].index:
            d, ad, u, au = 0, 0., 0, 0.
            for i in range(period):
                diff = rsi_df.shift(i).loc[p, 'diff']
                if diff >= 0:
                    u += 1
                    au += diff
                elif diff < 0:
                    d += 1
                    ad -= diff
            if not au + ad == 0:
                rsi = round (au / (au + ad), 4) * 100
            else:
                rsi = 50
            rsi_df.loc[p, 'RSI'+str(period)] = rsi
        return (rsi_df[base_date:])

    def RSI_old(self, df, period, base_date):
        rsi_df = df.dropna()
        rsi_df['diff'] = rsi_df[cd] - rsi_df[cd].shift(1)
        for p in rsi_df.iloc[period:].index:
            d, ad, u, au = 0, 0., 0, 0.
            for i in range(period):
                diff = rsi_df.shift(i).loc[p, 'diff']
                if diff >= 0:
                    u += 1
                    au += diff
                elif diff < 0:
                    d += 1
                    ad -= diff
            if not au + ad == 0:
                rsi = round (au / (au + ad), 4) * 100
            else:
                rsi = 50
            rsi_df.loc[p, 'RSI'+str(period)] = rsi
        return (rsi_df[base_date:])
    
    
    def WRSI(self, df, cd, period, base_date):
        rsi_df = pd.DataFrame()
        rsi_df[cd] = df[cd].copy()
        rsi_df = rsi_df.dropna()
        rsi_df['diff'] = rsi_df[cd] - rsi_df[cd].shift(1)
        for p in rsi_df.iloc[period:].index:
            d, ad, u, au, multiple = 0, 0., 0, 0., 0.
            for i in range(period):
                multiple = (period - i) / period
                diff = rsi_df.shift(i).loc[p, 'diff']
                if diff >= 0:
                    u += 1
                    au += diff * multiple
                elif diff < 0:
                    d += 1
                    ad -= diff * multiple
            if not au + ad == 50:
                rsi = round (au / (au + ad), 4) * 100
            else:
                rsi = 0
            rsi_df.loc[p, 'WRSI'+str(period)] = rsi
        return (rsi_df[base_date:])
