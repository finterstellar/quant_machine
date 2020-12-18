from . import util
import pandas as pd
import pandas_datareader.data as web


def get_price(symbol, start_date=None, end_date=None):
    symbol = util.str_to_list(symbol)
    end_date = pd.to_datetime(end_date).date() if end_date else pd.Timestamp.today().date()
    start_date = pd.to_datetime(start_date).date() if start_date else util.months_before(end_date, 12)
    df = web.DataReader(symbol, 'yahoo', start=start_date, end=end_date)['Adj Close'].round(2)
    return df[symbol]


def price_df_cleanse(df):
    try:
        if 'trade_date' in df.columns:
            df.set_index('trade_date', inplace=True)
        df = df.reindex(df.index, method='ffill')
    except:
        pass
    return df


def sampling_by_date(df, start_date, end_date):
    try:
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()
        df = df.loc[start_date:end_date].copy()
        df.dropna(inplace=True)
        df = df.reindex(df.index, method='ffill')
        return df
    except:
        return None

def sampling_by_period(df, start_pt, period):
    try:
        df = df.iloc[start_pt:start_pt + period].copy()
        return df
    except:
        return None

def sampling_by_code(df, cds):
    try:
        df = df[df.columns.intersection(cds)].copy()
        df.dropna(inplace=True)
        df = df.reindex(df.index, method='ffill')
        return df
    except:
        return None