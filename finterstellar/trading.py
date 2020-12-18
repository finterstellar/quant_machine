import finterstellar as fs
import pandas as pd
import numpy as np


def get_period(df):
    df.dropna(inplace=True)
    end_date = df.index[-1]
    start_date = df.index[0]
    days_between = (end_date - start_date).days
    return abs(days_between)


def annualize(rate, period):
    if period < 365:
        rate = ((rate-1) / period * 365) + 1
    elif period > 365:
        rate = rate ** (365 / period)
    return round(rate, 4)


def create_signal(df, factor, buy, sell):
    df['trade'] = np.nan
    if buy > sell:
        df['trade'].mask(df[factor]>buy, 'buy', inplace=True)
        df['trade'].mask(df[factor]<sell, 'zero', inplace=True)
    else:
        df['trade'].mask(df[factor]<buy, 'buy', inplace=True)
        df['trade'].mask(df[factor]>sell, 'zero', inplace=True)
    df['trade'].fillna(method='ffill', inplace=True)
    df['trade'].fillna('zero', inplace=True)
    return df['trade']


def position(df):
    df['position'] = ''
    df['position'].mask((df['trade'].shift(1)=='zero') & (df['trade']=='zero'), 'zz', inplace=True)
    df['position'].mask((df['trade'].shift(1)=='zero') & (df['trade']=='buy'), 'zl', inplace=True)
    df['position'].mask((df['trade'].shift(1)=='buy') & (df['trade']=='zero'), 'lz', inplace=True)
    df['position'].mask((df['trade'].shift(1)=='buy') & (df['trade']=='buy'), 'll', inplace=True)
    return df['position']


def trade(df, cost=.001):
    df['signal_price'] = np.nan
    df['signal_price'].mask(df['position']=='zl', df.iloc[:,0], inplace=True)
    df['signal_price'].mask(df['position']=='lz', df.iloc[:,0], inplace=True)
    record = df[['position','signal_price']].dropna()
    record['rtn'] = 1
    record['rtn'].mask(record['position']=='lz', (record['signal_price']*(1-cost))/record['signal_price'].shift(1), inplace=True)
    record['acc_rtn'] = record['rtn'].cumprod()
    df['signal_price'].mask(df['position']=='ll', df.iloc[:,0], inplace=True)
    df['rtn'] = record['rtn']
    df['rtn'].fillna(1, inplace=True)
    df['daily_rtn'] = 1
    df['daily_rtn'].mask(df['position'] == 'll', df['signal_price'] / df['signal_price'].shift(1), inplace=True)
    df['daily_rtn'].mask(df['position'] == 'lz', (df['signal_price']*(1-cost)) / df['signal_price'].shift(1), inplace=True)
    df['daily_rtn'].fillna(1, inplace=True)
    df['acc_rtn'] = df['daily_rtn'].cumprod()
    df['mdd'] = (df['acc_rtn'] / df['acc_rtn'].cummax()).round(4)
    df['bm_mdd'] = (df.iloc[:, 0] / df.iloc[:, 0].cummax()).round(4)
    df.drop(columns='signal_price', inplace=True)
    return df


def get_sharpe_ratio(df, rf_rate):
    rf_rate = rf_rate / 365 + 1
    sharpe_ratio = (df['daily_rtn']-rf_rate).mean() / (df['daily_rtn']-rf_rate).std()
    return round(sharpe_ratio, 4)


def evaluate(df, rf_rate, cost):
    rst = {}
    rst['no_trades'] = (df['position']=='lz').sum()
    rst['no_win'] = (df['rtn']>1).sum()
    rst['acc_rtn'] = df['acc_rtn'][-1].round(4)
    rst['hit_ratio'] = round((df['rtn']>1).sum() / rst['no_trades'], 4)
    rst['avg_rtn'] = round(df[df['rtn']>1]['rtn'].mean(), 4)
    rst['period'] = get_period(df)
    rst['annual_rtn'] = annualize(rst['acc_rtn'], rst['period'])
    rst['bm_rtn'] = round(df.iloc[-1,0]/df.iloc[0,0], 4)
    rst['sharpe_ratio'] = get_sharpe_ratio(df, rf_rate)
    rst['mdd'] = df['mdd'].min()
    rst['bm_mdd'] = df['bm_mdd'].min()
    rst['transaction_cost'] = cost

    print('Accumulated return: {:.2%}'.format(rst['acc_rtn'] - 1))
    print('Annualized return : {:.2%}'.format(rst['annual_rtn'] - 1))
    print('Average return: {:.2%}'.format(rst['avg_rtn'] - 1))
    print('Benchmark return : {:.2%}'.format(rst['bm_rtn']-1))
    print('Number of trades: {}'.format(rst['no_trades']))
    print('Number of win: {}'.format(rst['no_win']))
    print('Hit ratio: {:.2%}'.format(rst['hit_ratio']))
    print('Investment period: {:.2f}yrs'.format(rst['period']/365))
    print('Sharpe ratio: {:.2%}'.format(rst['sharpe_ratio']))
    print('MDD: {:.2%}'.format(rst['mdd']-1))
    print('Benchmark MDD: {:.2%}'.format(rst['bm_mdd']-1))
    return rst


    
class Trade:

    def standardize(self, prices_df, base_date, codes):
        for c in codes:
            std = prices_df[c] / prices_df.loc[base_date][c] * 100
            prices_df[c+' idx'] = round(std, 4)
        return (prices_df)
    

    def create_trade_book(self, sample, s_cd):
        
        cds = fs.str_list(s_cd)
        
        book = pd.DataFrame()
        book[cds] = sample[cds]

        for c in cds:
            book['high_'+c] = book[c].cummax()
            book['low_'+c] = book[c].cummin()
            book['t '+c] = ''
            book['p '+c] = ''

        return (book)    
    
    
    def position(self, book, s_cd):
        
        cds = fs.str_list(s_cd)
        
        for c in cds:
            status = ''
            p = 0
            for i in book.index:
                if p == 0:
                    status_prev = 'z'
                else:
                    prev = book.index[p-1]
                    if book.loc[prev, 't '+c] == 'buy':
                        status_prev = 'l'
                    elif book.loc[prev, 't '+c] == '':
                        status_prev = 'z'
                    elif book.loc[prev, 't '+c] == 'sell':
                        status_prev = 's'
                    else:
                        status_prev = 'z'

                if book.loc[i, 't '+c] == 'buy':
                    status_now = 'l'
                elif book.loc[i, 't '+c] == '':
                    status_now = 'z'
                elif book.loc[i, 't '+c] == 'sell':
                    status_now = 's'
                else:
                    status_now = 'z'

                status = status_prev + status_now

                book.loc[i, 'p '+c] = status
                p += 1
        return book


    def position_old(self, book, s_cd):

        cds = fs.str_list(s_cd)

        for c in cds:
            status = ''
            for i in book.index:
                if book.loc[i, 't ' + c] == 'buy':
                    if book.shift(1).loc[i, 't ' + c] == 'buy':
                        status = 'll'
                    elif book.shift(1).loc[i, 't ' + c] == '':
                        status = 'zl'
                    elif book.shift(1).loc[i, 't ' + c] == 'sell':
                        status = 'sl'
                    else:
                        status = 'zl'
                elif book.loc[i, 't ' + c] == 'sell':
                    if book.shift(1).loc[i, 't ' + c] == 'buy':
                        status = 'ls'
                    elif book.shift(1).loc[i, 't ' + c] == '':
                        status = 'zs'
                    elif book.shift(1).loc[i, 't ' + c] == 'sell':
                        status = 'ss'
                    else:
                        status = 'zs'
                elif book.loc[i, 't ' + c] == '':
                    if book.shift(1).loc[i, 't ' + c] == 'buy':
                        status = 'lz'
                    elif book.shift(1).loc[i, 't ' + c] == '':
                        status = 'zz'
                    elif book.shift(1).loc[i, 't ' + c] == 'sell':
                        status = 'sz'
                    else:
                        status = 'zz'
                else:
                    status = 'zz'
                book.loc[i, 'p ' + c] = status
        return (book)


    def position_strategy(self, book, s_cd, last_date, report_name='', report={}):
        
        cds = fs.str_list(s_cd)
        
        strategy = ''
        strategy_x = ''
        strategy_y = ''
        for c in cds:
            i = book.index[-1]
            if book.loc[i, 'p '+c] == 'lz' or book.loc[i, 'p '+c] == 'sz' or book.loc[i, 'p '+c] == 'zz':
                strategy = 'Zero '+c
            elif book.loc[i, 'p '+c] == 'll' or book.loc[i, 'p '+c] == 'sl' or book.loc[i, 'p '+c] == 'zl':
                strategy = 'Long '+c
            elif book.loc[i, 'p '+c] == 'ls' or book.loc[i, 'p '+c] == 'ss' or book.loc[i, 'p '+c] == 'zs':
                strategy = 'Short '+c
            if c == cds[0]:
                strategy_x = strategy.split(' ')[0]
            else:
                strategy_y = strategy.split(' ')[0]
        strategy = cds[0] + ': ' + strategy_x 
        if len(cds) > 1:
            strategy = strategy + ' & ' + cds[1] + ': ' + strategy_y
        #print ('As of', last_date, 'your model portfolio', cds,'needs to be composed of', strategy)
        if not report == {}:
            report['last_date'] = last_date
            report['position_strategy_x'] = strategy_x
            report['position_strategy_y'] = strategy_y

        return strategy

 
    # 수익률
    def returns(self, book, s_cd, display=False, report_name='', report={}, fee=0.0):
        # 손익 계산
        cds = fs.str_list(s_cd)
           
        rtn = 1.0
        book['return'] = 1
        no_trades = 0
        no_win = 0
        
        for c in cds:
            buy = 0.0
            sell = 0.0
            for i in book.index:
            
                if book.loc[i, 'p '+c] == 'zl' or book.loc[i, 'p '+c] == 'sl' :     # long 진입
                    buy = book.loc[i, c]
                    if fee > 0.0:
                        buy = round(buy * (1 + fee), 3)
                    if display:
                        print(i, 'long '+c, buy)
                    
                elif book.loc[i, 'p '+c] == 'lz' or book.loc[i, 'p '+c] == 'ls' :     # long 청산
                    sell = book.loc[i, c]
                    if fee > 0.0:
                        sell = round(sell * (1 - fee), 3)
                    # 손익 계산
                    rtn = sell / buy
                    book.loc[i, 'return'] = rtn
                    no_trades += 1
                    if rtn > 1:
                        no_win += 1
                    if display:
                        print(i, 'long '+c, buy, ' | unwind long '+c, sell, ' | return: %.2f' % (rtn-1))
                    
                elif book.loc[i, 'p '+c] == 'zs' or book.loc[i, 'p '+c] == 'ls' :     # short 진입
                    sell = book.loc[i, c]
                    if fee > 0.0:
                        sell = sell * (1 - fee)
                    if display:
                        print(i, 'short '+c, sell)
                elif book.loc[i, 'p '+c] == 'sz' or book.loc[i, 'p '+c] == 'sl' :     # short 청산
                    buy = book.loc[i, c]
                    if fee > 0.0:
                        buy = buy * (1 + fee)
                    # 손익 계산
                    rtn = buy / sell
                    book.loc[i, 'return'] = rtn
                    no_trades += 1
                    if rtn > 1:
                        no_win += 1
                    if display:
                        print(i, 'short '+c, sell, ' | unwind short '+c, buy, ' | return: %.2f' % (rtn-1))
                
            if book.loc[i, 't '+c] == '' and book.loc[i, 'p '+c] == '':     # zero position
                buy = 0.0
                sell = 0.0

        # Accumulated return
        acc_rtn = 1.0
        for i in book.index:
            rtn = book.loc[i, 'return']
            acc_rtn = acc_rtn * rtn
            book.loc[i, 'acc return'] = acc_rtn

        try:
            first_day = pd.to_datetime(book.index[0])
            last_day = pd.to_datetime(book.index[-1])
            total_days = (last_day - first_day).days
            annualizer = total_days / 365
        except:
            annualizer = 1 / 252

        print (fs.FontStyle.bg_white+'Accumulated return:', round((acc_rtn - 1) * 100, 2), '%'+fs.FontStyle.end_bg, \
               ' ( # of trade:', no_trades, ', # of win:', no_win, ', fee: %.2f' % (fee*100), \
               '%,', 'period: %.2f' % annualizer, 'yr )')

        if no_trades > 0:
            avg_rtn = acc_rtn ** (1 / no_trades)
            prob_win = round((no_win / no_trades), 4)
        else:
            avg_rtn = 1.0
            prob_win = 0.0
        avg_rtn = round(avg_rtn, 4)

        bet = fs.Bet()
        kelly_ratio = bet.kelly_formular(prob_win)
        kelly_ratio = round(kelly_ratio, 4)
        
        print('Avg return: %.2f' % ((avg_rtn-1)*100), end=' %')
        if prob_win > 0.5:
            print(fs.FontStyle.orange, end='')
        print(', Prob. of win: %.2f' % (prob_win*100), end=' %')
        if prob_win > 0.5:
            print(fs.FontStyle.end_c, end='')
        print(', Kelly ratio: %.2f' % (kelly_ratio*100), end=' %')

        mdd = round((book['return'].min()), 4)
        print(', MDD: %.2f' % ((mdd-1)*100), '%')

        if not report == {}:
            report['acc_rtn'] = round((acc_rtn) * 100, 2)
            report['no_trades'] = no_trades
            report['avg_rtn'] = round((avg_rtn * 100), 2)
            report['prob_win'] = round((prob_win * 100), 2)
            report['kelly_ratio'] = round((kelly_ratio * 100), 2)
            report['fee'] = round((fee * 100), 2)
            report['mdd'] = round((mdd * 100), 2)

        return round(acc_rtn, 4)

    
    # 벤치마크 수익
    def benchmark_return(self, book, s_cd, report_name='', report={}):
        # 벤치마크 수익률
        
        cds = fs.str_list(s_cd)
        
        n = len(cds)
        rtn = dict()
        bm_rtn = float()
        mdd = dict()
        bm_mdd = float()
        for c in cds:
            rtn[c] = round(book[c].iloc[-1] / book[c].iloc[0], 4)
            bm_rtn += rtn[c]/n
            mdd[c] = round( min( book['low_'+c]/ book['high_'+c]), 4)
            bm_mdd += mdd[c]/n
        print('BM return:', round((bm_rtn-1) * 100, 2), '%', rtn, end=' / ')
        print('BM MDD:', round((bm_mdd-1) * 100, 2), '%', mdd)

        if not report == {}:
            report['BM_rtn'] = (round((acc_rtn) * 100, 2))
            report['BM_rtn_A'] = (round(rtn[cds[0]] * 100, 2))
            report['BM_rtn_B'] = (round(rtn[cds[1]] * 100, 2))
            report['BM_mdd'] = (round((acc_mdd) * 100, 2))
        
        return round(bm_rtn, 4)


    # 초과수익률
    def excess_return(self, fund_rtn, bm_rtn, report_name='', report={}):
        exs_rtn = fund_rtn - bm_rtn
        print('Excess return: %.2f' % (exs_rtn*100), '%', \
              ' ( %.2f' % ((fund_rtn-1)*100), '- %.2f' % ((bm_rtn-1)*100), ')' )
        if not report == {}:
            report['excess_rtn'] = round(exs_rtn * 100, 2)
        
        return exs_rtn


    def annualize(self, book, cd):

        first_day = pd.to_datetime(book.index[0])
        last_day = pd.to_datetime(book.index[-1])
        total_days = (last_day - first_day).days
        if total_days < 1:
            total_days = 1
        annualizer = total_days / 365

        acc_return = book['acc return'][-1] - 1
        total_bm_return = book[cd][-1] / book[cd][0] - 1
        if annualizer >= 1:
            annual_return = (acc_return + 1) ** (1 / annualizer) - 1
            annual_bm_return = (total_bm_return + 1) ** (1 / annualizer) - 1
        else:
            annual_return = acc_return * (1 / annualizer)
            annual_bm_return = total_bm_return * (1 / annualizer)
        annual_return = round(annual_return, 4)
        annual_bm_return = round(annual_bm_return, 4)

        print(fs.FontStyle.bg_white+'CAGR: %.2f' % ((annual_return) * 100), end=' %'+fs.FontStyle.end_bg)
        print(', Annual BM return: %.2f' % ((annual_bm_return) * 100), end=' %')
        print(', Annual excess return: %.2f' % ((annual_return - annual_bm_return) * 100), '%')

        return annual_return


    def annualize_return(self, book, fund_rtn, bm_rtn):

        first_day = pd.to_datetime(book.index[0])
        last_day = pd.to_datetime(book.index[-1])
        total_days = (last_day - first_day).days
        if total_days < 1:
            total_days = 1
        annualizer = total_days / 365

        acc_return = fund_rtn - 1
        bm_return = bm_rtn - 1
        if annualizer >= 1:
            annual_return = (acc_return + 1) ** (1 / annualizer) - 1
            annual_bm_return = (bm_return + 1) ** (1 / annualizer) - 1
        else:
            annual_return = acc_return * (1 / annualizer)
            annual_bm_return = bm_return * (1 / annualizer)
        annual_return = round(annual_return, 4)
        annual_bm_return = round(annual_bm_return, 4)

        print(fs.FontStyle.bg_white+'CAGR: %.2f' % ((annual_return) * 100), end=' %'+fs.FontStyle.end_bg)
        print(', Annual BM return: %.2f' % ((annual_bm_return) * 100), end=' %')
        print(', Annual excess return: %.2f' % ((annual_return - annual_bm_return) * 100), '%')

        return annual_return


    # 수익률(로그)
    def returns_log(self, book, s_cd, display=False):
        # 손익 계산
        
        cds = fs.str_list(s_cd)
            
        rtn = 0.0
        book['return'] = rtn
        
        for c in cds:
            buy = 0.0
            sell = 0.0
            for i in book.index:
            
                if book.loc[i, 'p '+c] == 'zl' or book.loc[i, 'p '+c] == 'sl' :     # long 진입
                    buy = book.loc[i, c]
                    if display:
                        print(i.date(), 'long '+c, buy)
                elif book.loc[i, 'p '+c] == 'lz' or book.loc[i, 'p '+c] == 'ls' :     # long 청산
                    sell = book.loc[i, c]
                    # 손익 계산
                    rtn = np.log(sell / buy) * 100
                    #(sell - buy) / buy + 1
                    book.loc[i, 'return'] = rtn
                    if display:
                        print(i.date(), 'long '+c, buy, ' | unwind long '+c, sell, ' | return:', round(rtn, 4))
                    
                elif book.loc[i, 'p '+c] == 'zs' or book.loc[i, 'p '+c] == 'ls' :     # short 진입
                    sell = book.loc[i, c]
                    if display:
                        print(i.date(), 'short '+c, sell)
                elif book.loc[i, 'p '+c] == 'sz' or book.loc[i, 'p '+c] == 'sl' :     # short 청산
                    buy = book.loc[i, c]
                    # 손익 계산
                    rtn = np.log(sell / buy) * 100
                    book.loc[i, 'return'] = rtn
                    if display:
                        print(i.date(), 'short '+c, sell, ' | unwind short '+c, buy, ' | return:', round(rtn, 4))
                
            if book.loc[i, 't '+c] == '' and book.loc[i, 'p '+c] == '':     # zero position
                buy = 0.0
                sell = 0.0
        
        acc_rtn = 0.0
        for i in book.index:
            rtn = book.loc[i, 'return']
            acc_rtn = acc_rtn + rtn
            book.loc[i, 'acc return'] = acc_rtn
            
        print('Accunulated return:', round(acc_rtn, 2), '%')
        return round(acc_rtn, 4)

    
    # 벤치마크수익률(로그)
    def benchmark_return_log(self, book, s_cd):
        # 벤치마크 수익률
        
        cds = fs.str_list(s_cd)
        
        n = len(cds)
        rtn = dict()
        acc_rtn = float()
        for c in cds:
            rtn[c] = round ( np.log(book[c].iloc[-1] / book[c].iloc[0]) * 100 , 4)   
            acc_rtn += rtn[c]/n
        print('BM return:', round(acc_rtn, 2), '%')
        print(rtn)
        return (round(acc_rtn, 4))
    
    
    # 초과수익률(로그)
    def excess_return_log(self, fund_rtn, bm_rtn):
        exs_rtn = fund_rtn - bm_rtn
        print('Excess return:', round(exs_rtn, 2), '%')
        return (exs_rtn)


class SingleAsset(Trade):

    def BB_hyper_trading(self, sample, book, cd, buy_when='in', \
                         bb_sell=0.5, trend_sell=0.9, \
                         go_velocity=10, go_volume=5):

        print(fs.FontStyle.green + 'BB hyper trading : Trend + BB w go_veolcity, go_volume' + fs.FontStyle.end_c)

        trade, trade_prev, cause, cause_prev = '', '', '', ''

        for i in sample.index:
            # price = sample.loc[i, cd]
            p_score = sample.loc[i, 'pct_b']
            v_chg = sample.loc[i, 'volume_chg']
            book.loc[i, 'size_chg'] = sample.loc[i, 'size_chg']

            # Zone 1
            if p_score > 1:
                if sample.loc[i, 'size_chg'] > go_velocity and v_chg > go_volume:
                    trade = 'buy'
                    cause = 'trend'
                else:
                    if trade_prev == 'buy':
                        trade = 'buy'
                        cause = 'trend'
                    else:
                        trade = ''
                        cause = ''

            # Zone 2
            elif 1 > p_score >= 0:
                # trend sell
                if cause_prev == 'trend':
                    if p_score >= trend_sell:
                        if trade_prev == 'buy':
                            trade = 'buy'
                            cause = 'trend'
                        else:
                            trade = ''
                            cause = ''
                    else:
                        trade = ''
                        cause = ''
                if cause_prev == 'bb':
                    if p_score > bb_sell:
                        trade = ''
                        cause = ''
                    else:
                        if trade_prev == 'buy' or 'ready':
                            trade = 'buy'
                            cause = 'bb'
                        else:
                            trade = ''
                            cause = ''

            # Zone 3
            else:
                cause = 'bb'
                if buy_when == 'in':  # 밴드 진입 시 매수
                    trade = 'ready'  # 대기
                else:
                    trade = 'buy'

            book.loc[i, 't ' + cd] = trade
            book.loc[i, 'c ' + cd] = cause
            trade_prev = trade
            cause_prev = cause

        return book


    def BB_hyper_trading_wo_volume(self, sample, book, cd, buy_when='in', \
                         bb_sell=0.5, trend_sell=0.9):

        print(fs.FontStyle.green + 'BB hyper trading wo volume : Trend + BB' + fs.FontStyle.end_c)

        trade, trade_prev, cause, cause_prev = '', '', '', ''

        for i in sample.index:
            #price = sample.loc[i, cd]
            p_score = sample.loc[i, 'pct_b']

            # Zone 1
            if p_score > 1:
                trade = 'buy'
                cause = 'trend'
                
            # Zone 2
            elif 1 > p_score >= 0:
                # trend sell
                if cause_prev == 'trend':
                    if p_score >= trend_sell:
                        if trade_prev == 'buy':
                            trade = 'buy'
                            cause = 'trend'
                        else:
                            trade = ''
                            cause = ''
                    else:
                        trade = ''
                        cause = ''
                if cause_prev == 'bb':
                    if p_score > bb_sell:
                        trade = ''
                        cause = ''
                    else:
                        if trade_prev == 'buy' or 'ready':
                            trade = 'buy'
                            cause = 'bb'
                        else:
                            trade = ''
                            cause = ''
                    
            # Zone 3
            else:
                cause = 'bb'
                if buy_when == 'in':   # 밴드 진입 시 매수
                    trade = 'ready'    # 대기
                else:
                    trade = 'buy'

            book.loc[i, 't ' + cd] = trade
            book.loc[i, 'c ' + cd] = cause
            trade_prev = trade
            cause_prev = cause
                
        return book
 

    def BB_trend_trading(self, sample, book, cd, trend_buy=1., trend_sell=.9, go_velocity=10):

        print(fs.FontStyle.green + 'BB trend trading : Trend w go_veolcity' + fs.FontStyle.end_c)

        p = 0
        for i in sample.index:
            if not p == 0:
                prev = book.index[p - 1]
            else:
                prev = book.index[0]

            p_score = sample.loc[i, 'pct_b']

            if trend_buy < p_score and sample.loc[i, 'size_chg'] > go_velocity:
                book.loc[i, 't '+cd] = 'buy'
            elif p_score < trend_sell:
                book.loc[i, 't '+cd] = ''
            else:
                if book.loc[prev, 't ' + cd] == 'buy':
                    book.loc[i, 't '+cd] = 'buy'
                else:
                    book.loc[i, 't '+cd] = ''
            p += 1
        return book


    def BB_trend_volume_trading(self, sample, book, cd, trend_buy=1., trend_sell=.8, go_velocity=10, go_volume=5):

        print(fs.FontStyle.green + 'BB trend trading : Trend w go_veolcity, go_volume' + fs.FontStyle.end_c)

        p = 0
        for i in sample.index:
            if not p == 0:
                prev = book.index[p - 1]
            else:
                prev = book.index[0]

            p_score = sample.loc[i, 'pct_b']
            v_chg = sample.loc[i, 'volume_chg']

            if trend_buy < p_score and sample.loc[i, 'size_chg'] > go_velocity and v_chg > go_volume:
                book.loc[i, 't ' + cd] = 'buy'
            elif p_score < trend_sell:
                book.loc[i, 't ' + cd] = ''
            else:
                if book.loc[prev, 't ' + cd] == 'buy':
                    book.loc[i, 't ' + cd] = 'buy'
                else:
                    book.loc[i, 't ' + cd] = ''
            p += 1
        return book


    def BB_traditional_trading(self, sample, book, cd, buy_when='in', bb_sell='center', short=False):

        print(fs.FontStyle.green + 'BB traditional trading : BB' + fs.FontStyle.end_c)

        for i in sample.index:
            price = sample.loc[i, cd]

            if short:

                if price > sample.loc[i, 'ub']:
                    if book.shift(1).loc[i, 't ' + cd] == 'sell':  # 이미 매수상태라면
                        book.loc[i, 't ' + cd] = 'sell'  # 매수상태 유지
                    else:
                        if buy_when == 'in':  # 밴드 진입 시 매수
                            book.loc[i, 't ' + cd] = ''  # 대기
                        else:
                            book.loc[i, 't ' + cd] = 'sell'  # 매수

                elif sample.loc[i, 'ub'] >= price > sample.loc[i, 'center']:
                    if buy_when == 'out':
                        if book.shift(1).loc[i, 't ' + cd] == 'sell':  # 숏 유지
                            book.loc[i, 't ' + cd] = 'sell'
                        elif book.shift(1).loc[i, 't ' + cd] == 'buy':  # 롱 청산
                            book.loc[i, 't ' + cd] = ''
                        else:
                            book.loc[i, 't ' + cd] = ''
                    else:
                        if book.shift(1).loc[i, 't ' + cd] == 'sell' or book.shift(1).loc[i, 't ' + cd] == '':
                            book.loc[i, 't ' + cd] = 'sell'
                        elif book.shift(1).loc[i, 't ' + cd] == 'buy':  # 롱 청산
                            book.loc[i, 't ' + cd] = ''
                        else:
                            book.loc[i, 't ' + cd] = ''

                elif sample.loc[i, 'center'] >= price > sample.loc[i, 'lb']:
                    if buy_when == 'out':
                        if book.shift(1).loc[i, 't ' + cd] == 'sell':  # 숏 청산
                            book.loc[i, 't ' + cd] = ''
                        elif book.shift(1).loc[i, 't ' + cd] == 'buy':  # 롱 유지
                            book.loc[i, 't ' + cd] = 'buy'
                        else:
                            book.loc[i, 't ' + cd] = ''
                    else:
                        if book.shift(1).loc[i, 't ' + cd] == 'sell':  # 숏 청산
                            book.loc[i, 't ' + cd] = ''
                        elif book.shift(1).loc[i, 't ' + cd] == 'buy' or book.shift(1).loc[i, 't ' + cd] == '':
                            book.loc[i, 't ' + cd] = 'buy'
                        else:
                            book.loc[i, 't ' + cd] = ''

                elif sample.loc[i, 'lb'] >= price:
                    if book.shift(1).loc[i, 't ' + cd] == 'buy':  # 이미 매수상태라면
                        book.loc[i, 't ' + cd] = 'buy'  # 매수상태 유지
                    else:
                        if buy_when == 'in':  # 밴드 진입 시 매수
                            book.loc[i, 't ' + cd] = ''  # 대기
                        else:
                            book.loc[i, 't ' + cd] = 'buy'  # 매수

            else:

                if price > sample.loc[i, bb_sell]:
                    book.loc[i, 't ' + cd] = ''
                elif sample.loc[i, bb_sell] >= price >= sample.loc[i, 'lb']:
                    if buy_when == 'in':
                        if book.shift(1).loc[i, 't ' + cd] == 'buy' or book.shift(1).loc[
                            i, 't ' + cd] == 'ready':
                            # 이미 매수상태 또는 Ready에서 넘어온 상태
                            book.loc[i, 't ' + cd] = 'buy'  # trade : buy (매수상태 유지)
                        else:
                            book.loc[i, 't ' + cd] = ''  # trade : clear (zero상태 유지)
                    else:
                        if book.shift(1).loc[i, 't ' + cd] == 'buy':
                            book.loc[i, 't ' + cd] = 'buy'
                        else:
                            book.loc[i, 't ' + cd] = ''
                elif sample.loc[i, 'lb'] > price:
                    if buy_when == 'in':
                        if book.shift(1).loc[i, 't ' + cd] == 'buy':
                            book.loc[i, 't ' + cd] = 'buy'  # 이미 buy
                        else:
                            book.loc[i, 't ' + cd] = 'ready'
                    else:
                        book.loc[i, 't ' + cd] = 'buy'  # 매수상태 유지

        return book

    def BB_hyper_trading_strategy(self, sample, cd, bb_sell, last_date):

        i = sample.index[-1]
        if sample.loc[i, cd] >= sample.loc[i, bb_sell]:
            strategy = ''
        elif sample.loc[i, cd] <= sample.loc[i, 'lb']:
            strategy = 'buy ' + cd
        else:
            strategy = 'just wait'
        print('As of', last_date, ', this model suggests you to', strategy)
        return (strategy)


    def WRSI_trading(self, sample, book, cd, wrsi_buy=70, wrsi_sell=50):            
        p = 0
        for i in sample.index:
            if not p == 0:
                prev = book.index[p - 1]
            else:
                prev = book.index[0]

            wrsi = sample.loc[i, 'WRSI']
            
            if wrsi > wrsi_buy:
                book.loc[i, 't '+cd] = 'buy'
            elif wrsi < wrsi_sell:
                book.loc[i, 't '+cd] = ''
            else:
                if book.loc[prev, 't '+cd] == 'buy':
                    book.loc[i, 't '+cd] = 'buy'
                else:
                    book.loc[i, 't '+cd] = ''
            p += 1
        return (book) 

        
    def WRSI_trading_reverse(self, sample, book, cd, wrsi_buy=30, wrsi_sell=70):            
        p = 0
        for i in sample.index:
            if not p == 0:
                prev = book.index[p - 1]
            else:
                prev = book.index[0]

            wrsi = sample.loc[i, 'WRSI']
            
            if wrsi < wrsi_buy:
                book.loc[i, 't '+cd] = 'buy'
            elif wrsi > wrsi_sell:
                book.loc[i, 't '+cd] = ''
            else:
                if book.loc[prev, 't '+cd] == 'buy':
                    book.loc[i, 't '+cd] = 'buy'
                else:
                    book.loc[i, 't '+cd] = ''
            p += 1
        return (book) 
 

    def trend_tradings(self, sample, book, cd, f1, c1, f2, c2):   
        for i in sample.index:   # 데이터프레임을 하나씩 읽어가며
            if sample.loc[i, f1] >= c1:   # 요인1(f1) 값이 지표1(c1)보다 크면
                book.loc[i, 't '+cd] = 'buy'    # buy
                if sample.loc[i, f2] < c2:    # 요인2(f2) 값이 지표2(c2)보다 작으면
                    book.loc[i, 't '+cd] = ''    # clear(잔고청산)
            else:    # 위 판단지표에 해당하지 않으면
                book.loc[i, 't '+cd] = ''    # clear
        return book


    def trend_tradings_reverse(self, sample, book, cd, f1, c1, f2, c2):   
        p = 0
        for i in sample.index:   # 데이터프레임을 하나씩 읽어가며
            if not p == 0:
                prev = book.index[p - 1]
            else:
                prev = book.index[0]

            if sample.loc[i, f1] < c1:   # 요인1(f1) 값이 지표1(c1)보다 면
                book.loc[i, 't '+cd] = 'buy'    # buy
            elif sample.loc[i, f2] > c2:    # 요인2(f2) 값이 지표2(c2)보다 작으면
                book.loc[i, 't '+cd] = ''    # clear(잔고청산)
            else:    # 위 판단지표에 해당하지 않으면
                if book.loc[prev, 't '+cd] == 'buy':
                    book.loc[i, 't '+cd] = 'buy'
                else:
                    book.loc[i, 't '+cd] = ''
            p += 1
        return book

    
    def stochastic_trading(self, sample, book, cd, osi='slow'):            
        p = 0
        for i in sample.index:
            if not p == 0:
                prev = book.index[p - 1]
            else:
                prev = book.index[0]

            slow_k = sample.loc[i, 'slow_k']
            slow_d = sample.loc[i, 'slow_d']
            
            if slow_d < slow_k:
                book.loc[i, 't '+cd] = 'buy'
            elif slow_d >= slow_k:
                book.loc[i, 't '+cd] = ''
            else:
                if book.loc[prev, 't '+cd] == 'buy':
                    book.loc[i, 't '+cd] = 'buy'
                else:
                    book.loc[i, 't '+cd] = ''
            p += 1

        return (book) 
 
    
    def stochastic_trading_reverse(self, sample, book, cd, osi='slow'):
        p = 0
        for i in sample.index:
            if not p == 0:
                prev = book.index[p - 1]
            else:
                prev = book.index[0]

            slow_k = sample.loc[i, 'slow_k']
            slow_d = sample.loc[i, 'slow_d']
            
            if slow_d > slow_k:
                book.loc[i, 't '+cd] = 'buy'
            elif slow_d <= slow_k:
                book.loc[i, 't '+cd] = ''
            else:
                if book.loc[prev, 't '+cd] == 'buy':
                    book.loc[i, 't '+cd] = 'buy'
                else:
                    book.loc[i, 't '+cd] = ''
            p += 1
        return (book) 
 



class MultiAsset(Trade):
    
    pass



class PairTrade(Trade):
    
    def regression(self, sample, s_codes):
        sample.dropna(inplace=True)
        from sklearn.linear_model import LinearRegression
        x = sample[s_codes[0]]
        y = sample[s_codes[1]]
        # 1개 컬럼 np.array로 변환
        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        # Linear Regression
        regr = LinearRegression()
        regr.fit(x, y)
        result = {'Slope':regr.coef_[0,0], 'Intercept':regr.intercept_[0], 'R2':regr.score(x, y) }
        #result = {'Slope':regr.coef_, 'Intercept':regr.intercept_, 'R2':regr.score(x, y) }
        return(result)
    
    
    def compare_r2(self, prices_df, base_date, s_codes):
        comp_df = pd.DataFrame()
        s_df = self.sampling(prices_df, base_date, s_codes)
        s_df = s_df.dropna()
        n = len(s_codes)
        for i in range(0, n, 1):
            for j in range(i, n, 1):
                if i != j:
                    code_pairs = [ s_codes[i], s_codes[j] ]
                    regr = self.regression(s_df, code_pairs)
                    c_pair = s_codes[i]+' vs. '+s_codes[j]
                    #print(s_codes[i], '-', s_codes[j], ' : ', '{:,.2f}'.format(regr['R2']*100))
                    comp_df.loc[c_pair, 'R2'] = round(regr['R2'], 4) * 100
                    comp_df.loc[c_pair, 'Slope'] = round(regr['Slope'], 4)
                    comp_df.loc[c_pair, 'Correlation'] = s_df[code_pairs].corr(method='pearson', min_periods=1).iloc[1,0]
        comp_df.index.name = 'pair'
        comp_df = comp_df.sort_values(by='R2', ascending=False)
        return (comp_df)

    
        
    def expected_y(self, sample, regr, s_codes):
        sample[s_codes[1]+' expected'] = sample[s_codes[0]] * regr['Slope'] + regr['Intercept']
        sample[s_codes[1]+' spread'] = sample[s_codes[1]] - sample[s_codes[1]+' expected']
        return (sample)
    
    
    def price_analyze(self, sample, thd, s_codes):
        for i in sample.index:
            threshold = float( thd * sample.loc[i, s_codes[1]] )
            if sample.loc[i, s_codes[1]+' spread'] > threshold:
                sample.loc[i, 'cheaper'] = s_codes[0]
            elif sample.loc[i, s_codes[1]+' spread'] < -threshold:
                sample.loc[i, 'cheaper'] = s_codes[1]
            else:
                sample.loc[i, 'cheaper'] = 'E'
        print(sample.groupby('cheaper').count())
        return (sample)
    
    
    def trading(self, sample, book, thd, s_codes, short=False):
        for i in sample.index:
            #threshold = float( thd * sample.loc[i, s_codes[1]] )
            threshold = float( thd * sample.loc[i, s_codes[1]+' expected'] )
            if sample.loc[i, s_codes[1]+' spread'] > threshold:
                book.loc[i, 't '+s_codes[0]] = 'buy'
                if short:
                    book.loc[i, 't '+s_codes[1]] = 'sell'
                else:
                    book.loc[i, 't '+s_codes[1]] = ''
            elif threshold >= sample.loc[i, s_codes[1]+' spread'] >= 0:
                book.loc[i, 't '+s_codes[0]] = ''
                book.loc[i, 't '+s_codes[1]] = ''
            elif 0 > sample.loc[i, s_codes[1]+' spread'] >= -threshold:
                book.loc[i, 't '+s_codes[0]] = ''
                book.loc[i, 't '+s_codes[1]] = ''
            elif -threshold > sample.loc[i, s_codes[1]+' spread']:
                if short:
                    book.loc[i, 't '+s_codes[0]] = 'sell'
                else:
                    book.loc[i, 't '+s_codes[0]] = ''
                book.loc[i, 't '+s_codes[1]] = 'buy'       
        return (book) 
    


    def trading_strategy(self, sample, thd, s_codes, last_date, short=False, report={}):
        i = sample.index[-1]
        #threshold = float( thd * sample.loc[i, s_codes[1]] )
        threshold = float( thd * sample.loc[i, s_codes[1]+' expected'] )
        if sample.loc[i, s_codes[1]+' spread'] > threshold:
            strategy_x = 'Long '+s_codes[0]
            if short:
                strategy_y = 'Short '+s_codes[1]
            else:
                strategy_y = 'Clear '+s_codes[1]
        elif threshold >= sample.loc[i, s_codes[1]+' spread'] >= 0:
            strategy_x = 'Clear '+s_codes[0]
            strategy_y = 'Clear '+s_codes[1]
        elif 0 > sample.loc[i, s_codes[1]+' spread'] >= -threshold:
            strategy_x = 'Clear '+s_codes[0]
            strategy_y = 'Clear '+s_codes[1]
        elif -threshold > sample.loc[i, s_codes[1]+' spread']:
            strategy_y = 'Long '+s_codes[1]
            if short:
                strategy_x = 'Short '+s_codes[0]
            else:
                strategy_x = 'Clear '+s_codes[0]
        strategy = strategy_x + ' & ' + strategy_y
        #print ('As of', last_date, 'the model suggests you to', strategy)
        if not report == {}:
            report['trading_strategy_x'] = strategy_x.split(' ')[0]
            report['trading_strategy_y'] = strategy_y.split(' ')[0]
                
        present_price = sample.loc[i, s_codes[1]]
        expected_price = round(sample.loc[i, s_codes[1]+' expected'], 2)
        range_cap = round(expected_price * (1+thd), 2)
        range_floor = round(expected_price * (1-thd), 2)
        '''
        print ('The price of', s_codes[1], 'is', str(format(present_price, ',')), 'while expected price is', str(format(expected_price, ',')), \
               ', ranging from', str(format(range_floor, ',')), 'to', str(format(range_cap, ',')))
        '''
        if not report == {}:
            report['present_price'] = present_price
            report['expected_price'] = expected_price
            report['range_floor'] = range_floor
            report['range_cap'] = range_cap
      
        return (strategy) 

    
    def trading_inverse(self, sample, book, thd, s_codes, short=False):
        for i in sample.index:
            #threshold = float( thd * sample.loc[i, s_codes[1]] )
            threshold = float( thd * sample.loc[i, s_codes[1]+' expected'] )
            if sample.loc[i, s_codes[1]+' spread'] > threshold:
                book.loc[i, 't '+s_codes[1]] = 'buy'
                if short:
                    book.loc[i, 't '+s_codes[0]] = 'sell'
                else:
                    book.loc[i, 't '+s_codes[0]] = ''
            elif threshold >= sample.loc[i, s_codes[1]+' spread'] >= 0:
                book.loc[i, 't '+s_codes[1]] = ''
                book.loc[i, 't '+s_codes[0]] = ''
            elif 0 > sample.loc[i, s_codes[1]+' spread'] >= -threshold:
                book.loc[i, 't '+s_codes[1]] = ''
                book.loc[i, 't '+s_codes[0]] = ''
            elif -threshold > sample.loc[i, s_codes[1]+' spread']:
                if short:
                    book.loc[i, 't '+s_codes[1]] = 'sell'
                else:
                    book.loc[i, 't '+s_codes[1]] = ''
                book.loc[i, 't '+s_codes[0]] = 'buy'       
        return (book) 
    


    def trading_strategy_inverse(self, sample, thd, s_codes, last_date, short=False, report={}):
        i = sample.index[-1]
        #threshold = float( thd * sample.loc[i, s_codes[1]] )
        threshold = float( thd * sample.loc[i, s_codes[1]+' expected'] )
        if sample.loc[i, s_codes[1]+' spread'] > threshold:
            strategy_y = 'Long '+s_codes[1]
            if short:
                strategy_x = 'Short '+s_codes[0]
            else:
                strategy_x = 'Clear '+s_codes[0]
        elif threshold >= sample.loc[i, s_codes[1]+' spread'] >= 0:
            strategy_y = 'Clear '+s_codes[1]
            strategy_x = 'Clear '+s_codes[0]
        elif 0 > sample.loc[i, s_codes[1]+' spread'] >= -threshold:
            strategy_y = 'Clear '+s_codes[1]
            strategy_x = 'Clear '+s_codes[0]
        elif -threshold > sample.loc[i, s_codes[1]+' spread']:
            strategy_x = 'Long '+s_codes[0]
            if short:
                strategy_y = 'Short '+s_codes[1]
            else:
                strategy_y = 'Clear '+s_codes[1]
        strategy = strategy_x + ' & ' + strategy_y
        #print ('As of', last_date, 'the model suggests you to', strategy)
        if not report == {}:
            report['trading_strategy_x'] = strategy_x.split(' ')[0]
            report['trading_strategy_y'] = strategy_y.split(' ')[0]
                
        present_price = sample.loc[i, s_codes[1]]
        expected_price = round(sample.loc[i, s_codes[1]+' expected'], 2)
        range_cap = round(expected_price * (1+thd), 2)
        range_floor = round(expected_price * (1-thd), 2)
        
        if not report == {}:
            report['present_price'] = present_price
            report['expected_price'] = expected_price
            report['range_floor'] = range_floor
            report['range_cap'] = range_cap
      
        return (strategy) 
        
    


    
    
    
class FuturesTradeOnValue(PairTrade):
    
    def expected_y(self, sample, s_codes, r, d, T):
        from finterstellar import Valuation
        vu = Valuation()
        for i in sample.index:
            sample.loc[i, s_codes[1]+' expected'] = vu.futures_price(sample.loc[i, s_codes[0]], r, d, i, T)
        sample[s_codes[1]+' spread'] = sample[s_codes[1]] - sample[s_codes[1]+' expected']
        return (sample)
    
    def intraday_expected_y(self, sample, s_codes, r, d, t, T):
        from finterstellar import Valuation
        vu = Valuation()
        for i in sample.index:
            sample.loc[i, s_codes[1]+' expected'] = vu.futures_price(sample.loc[i, s_codes[0]], r, d, t, T)
        sample[s_codes[1]+' spread'] = sample[s_codes[1]] - sample[s_codes[1]+' expected']
        return (sample)

    
    def price_analyze(self, sample, thd, s_codes):
        for i in sample.index:
            threshold = float( thd * sample.loc[i, s_codes[1]] )
            if sample.loc[i, s_codes[1]+' spread'] > 0:
                sample.loc[i, 'cheaper'] = s_codes[0]
            elif sample.loc[i, s_codes[1]+' spread'] < -threshold:
                sample.loc[i, 'cheaper'] = s_codes[1]
            else:
                sample.loc[i, 'cheaper'] = 'E'
        print(sample.groupby('cheaper').count())
        return (sample)
    
    
    def tradings(self, sample, book, thd, s_codes):
        for i in sample.index:
            threshold = float( thd * sample.loc[i, s_codes[1]] )
            if sample.loc[i, s_codes[1]+' spread'] > threshold:
                book.loc[i, 't '+s_codes[1]] = 'sell'
            elif threshold >= sample.loc[i, s_codes[1]+' spread'] >= 0:
                book.loc[i, 't '+s_codes[1]] = 'sell'
            elif 0 > sample.loc[i, s_codes[1]+' spread'] >= -threshold:
                book.loc[i, 't '+s_codes[1]] = ''
            elif -threshold > sample.loc[i, s_codes[1]+' spread']:
                book.loc[i, 't '+s_codes[1]] = 'buy'
        return (book) 
    

    def trading_strategy(self, sample, thd, s_codes, last_date):
        i = sample.index[-1]
        threshold = float( thd * sample.loc[i, s_codes[1]] )
        
        if sample.loc[i, s_codes[1]+' spread'] > threshold:
            strategy = 'sell '+s_codes[1]
        elif threshold >= sample.loc[i, s_codes[1]+' spread'] >= 0:
            strategy = 'sell '+s_codes[1]
        elif 0 > sample.loc[i, s_codes[1]+' spread'] >= -threshold:
            strategy = 'do nothing'
        elif -threshold > sample.loc[i, s_codes[1]+' spread']:
            strategy = 'buy '+s_codes[1]

        print ('As of', last_date, 'this model suggests you to', strategy)
        return (strategy) 

    
    
class FuturesTradeOnBasis(Trade):
    
    def basis_calculate(self, df, pair):
        basis = df[pair[1]] - df[pair[0]]
        df['basis'] = basis
        return (df)
    
    
    def price_analyze(self, sample, thd, s_codes):
        for i in sample.index:
            if sample.loc[i, 'basis'] > thd:
                sample.loc[i, 'cheaper'] = s_codes[0]
            elif sample.loc[i, 'basis'] < 0:
                sample.loc[i, 'cheaper'] = s_codes[1]
            else:
                sample.loc[i, 'cheaper'] = 'E'
        print(sample.groupby('cheaper').count())
        return (sample)
    
    
    def tradings(self, sample, book, thd, s_codes):
        for i in sample.index:
            if sample.loc[i, 'basis'] > thd:
                book.loc[i, 't '+s_codes[1]] = 'sell'
            elif thd >= sample.loc[i, 'basis'] >= 0:
                book.loc[i, 't '+s_codes[1]] = ''
            elif 0 > sample.loc[i, 'basis']:
                book.loc[i, 't '+s_codes[1]] = 'buy'
        return (book) 
    
    
    def trading_strategy(self, sample, thd, s_codes, last_date):
        i = sample.index[-1]
        
        if sample.loc[i, 'basis'] > thd:
            strategy = 'sell '+s_codes[1]
        elif thd >= sample.loc[i, 'basis'] >= 0:
            strategy = 'do nothing'
        elif 0 > sample.loc[i, 'basis']:
            strategy = 'buy '+s_codes[s1]
  
        print ('As of', last_date, 'this model suggests you to', strategy)
        return (strategy) 
    

    
    
