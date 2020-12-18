# -*- coding: utf-8 -*-

import re
import pandas as pd


def str_to_usd(s):
    if is_number(s):
        return '{:,.2f}'.format(float(s))


def str_to_krw(s):
    if is_number(s):
        return '{:,}'.format(int(s))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def str_to_list(s):
    if type(s) == list:
        cds = s
    else:
        cds = []
        cds.append(s)
    return cds


# 날짜 관련
def today():
    return pd.Timestamp.today().date()


def days_before(date, n):
    d = pd.to_datetime(date) - pd.DateOffset(days=n)
    if d.weekday() > 4:
        adj = d.weekday() - 4
        d += pd.DateOffset(days=adj)
    else:
        d = d
    return d.date()


def days_after(date, n):
    d = pd.to_datetime(date) + pd.DateOffset(days=n)
    if d.weekday() > 4:
        adj = 7 - d.weekday()
        d += pd.DateOffset(days=adj)
    else:
        d = d
    return d.date()


def months_before(date, n):
    d = pd.to_datetime(date) - pd.DateOffset(months=n)
    if d.weekday() > 4:
        adj = d.weekday() - 4
        d += pd.DateOffset(days=adj)
    else:
        d = d
    return d.date()


def str_to_num(str_num):
    powers = {'B': 10 ** 9, 'M': 10 ** 6, 'K': 10 ** 3, '': 1}
    m = str_num.replace(',', '')
    m = re.search('([0-9\.]+)(M|B|K|)', m)
    if m:
        val = m.group(1)
        mag = m.group(2)
        return float(val) * powers[mag]
    return 0.0
