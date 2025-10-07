# -*- coding: utf-8 -*-
"""
get_a_share_pv_clean.py
功能：
  - 下载 2015-2025 全A股 量价类数据（后复权 OHLC、vol、amount、turnover_rate_f、float_share）
  - 清洗：剔除 ST/PT、剔除下一交易日停牌、剔除上市不足 N 天
  - 输出：MultiIndex(panel) DataFrame，可选保存 parquet/csv

使用：
  1) pip install tushare pandas numpy tqdm pyarrow
  2) 将你的 tushare token 填到 TS_TOKEN
  3) python get_a_share_pv_clean.py
"""

import os
import time
import math
from typing import List, Dict, Set
import pandas as pd
import numpy as np
from tqdm import tqdm
import tushare as ts

# ===================== 配置区 =====================
TS_TOKEN = os.environ.get("TUSHARE_TOKEN", "daa841661584d6922218a03003eaa2263101592d5b9c63e1753aeaa2")
START_DATE = "20150101"
END_DATE   = "20251231"
MIN_LIST_DAYS = 180              # 可改为 120
SAVE_PARQUET_PATH = "a_share_pv_clean_2015_2025.parquet"  # 改为 None 则不保存
SAVE_CSV_PATH     = None         # 比如 "a_share_pv_clean_2015_2025.csv"
SLEEP_SEC = 0.12                 # 每次请求后的轻微间隔，减少限流
MAX_RETRY = 3

# 需要的列（最终统一输出列）
FINAL_COLS = [
    "open","high","low","close","vol","amount",
    "turnover_rate_f","float_share"
]

# =================================================

def set_pro():
    ts.set_token(TS_TOKEN)
    return ts.pro_api()

pro = set_pro()

def _retry_call(fn, **kwargs):
    """通用带重试的调用器"""
    for i in range(MAX_RETRY):
        try:
            df = fn(**kwargs)
            time.sleep(SLEEP_SEC)
            return df
        except Exception as e:
            if i == MAX_RETRY - 1:
                raise
            time.sleep(1.0 + i * 1.5)
    return pd.DataFrame()

def get_trade_calendar(start_date, end_date) -> pd.DataFrame:
    cal = _retry_call(pro.trade_cal, exchange='SSE', start_date=start_date, end_date=end_date)
    cal = cal[cal['is_open'] == 1].sort_values('cal_date')
    return cal[['cal_date']].rename(columns={'cal_date': 'trade_date'}).reset_index(drop=True)

def get_next_trade_map(trade_days: pd.Series) -> Dict[str, str]:
    """给定交易日序列，返回 {today: next_day} 映射"""
    days = trade_days.tolist()
    nxt = {}
    for i in range(len(days)-1):
        nxt[days[i]] = days[i+1]
    # 最后一天没有下一天，不放
    return nxt

def get_all_listed_stocks() -> pd.DataFrame:
    """获取上市股票列表（全 A 股：上交所/深交所/北交所）。"""
    fields = 'ts_code,symbol,name,area,industry,market,exchange,list_date,list_status'
    base = _retry_call(pro.stock_basic, exchange='', list_status='', fields=fields)
    # 仅保留交易所为上/深/北(可选)；并排除基金/债券等（stock_basic 已经是股票）
    base = base.dropna(subset=['list_date'])
    return base

def build_namechange_st_flags(ts_code: str) -> pd.DataFrame:
    """
    从 namechange 表构建该股票在不同日期是否为 ST/PT 的区间。
    只取 name 包含 'ST'、'*ST'、'PT' 的时期。
    """
    try:
        df = _retry_call(pro.namechange, ts_code=ts_code, fields="ts_code,name,start_date,end_date,change_type")
    except Exception:
        return pd.DataFrame(columns=["ts_code","start_date","end_date","is_st"])
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts_code","start_date","end_date","is_st"])
    df['is_st'] = df['name'].astype(str).str.upper().str.contains("ST") | df['name'].astype(str).str.upper().str.contains("PT")
    df = df[df['is_st']]
    # 空 end_date 代表至今
    df['end_date'] = df['end_date'].replace('', np.nan)
    df['end_date'] = df['end_date'].fillna(END_DATE)
    return df[['ts_code','start_date','end_date','is_st']]

def get_suspensions(start_date, end_date, ts_code=None) -> pd.DataFrame:
    """
    获取停牌日期（逐股），tushare 有的账户是 pro.suspend，有的是 pro.suspend_d。
    统一返回列：ts_code, suspend_date
    """
    # 优先尝试 suspend_d
    try:
        if ts_code:
            df = _retry_call(pro.suspend_d, ts_code=ts_code, start_date=start_date, end_date=end_date)
        else:
            df = _retry_call(pro.suspend_d, start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            return df.rename(columns={'suspend_date': 'suspend_date'})[['ts_code','suspend_date']]
    except Exception:
        pass
    # 退而求其次 suspend
    try:
        if ts_code:
            df = _retry_call(pro.suspend, ts_code=ts_code, start_date=start_date, end_date=end_date)
        else:
            df = _retry_call(pro.suspend, start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            # pro.suspend 里一般也有 suspend_date
            return df.rename(columns={'suspend_date': 'suspend_date'})[['ts_code','suspend_date']]
    except Exception:
        pass
    return pd.DataFrame(columns=['ts_code','suspend_date'])

def get_bar_hfq(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    使用 ts.pro_bar 拉取后复权 OHLC、vol、amount。
    注意：pro_bar 是在 tushare 包层（非 pro 对象）。
    """
    for i in range(MAX_RETRY):
        try:
            df = ts.pro_bar(ts_code=ts_code, adj='hfq', start_date=start_date, end_date=end_date,
                            factors=None, freq='D', ma=None)
            if df is None:
                df = pd.DataFrame()
            time.sleep(SLEEP_SEC)
            return df
        except Exception:
            if i == MAX_RETRY - 1:
                raise
            time.sleep(1.0 + i * 1.5)
    return pd.DataFrame()

def get_daily_basic(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    fields = "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,total_share,float_share,free_share,total_mv,circ_mv"
    df = _retry_call(pro.daily_basic, ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields)
    return df

def date_diff_days(d1: str, d2: str) -> int:
    """字符串日期相减（YYYYMMDD），返回天数（d1 - d2）。"""
    return (pd.to_datetime(d1) - pd.to_datetime(d2)).days

def main():
    # 1) 交易日与 next_day 映射
    cal = get_trade_calendar(START_DATE, END_DATE)
    trade_days = cal['trade_date']  # str
    next_map = get_next_trade_map(trade_days)

    # 2) 全部股票列表
    base = get_all_listed_stocks()
    # 仅保留沪深北股票（根据 market/exchange 过滤，保险起见不过滤过度）
    base = base[base['list_status'].isin(['L','P','D'])]  # 在表里全保留，数据以实际可拉到为准

    # 3) 预下载全市场停牌日期（可选：逐股拉取会更稳，但慢）
    #    这里使用逐股获取，保证准确（因为不同股票停牌记录不均匀）
    #    若额度不够，可以改为不预拉，改为每只股票内拉。
    # 这里我们在循环内拉取该股票的停牌，以减少一次性大请求。
    # -----------------------------------------------------

    panels = []  # 分批存放结果，最后 concat
    fail_codes = []

    # 为了去 ST/PT，需要 namechange 的历史区间
    namechg_cache: Dict[str, pd.DataFrame] = {}

    # 4) 主循环（全 A 股）
    iterable = base['ts_code'].sort_values().tolist()

    print(f"Total stocks in base list: {len(iterable)}")
    for ts_code in tqdm(iterable, desc="Downloading & cleaning", ncols=90):
        list_date = str(base.loc[base['ts_code']==ts_code, 'list_date'].iloc[0])

        # 拉取后复权 bars
        bar = get_bar_hfq(ts_code, START_DATE, END_DATE)
        if bar is None or bar.empty:
            continue

        # 统一列
        # pro_bar 返回按 trade_date 降序，先排序
        bar = bar.sort_values('trade_date')
        cols_needed = ['ts_code','trade_date','open','high','low','close','vol','amount']
        bar = bar[cols_needed].copy()

        # 拉取 daily_basic（换手率&流通股本等）
        db = get_daily_basic(ts_code, START_DATE, END_DATE)
        if db is None or db.empty:
            # 至少保留价格数据
            db = pd.DataFrame(columns=['ts_code','trade_date','turnover_rate_f','float_share'])
        else:
            db = db.sort_values('trade_date')
        db = db[['ts_code','trade_date','turnover_rate_f','float_share']].copy()

        # 合并
        df = pd.merge(bar, db, on=['ts_code','trade_date'], how='left')

        # 与交易日历对齐，防止个别天缺失
        df = pd.merge(
            pd.DataFrame({'trade_date': trade_days}),
            df,
            on='trade_date',
            how='left'
        )
        # 填回 ts_code（对齐后缺失的位置补 ts_code）
        df['ts_code'] = df['ts_code'].ffill().bfill()

        # 剔除上市不足 N 天
        df['list_days'] = df['trade_date'].apply(lambda d: date_diff_days(d, list_date))
        df = df[df['list_days'] >= MIN_LIST_DAYS]

        if df.empty:
            continue

        # 计算“下一交易日停牌”掩码：
        # 获取该股票的停牌日集合
        susp = get_suspensions(START_DATE, END_DATE, ts_code=ts_code)
        if susp is not None and not susp.empty:
            susp_set: Set[str] = set(susp['suspend_date'].astype(str).tolist())
        else:
            susp_set = set()

        # 给每条记录找到下一交易日（若不存在则视为无下一日，不剔除）
        df['next_trade_date'] = df['trade_date'].map(next_map)
        df['suspend_next'] = df['next_trade_date'].isin(susp_set)
        df = df[~df['suspend_next']].drop(columns=['next_trade_date','suspend_next'])

        # 剔除 ST / PT：使用 namechange 历史（优先）
        if ts_code not in namechg_cache:
            namechg_cache[ts_code] = build_namechange_st_flags(ts_code)
        stdf = namechg_cache[ts_code]

        if stdf is not None and not stdf.empty:
            # 按区间标记
            stdf = stdf.copy()
            stdf['start_date'] = pd.to_datetime(stdf['start_date'])
            stdf['end_date']   = pd.to_datetime(stdf['end_date'])
            df['d'] = pd.to_datetime(df['trade_date'])
            # 汇总所有 ST 区间
            mask_st = pd.Series(False, index=df.index)
            for _, r in stdf.iterrows():
                mask_st |= (df['d'] >= r['start_date']) & (df['d'] <= r['end_date'])
            df = df[~mask_st].drop(columns=['d'])
        else:
            # 兜底：用当前名称判断（不完全准确，但大多数情况有效）
            name_current = str(base.loc[base['ts_code']==ts_code,'name'].iloc[0]).upper()
            if ("ST" in name_current) or ("PT" in name_current):
                # 整体剔除
                df = df.iloc[0:0]

        if df.empty:
            continue

        # 仅保留最终需要列
        keep_cols = ['trade_date','ts_code'] + FINAL_COLS
        for c in FINAL_COLS:
            if c not in df.columns:
                df[c] = np.nan
        df = df[keep_cols].copy()

        # 累加
        panels.append(df)

    if not panels:
        raise RuntimeError("没有拉到任何合格数据，请检查 token/额度或时间范围。")

    panel = pd.concat(panels, ignore_index=True)
    panel = panel.dropna(subset=['ts_code','trade_date']).sort_values(['trade_date','ts_code'])

    # 设置 Panel 形式索引
    panel = panel.set_index(['trade_date','ts_code'])

    # 可选保存
    if SAVE_PARQUET_PATH:
        panel.to_parquet(SAVE_PARQUET_PATH)
        print(f"[OK] 保存 parquet: {SAVE_PARQUET_PATH}")
    if SAVE_CSV_PATH:
        # 注意：CSV 会比较大
        panel.to_csv(SAVE_CSV_PATH)
        print(f"[OK] 保存 csv: {SAVE_CSV_PATH}")

    # 简要概览
    print(panel.info(show_counts=True))
    print(panel.head(10))

if __name__ == "__main__":
    main()
