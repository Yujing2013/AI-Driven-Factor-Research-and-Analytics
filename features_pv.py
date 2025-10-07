# features_pv.py
# -*- coding: utf-8 -*-
"""
特征工程模块

- 量价类特征
- 输入: MultiIndex(df.index = [trade_date, ts_code])，含基础量价列
- 输出: (df, feature_cols) —— 在原 df 上添加列，并返回特征列名列表
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple

# 与主脚本保持一致的基础列
BASE_COLS: List[str] = [
    "open","high","low","close","vol","amount","turnover_rate_f","float_share"
]

# 导出给主脚本用的特征列（build_features 之后才会被填充）
FEATURE_COLS: List[str] = []

def _ensure_base_cols(df: pd.DataFrame) -> None:
    """若缺基础列则补 NaN（保证管线稳健）。"""
    for c in BASE_COLS:
        if c not in df.columns:
            df[c] = np.nan

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    在 df 上添加“示例特征”，返回 (df, feature_cols)
    —— 这里是demo，在此处添加更多特征
    """
    _ensure_base_cols(df)

    # ===== 示例：与你主脚本一致的三列衍生特征 =====
    # 1) 高低价价差占比
    df['hl_spread'] = (df['high'] - df['low']) / df['close'].replace(0, np.nan)

    # 2) 开收涨跌幅
    df['oc_ret'] = df['close'] / df['open'] - 1.0

    # 3) 成交量 / (成交额/万) —— 用 np.divide，分母为 0 时给 NaN
    df['v_a_ratio'] = np.divide(
        df['vol'],
        (df['amount'] / 1e4),
        out=np.full(len(df), np.nan),
        where=(df['amount'] != 0)
    )
    
    # ===== 组合特征列：基础列 + 示例衍生列 =====
    feature_cols = BASE_COLS + ['hl_spread', 'oc_ret', 'v_a_ratio']

    # 更新导出量，方便主脚本直接 import 使用
    global FEATURE_COLS
    FEATURE_COLS = feature_cols

    return df, feature_cols
