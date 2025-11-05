# run_gp_on_panel_slim.py
# -*- coding: utf-8 -*-
# 说明：本脚本对原版进行“功能等价”的精简：
# 1) 去除所有 try/except 与绘图相关代码；
# 2) 保留日度横截面 Winsorize + 标准化、IC/RankIC 的“逐日带号均值”口径；
# 3) 合并重复逻辑，注释更直白。

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm.auto import tqdm
tqdm.monitor_interval = 0  # 避免 tqdm 在个别环境报小警告

from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression

# 保证能 import 到本地 gplearn 包
sys.path.append(os.path.abspath("."))
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import ts_lag1, ts_mean5, ts_mean10, cs_rank_pct
from gplearn.functions import set_function_context, build_context_from_index
from gplearn.fitness import make_fitness

# 你自己的特征构造模块（保持一致）
from features_pv import build_features

# ===================== 基本路径配置 =====================
PANEL_PARQUET = "/Users/astrologer/Desktop/a_share_pv_clean_2015_2025_trimmed.parquet"

# ===================== 读取与目标构造 =====================
# 读取标准面板：MultiIndex = (trade_date, ts_code)
df = pd.read_parquet(PANEL_PARQUET).sort_index()

# 目标值 = 未来两天的相对收益（避免过强噪声，后续会做裁剪）
close = df['close'].unstack('ts_code').sort_index()
ret_fwd1 = (close.shift(-2) / close - 1.0).stack().rename('ret_fwd1')
df = df.join(ret_fwd1, how='left').dropna(subset=['ret_fwd1'])

# 对异常收益做轻剪裁，减少极端值对训练的破坏
df['ret_fwd1'] = df['ret_fwd1'].clip(-0.3, 0.3)

# 特征工程：返回处理后的 df 与特征列名列表
df, feature_cols = build_features(df)

# ===================== 计算中性化所需的基础特征 =====================
# 计算市值（流通股本 × 收盘价）
if 'float_share' in df.columns and 'close' in df.columns:
    df['market_cap'] = df['float_share'] * df['close']
else:
    df['market_cap'] = np.nan

# 计算20日收益率、20日换手率、20日波动率（按股票分组，跨日期计算）
print("计算20日收益率、换手率、波动率...")
close_unstacked = df['close'].unstack('ts_code').sort_index()
ret_20d = (close_unstacked / close_unstacked.shift(20) - 1.0).stack().rename('ret_20d')
df = df.join(ret_20d, how='left')

if 'turnover_rate_f' in df.columns:
    turnover_unstacked = df['turnover_rate_f'].unstack('ts_code').sort_index()
    turnover_20d = turnover_unstacked.rolling(20, min_periods=1).mean().stack().rename('turnover_20d')
    df = df.join(turnover_20d, how='left')
else:
    df['turnover_20d'] = np.nan

# 20日波动率：过去20日的收益率标准差
ret_daily_unstacked = (close_unstacked / close_unstacked.shift(1) - 1.0)
volatility_20d = ret_daily_unstacked.rolling(20, min_periods=1).std().stack().rename('volatility_20d')
df = df.join(volatility_20d, how='left')

# 暂时不使用行业中性化
# if 'industry' not in df.columns:
#     df['industry'] = None
#     print("警告：数据中未找到 'industry' 列，中性化时将跳过行业维度")

# ===================== 横截面标准化（每日一组） =====================
# 步骤（对每个交易日）：(1) 按列做 1%-99% 分位裁剪；(2) 替换 inf 为 NaN；
# (3) Z-Score 标准化（均值 0/方差 1），便于后续回归/相关度更稳定
def cs_winsorize_zscore(group: pd.DataFrame) -> pd.DataFrame:
    X = group[feature_cols].copy()
    lower = X.quantile(0.01)
    upper = X.quantile(0.99)
    X = X.clip(lower=lower, upper=upper, axis=1).replace([np.inf, -np.inf], np.nan)
    X = (X - X.mean()) / (X.std(ddof=0) + 1e-9)
    group[feature_cols] = X
    return group

tqdm.pandas(desc="CS winsorize + zscore (by day)")
df = df.groupby(level=0, group_keys=False).progress_apply(cs_winsorize_zscore)
df = df.dropna(subset=feature_cols + ['ret_fwd1'])

# ===================== 时间切片（训练/验证/测试） =====================
# 说明：日期范围可按需更改；mask 的索引需与 df 完全对齐
dates = df.index.get_level_values(0).astype(str)

train_mask = pd.Series((dates >= "20170101") & (dates <= "20170601"), index=df.index)
valid_mask = pd.Series((dates >= "20190302") & (dates <= "20190602"), index=df.index)
test_mask  = pd.Series((dates >= "20200420") & (dates <= "20200620"), index=df.index)

# 简单的非空保证（不用 try/except；直接断言）
def _assert_nonempty(name: str, mask: pd.Series):
    if int(mask.sum()) == 0:
        raise RuntimeError(f"[{name}] 时间切片无样本，请检查日期范围/数据清洗输出。")

_assert_nonempty("train", train_mask)
_assert_nonempty("valid", valid_mask)
_assert_nonempty("test",  test_mask)

# 取 numpy 矩阵用于 gplearn 拟合
X_train = df.loc[train_mask, feature_cols].values
y_train = df.loc[train_mask, 'ret_fwd1'].values
X_valid = df.loc[valid_mask, feature_cols].values
y_valid = df.loc[valid_mask, 'ret_fwd1'].values
X_test  = df.loc[test_mask,  feature_cols].values
y_test  = df.loc[test_mask,  'ret_fwd1'].values

# ===================== 样本权重（让日度权重更均衡） =====================
# 思路：每天样本数不同，为避免“样本多的那天权重更大”，
# 给“每条样本权重 = 1/当日样本数”，使“每个交易日的总权重”接近一致。
def day_uniform_weights(mask: pd.Series) -> np.ndarray:
    sub_idx = df.index[mask]
    day = pd.Index(sub_idx.get_level_values(0))
    cnt = day.value_counts()
    # 注意：map 后是按索引回填对应计数
    w = 1.0 / day.map(cnt).astype(float)
    return w.values

w_train = day_uniform_weights(train_mask)
w_valid = day_uniform_weights(valid_mask)
w_test  = day_uniform_weights(test_mask)

# ===================== 中性化函数（横截面回归中性化） =====================
# 全局变量：存储中性化特征数据（按索引映射）
_neutralization_data = {}

def prepare_neutralization_data(dataframe: pd.DataFrame, index: pd.MultiIndex):
    """准备中性化所需的数据，按索引存储（不包括行业）"""
    global _neutralization_data
    # 确保索引在 dataframe 中
    available_indices = index.intersection(dataframe.index)
    sub_df = dataframe.loc[available_indices]
    neutral_features = {}
    
    for idx in index:
        if idx in sub_df.index:
            row = sub_df.loc[idx]
            if isinstance(row, pd.Series):
                neutral_features[idx] = {
                    # 'industry': row.get('industry', None) if 'industry' in row else None,  # 暂时不使用行业
                    'market_cap': row.get('market_cap', np.nan) if 'market_cap' in row else np.nan,
                    'ret_20d': row.get('ret_20d', np.nan) if 'ret_20d' in row else np.nan,
                    'turnover_20d': row.get('turnover_20d', np.nan) if 'turnover_20d' in row else np.nan,
                    'volatility_20d': row.get('volatility_20d', np.nan) if 'volatility_20d' in row else np.nan
                }
            else:
                # 如果返回的是 DataFrame（多行），取第一行
                neutral_features[idx] = {
                    # 'industry': row.iloc[0].get('industry', None) if 'industry' in row.columns else None,  # 暂时不使用行业
                    'market_cap': row.iloc[0].get('market_cap', np.nan) if 'market_cap' in row.columns else np.nan,
                    'ret_20d': row.iloc[0].get('ret_20d', np.nan) if 'ret_20d' in row.columns else np.nan,
                    'turnover_20d': row.iloc[0].get('turnover_20d', np.nan) if 'turnover_20d' in row.columns else np.nan,
                    'volatility_20d': row.iloc[0].get('volatility_20d', np.nan) if 'volatility_20d' in row.columns else np.nan
                }
        else:
            # 如果索引不在 dataframe 中，使用默认值
            neutral_features[idx] = {
                # 'industry': None,  # 暂时不使用行业
                'market_cap': np.nan,
                'ret_20d': np.nan,
                'turnover_20d': np.nan,
                'volatility_20d': np.nan
            }
    
    _neutralization_data = neutral_features

def neutralize_factor_cross_section(factor_values: np.ndarray, 
                                     neutral_features_list: list,
                                     min_samples: int = 50) -> np.ndarray:
    """
    对因子进行横截面中性化（去除市值、20日收益率、20日换手率、20日波动率的影响）
    暂时不包括行业中性化
    参数:
        factor_values: 因子值数组 (n,)
        neutral_features_list: 中性化特征列表，每个元素是一个字典，包含：
            - market_cap: 市值
            - ret_20d: 20日收益率
            - turnover_20d: 20日换手率
            - volatility_20d: 20日波动率
        min_samples: 最小样本数，少于该数量则跳过中性化
    返回:
        中性化后的因子值数组
    """
    if len(factor_values) < min_samples:
        return factor_values
    
    # 收集连续特征（仅使用市值、20日收益率、20日换手率、20日波动率，不包括行业）
    market_caps = np.array([feat.get('market_cap', np.nan) for feat in neutral_features_list])
    ret_20ds = np.array([feat.get('ret_20d', np.nan) for feat in neutral_features_list])
    turnover_20ds = np.array([feat.get('turnover_20d', np.nan) for feat in neutral_features_list])
    volatility_20ds = np.array([feat.get('volatility_20d', np.nan) for feat in neutral_features_list])
    
    # 构建特征矩阵（只包含有有效值的特征）
    continuous_features = []
    
    if np.isfinite(market_caps).sum() > 10:  # 至少要有10个有效值
        continuous_features.append(market_caps.reshape(-1, 1))
    
    if np.isfinite(ret_20ds).sum() > 10:
        continuous_features.append(ret_20ds.reshape(-1, 1))
    
    if np.isfinite(turnover_20ds).sum() > 10:
        continuous_features.append(turnover_20ds.reshape(-1, 1))
    
    if np.isfinite(volatility_20ds).sum() > 10:
        continuous_features.append(volatility_20ds.reshape(-1, 1))
    
    # 暂时不使用行业中性化
    # industries = [feat.get('industry') for feat in neutral_features_list]
    # has_industry = any(ind is not None and pd.notna(ind) for ind in industries)
    # if has_industry:
    #     industry_series = pd.Series([ind if ind is not None else 'UNKNOWN' 
    #                                  for ind in industries])
    #     industry_dummies = pd.get_dummies(industry_series, prefix='industry', 
    #                                       drop_first=True).values
    
    if len(continuous_features) == 0:
        # 没有可用的中性化特征，直接返回原值
        return factor_values
    
    X_neutral = np.hstack(continuous_features)
    
    # 检查有效样本：因子值必须有效，中性化特征允许部分缺失（只要有至少2个特征有效即可）
    factor_mask = np.isfinite(factor_values)
    
    # 对每个样本，计算有效特征的数量
    X_valid_count = np.isfinite(X_neutral).sum(axis=1)
    # 至少要有2个有效特征（或者至少50%的特征有效）
    min_valid_features = max(2, int(X_neutral.shape[1] * 0.5))
    X_mask = X_valid_count >= min_valid_features
    
    valid_mask = factor_mask & X_mask
    
    if valid_mask.sum() < min_samples:
        # 如果有效样本太少，放宽条件：只要因子有效且至少有一个特征有效即可
        X_mask_loose = np.isfinite(X_neutral).sum(axis=1) >= 1
        valid_mask = factor_mask & X_mask_loose
        if valid_mask.sum() < min_samples:
            return factor_values
    
    # 提取有效数据
    y_valid = factor_values[valid_mask]
    X_valid = X_neutral[valid_mask, :]
    
    # 对缺失值用该特征的均值填充（仅用于回归拟合）
    X_valid_filled = X_valid.copy()
    for col_idx in range(X_valid.shape[1]):
        col = X_valid[:, col_idx]
        mask = np.isfinite(col)
        if mask.sum() > 0:
            col_mean = np.mean(col[mask])
            X_valid_filled[~mask, col_idx] = col_mean
    
    # 对中性化特征进行标准化（提高回归稳定性）
    X_mean = np.mean(X_valid_filled, axis=0, keepdims=True)
    X_std = np.std(X_valid_filled, axis=0, keepdims=True) + 1e-9
    X_valid_std = (X_valid_filled - X_mean) / X_std
    
    # 回归中性化
    try:
        reg = LinearRegression()
        reg.fit(X_valid_std, y_valid)
        
        # 对所有样本进行预测并计算残差
        # 先填充缺失值
        X_neutral_filled = X_neutral.copy()
        for col_idx in range(X_neutral.shape[1]):
            col = X_neutral[:, col_idx]
            mask = np.isfinite(col)
            if mask.sum() > 0:
                col_mean = np.mean(col[mask])
                X_neutral_filled[~mask, col_idx] = col_mean
        
        # 进行相同的标准化
        X_neutral_std = (X_neutral_filled - X_mean) / X_std
        y_pred_all = reg.predict(X_neutral_std)
        
        # 返回残差（中性化后的因子）
        factor_neutralized = factor_values - y_pred_all
        
        # 对于无效样本，保持原值
        factor_neutralized[~valid_mask] = factor_values[~valid_mask]
        return factor_neutralized
    except Exception:
        # 如果回归失败，返回原值
        return factor_values

# ===================== 逐日相关度度量（严格口径） =====================
# 统一的“分组切片”函数：把一个按日期排序的 MultiIndex 切成若干 [s, t) 段
def _prepare_groups(mi: pd.MultiIndex):
    days = mi.get_level_values(0).to_numpy()
    bounds = []
    s = 0
    for i in range(1, len(days) + 1):
        if i == len(days) or days[i] != days[s]:
            bounds.append((s, i))
            s = i
    return bounds

# 训练集索引 → 预先准备"逐日片段"，供 fitness 使用（避免重复 groupby）
_train_groups = None
_train_index_for_neutral = None
def set_train_index_for_metric(mi: pd.MultiIndex):
    global _train_groups, _train_index_for_neutral
    _train_groups = _prepare_groups(mi)
    _train_index_for_neutral = mi
    prepare_neutralization_data(df, mi)

# 在每个交易日内：对预测值做 MAD 去极值 + 中性化 + 标准化，再与真实值算相关度
# kind='rank' 用 Spearman（RankIC），kind='pearson' 用 Pearson（IC）
def _daily_mean_corr(mi: pd.MultiIndex, y_true: np.ndarray, y_pred: np.ndarray, kind: str) -> float:
    groups = _prepare_groups(mi)
    y = np.asarray(y_true)
    h = np.asarray(y_pred)

    if not np.isfinite(h).any() or np.nanstd(h) < 1e-12:
        return 0.0

    # 准备中性化数据
    prepare_neutralization_data(df, mi)
    
    vals = []
    for (s, t) in groups:
        yy = y[s:t]
        hh = h[s:t]
        m = np.isfinite(yy) & np.isfinite(hh)
        if int(m.sum()) < 50:
            vals.append(0.0)
            continue
        yy = yy[m]
        hh = hh[m]
        
        # 获取对应的索引和中性化特征
        day_indices = mi[s:t][m]
        neutral_features_list = [_neutralization_data.get(idx, {}) for idx in day_indices]

        # MAD 去极值（稳健，不被极少数极端值牵着走）
        med = np.median(hh)
        mad = np.median(np.abs(hh - med))
        if mad > 0.0:
            hh = np.clip(hh, med - 5*mad, med + 5*mad)

        # 中性化（在每个截面上剔除市值、20日收益率、20日换手率、20日波动率的影响，暂时不包括行业）
        hh = neutralize_factor_cross_section(hh, neutral_features_list, min_samples=50)

        # 标准化（避免"几乎常数"的退化）
        std = np.std(hh)
        if std <= 1e-12:
            vals.append(0.0)
            continue
        hh = (hh - np.mean(hh)) / (std + 1e-9)

        # 横截面区分度太弱 → 记 0
        nunq = len(np.unique(hh))
        if nunq < 3 or (nunq / len(hh)) < 0.05 or len(np.unique(yy)) < 3:
            vals.append(0.0)
            continue

        if kind == 'rank':
            r = spearmanr(yy, hh).statistic
        else:
            r = pearsonr(yy, hh).statistic
        vals.append(float(r) if (r is not None and np.isfinite(r)) else 0.0)

    return float(np.mean(vals)) if len(vals) > 0 else 0.0

# gplearn 的自定义适应度：训练时按“逐日相关度的带号均值”作为目标
FITNESS_KIND = 'pearson'  # 'rank' → RankIC；'pearson' → IC

def _fitness_daily_corr_numpy(y_true, y_pred, sample_weight):
    # 与 _daily_mean_corr 一致，但直接使用"预先准备好的训练期分组"
    if _train_groups is None or _train_index_for_neutral is None:
        return 0.0

    y = np.asarray(y_true)
    h = np.asarray(y_pred)
    if not np.isfinite(h).any() or np.nanstd(h) < 1e-12:
        return 0.0

    scores = []
    zero_days = 0
    for (s, t) in _train_groups:
        yy = y[s:t]
        hh = h[s:t]
        m = np.isfinite(yy) & np.isfinite(hh)
        if int(m.sum()) < 50:
            scores.append(0.0); zero_days += 1; continue

        yy = yy[m]
        hh = hh[m]
        
        # 获取对应的索引和中性化特征
        day_indices = _train_index_for_neutral[s:t][m]
        neutral_features_list = [_neutralization_data.get(idx, {}) for idx in day_indices]

        # MAD 去极值（稳健，不被极少数极端值牵着走）
        med = np.median(hh)
        mad = np.median(np.abs(hh - med))
        if mad > 0.0:
            hh = np.clip(hh, med - 5*mad, med + 5*mad)

        # 中性化（在每个截面上剔除市值、20日收益率、20日换手率、20日波动率的影响，暂时不包括行业）
        hh = neutralize_factor_cross_section(hh, neutral_features_list, min_samples=50)

        std = np.std(hh)
        if std <= 1e-12:
            scores.append(0.0); zero_days += 1; continue
        hh = (hh - np.mean(hh)) / (std + 1e-9)

        nunq = len(np.unique(hh))
        uniq_ratio = nunq / len(hh)
        if nunq < 3 or uniq_ratio < 0.05 or len(np.unique(yy)) < 3:
            scores.append(0.0); zero_days += 1; continue

        if FITNESS_KIND == 'rank':
            r = spearmanr(yy, hh).statistic
        else:
            r = pearsonr(yy, hh).statistic

        if r is None or not np.isfinite(r):
            r = 0.0; zero_days += 1
        scores.append(float(r))

    total_days = len(_train_groups)
    if total_days == 0 or (zero_days / total_days) > 0.5:
        return 0.0
    return float(np.mean(scores))

fitness_gp = make_fitness(function=_fitness_daily_corr_numpy, greater_is_better=True)

# ===================== 定义函数集 & 训练 GP =====================
func_set = (
    'add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs', 'neg', 'max', 'min',
    ts_lag1, ts_mean5, ts_mean10, cs_rank_pct
)

train_index = df.loc[train_mask].index
set_train_index_for_metric(train_index)  # 预设训练期“逐日片段”

est = SymbolicRegressor(
    population_size=600,
    generations=3,
    init_depth=(2, 5),
    tournament_size=20,
    function_set=func_set,
    metric=fitness_gp,
    parsimony_coefficient='auto',
    p_crossover=0.5,
    p_subtree_mutation=0.05,
    p_hoist_mutation=0.0,
    p_point_mutation=0.04,
    p_point_replace=0.2,
    max_samples=0.7,
    n_jobs=1,                # 为与上下文兼容，保持单线程
    random_state=24,
    verbose=1,
    stopping_criteria=0.20,  # 早停阈值
    feature_names=feature_cols,
    const_range=None         # 不产生常数节点，避免退化
)

# 训练前设定上下文（用于时间序列函数）
est.set_context_index(train_index)
est.fit(X_train, y_train, sample_weight=w_train)

# ===================== 预测与评估（严格口径） =====================
valid_index = df.loc[valid_mask].index
test_index  = df.loc[test_mask].index

def _predict_with_context(estimator, index, X):
    estimator.set_context_index(index)
    return estimator.predict(X)

yhat_valid = _predict_with_context(est, valid_index, X_valid)
yhat_test  = _predict_with_context(est, test_index,  X_test)

print("\n# ==== Overall (current best) ====")
print(f"RankIC  VALID={_daily_mean_corr(valid_index, y_valid, yhat_valid, 'rank'):+.4f}   "
      f"TEST={_daily_mean_corr(test_index,  y_test,  yhat_test,  'rank'):+.4f}")
print(f"IC      VALID={_daily_mean_corr(valid_index, y_valid, yhat_valid, 'pearson'):+.4f}   "
      f"TEST={_daily_mean_corr(test_index,  y_test,  yhat_test,  'pearson'):+.4f}")

# ===================== 收集与去重 Top 程序（不画图，仅文本输出） =====================
def get_topk_programs(estimator, topk=40):
    """从各世代收集 Program，按表达式字符串去重后，按 fitness_ 由高到低取前 k。"""
    all_progs = []
    for gen in getattr(estimator, "_programs", []):
        for prog in gen:
            if (prog is not None) and (getattr(prog, "fitness_", None) is not None):
                all_progs.append(prog)
    if not all_progs and getattr(estimator, "_program", None) is not None:
        all_progs = [estimator._program]

    seen = set()
    uniq = []
    for prog in all_progs:
        sig = str(prog)
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(prog)

    uniq.sort(key=lambda p: p.fitness_, reverse=True)
    return uniq[: min(topk, len(uniq))]

topk_pool = get_topk_programs(est, topk=50)

# 评估：为了“多样性”，剔除与已有因子/彼此近似重复的程序（基于相关度阈值）
def _dedup_and_rank(programs,
                    X_ref: np.ndarray,
                    context_ref,
                    feature_cols,
                    topk: int = 10,
                    corr_tol: float = 0.995,
                    novelty_weight: float = 0.75,
                    min_length: int = 2,
                    feature_sample: int = 12):
    if len(programs) == 0:
        return []

    kept_preds = []   # 已保留程序的预测序列（用于去重）
    ranked = []      # 记录列表（含得分/新颖度等）
    sample_idx = np.arange(min(len(feature_cols), feature_sample))
    feat_ref = X_ref[:, sample_idx] if sample_idx.size > 0 else np.empty((len(X_ref), 0))

    # 为调用 ts_* 函数，先设置好“参考索引上下文”
    set_function_context(context_ref)

    for prog in programs:
        if getattr(prog, "length_", 0) < min_length:
            continue

        yhat = prog.execute(X_ref)
        if (not np.isfinite(yhat).any()) or (np.nanstd(yhat) < 1e-12):
            continue

        # 与已选程序的预测高度相似（|corr|>阈值）→ 视为重复，跳过
        is_dup = False
        for ex in kept_preds:
            mask = np.isfinite(ex) & np.isfinite(yhat)
            if int(mask.sum()) < 50:
                continue
            r_p = pearsonr(ex[mask], yhat[mask]).statistic
            r_s = spearmanr(ex[mask], yhat[mask]).statistic
            corr = max(abs(r_p if np.isfinite(r_p) else 0.0),
                       abs(r_s if np.isfinite(r_s) else 0.0))
            if corr > corr_tol:
                is_dup = True
                break
        if is_dup:
            continue

        # 与“原始特征”最高相关度，越低说明“新信息”越多
        max_corr_feat = 0.0
        for j in range(feat_ref.shape[1]):
            col = feat_ref[:, j]
            mask = np.isfinite(col) & np.isfinite(yhat)
            if int(mask.sum()) < 50:
                continue
            r_p = pearsonr(col[mask], yhat[mask]).statistic
            r_s = spearmanr(col[mask], yhat[mask]).statistic
            corr = max(abs(r_p if np.isfinite(r_p) else 0.0),
                       abs(r_s if np.isfinite(r_s) else 0.0))
            if corr > max_corr_feat:
                max_corr_feat = corr

        novelty = max(0.0, 1.0 - max_corr_feat)   # “越不相关”越新颖
        fitness = float(getattr(prog, "fitness_", 0.0))
        score = fitness * (1.0 + novelty_weight * novelty)

        ranked.append({
            "program": prog,
            "novelty": novelty,
            "fitness": fitness,
            "score": score,
            "values": yhat
        })
        kept_preds.append(yhat)

    ranked.sort(key=lambda it: it["score"], reverse=True)
    return ranked[:topk]

context_valid = build_context_from_index(valid_index)
context_test  = build_context_from_index(test_index)

topk_summary = _dedup_and_rank(
    topk_pool,
    X_valid,
    context_valid,
    feature_cols,
    topk=10,
    corr_tol=0.995,
    novelty_weight=0.75,
    min_length=2
)

# 打印 Top 程序在 VALID/TEST 的 RankIC/IC 表现（逐日口径的带号均值）
print("\n# ==== Top Programs: VALID / TEST (signed daily means of RankIC & IC) ====")
for i, meta in enumerate(topk_summary, 1):
    prog = meta["program"]
    yv = meta["values"]
    ric_v = _daily_mean_corr(valid_index, y_valid, yv, kind='rank')
    ic_v  = _daily_mean_corr(valid_index, y_valid, yv, kind='pearson')

    set_function_context(context_test)
    yt = prog.execute(X_test)
    ric_t = _daily_mean_corr(test_index, y_test, yt, kind='rank')
    ic_t  = _daily_mean_corr(test_index, y_test, yt, kind='pearson')

    print(f"#{i:02d}  RankIC[V,T]={ric_v:+.4f}, {ric_t:+.4f}   IC[V,T]={ic_v:+.4f}, {ic_t:+.4f}   "
          f"nov={meta['novelty']:.3f}  fit={meta['fitness']:+.4f} :: {prog}")
