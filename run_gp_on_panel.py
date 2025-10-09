# run_gp_on_panel.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
tqdm.monitor_interval = 0  # 避免少数环境警告

from scipy.stats import spearmanr
import sys
sys.path.append(os.path.abspath("."))  # 确保能 import 到本地 gplearn 包
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import ts_lag1, ts_mean5, ts_mean10, cs_rank_pct
from gplearn.functions import set_function_context, build_context_from_index
from gplearn.fitness import make_fitness

# ===== 新增：引入特征模块 =====
from features_pv import build_features, FEATURE_COLS

# ===================== 路径配置 =====================
PANEL_PARQUET = "/Users/yz/Desktop/MFE5240/a_share_pv_clean_2015_2025_trimmed.parquet"  # 改成你的实际路径

# ===================== 读取与基础处理 =====================
df = pd.read_parquet(PANEL_PARQUET)  # MultiIndex: (trade_date, ts_code)
df = df.sort_index()

# 目标：下二十日收益 ret_fwd1
close = df['close'].unstack('ts_code').sort_index()
ret_fwd1 = (close.shift(-20) / close - 1.0).stack().rename('ret_fwd1')
df = df.join(ret_fwd1, how='left').dropna(subset=['ret_fwd1'])

# 可选稳健化：裁剪极端收益，防止极少数异常影响训练
df['ret_fwd1'] = df['ret_fwd1'].clip(lower=-0.3, upper=0.3)

# 自变量：基础列 + 简单派生
base_cols = ["open","high","low","close","vol","amount","turnover_rate_f","float_share"]
for c in base_cols:
    if c not in df.columns:
        df[c] = np.nan


# # 可以单独起个py文件专门特征
# # 简单派生（更丰富的特征可后续逐步加入）
# df['hl_spread'] = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
# df['oc_ret']    = df['close'] / df['open'] - 1.0
# # 使用 np.divide 更稳健（分母为 0 返回 NaN） 
# df['v_a_ratio'] = np.divide(df['vol'], (df['amount'] / 1e4), out=np.full(len(df), np.nan), where=(df['amount'] != 0))

# feature_cols = base_cols + ['hl_spread', 'oc_ret', 'v_a_ratio']


# ……读取 df，并完成 ret_fwd1 的计算与裁剪后：
df, feature_cols = build_features(df)  
# 若你想用模块里导出的常量也行：
# feature_cols = FEATURE_COLS


# 横截面去极值 & 标准化（按“交易日”分组）
def cs_winsorize_zscore(g: pd.DataFrame) -> pd.DataFrame:
    x = g[feature_cols].copy()
    # 1%~99% 分位剪裁（逐列）
    lower = x.quantile(0.01)
    upper = x.quantile(0.99)
    x = x.clip(lower, upper, axis=1)
    # 替换 inf 为 NaN，避免影响 zscore
    x = x.replace([np.inf, -np.inf], np.nan)
    # 标准化
    x = (x - x.mean()) / (x.std(ddof=0) + 1e-9)
    g[feature_cols] = x
    return g

tqdm.pandas(desc="CS winsorize/zscore by day")
df = df.groupby(level=0, group_keys=False).progress_apply(cs_winsorize_zscore)
df = df.dropna(subset=feature_cols + ['ret_fwd1'])

# ===================== 时间切片（你要求的短窗口） =====================
# 关键：遮罩的 index 必须与 df.index 完全一致，避免 Unalignable boolean Series
dates = df.index.get_level_values(0).astype(str)

train_mask = pd.Series(
    (dates >= "20150101") & (dates <= "20190101"),
    index=df.index
)
valid_mask = pd.Series(
    (dates >= "20190302") & (dates <= "20200302"),
    index=df.index
)
test_mask = pd.Series(
    (dates >= "20200320") & (dates <= "20210320"),
    index=df.index
)

# 若某阶段没有样本，给出友好提示
def _check_nonempty(name, mask):
    if mask.sum() == 0:
        raise RuntimeError(f"[{name}] 切片无样本，请检查日期范围或清洗输出。")
_check_nonempty("train", train_mask)
_check_nonempty("valid", valid_mask)
_check_nonempty("test",  test_mask)

# 取矩阵
X_train = df.loc[train_mask, feature_cols].values
y_train = df.loc[train_mask, 'ret_fwd1'].values
X_valid = df.loc[valid_mask, feature_cols].values
y_valid = df.loc[valid_mask, 'ret_fwd1'].values
X_test  = df.loc[test_mask,  feature_cols].values
y_test  = df.loc[test_mask,  'ret_fwd1'].values

# ===================== 可选：按日均匀权重 =====================
def day_uniform_weights(mask: pd.Series) -> np.ndarray:
    sub_idx = df.index[mask]
    day = pd.Index(sub_idx.get_level_values(0))
    counts = day.value_counts()
    # 每条样本权重 = 1 / 当天样本数，使得每个“交易日”的总权重大致相等
    w = 1.0 / day.map(counts).astype(float)
    return w.values

w_train = day_uniform_weights(train_mask)
w_valid = day_uniform_weights(valid_mask)
w_test  = day_uniform_weights(test_mask)

# ====== 严格口径的逐日 IC（签名）与多空夏普 ======
from gplearn.fitness import make_fitness

def _groupby_day(index_like):
    # index_like 是 MultiIndex: (trade_date, ts_code)
    return pd.Index(index_like.get_level_values(0))

def daily_ic_series_strict(index, y_true, y_pred,
                           min_crosssec_n=50,
                           uniq_ratio_th=0.05):
    """
    返回逐日“签名 RankIC”的 pd.Series（每个交易日都计入；不达标=0）
    - 截面样本 < min_crosssec_n → 0
    - 横截面唯一值比例 < uniq_ratio_th 或 横截面std≈0 → 0
    - y 或 yhat 横截面可变性不足（唯一值<3）→ 0
    """
    from scipy.stats import spearmanr
    day = _groupby_day(index)
    df_ = (pd.DataFrame({"day": day, "y": y_true, "yhat": y_pred})
             .replace([np.inf, -np.inf], np.nan)
             .dropna())
    if df_.empty:
        return pd.Series(dtype=float)

    scores = {}
    for d, g in df_.groupby("day"):
        score = 0.0  # 严格口径：默认 0 分
        if len(g) >= min_crosssec_n:
            yhat_u = g["yhat"].nunique()
            uniq_ratio = yhat_u / len(g)
            if (uniq_ratio >= uniq_ratio_th) and (np.nanstd(g["yhat"]) >= 1e-8) \
               and (g["y"].nunique() >= 3) and (yhat_u >= 3):
                ic = spearmanr(g["y"], g["yhat"]).statistic
                if np.isfinite(ic):
                    score = float(ic)  # 签名 IC，不取绝对值
        scores[d] = score
    return pd.Series(scores).sort_index()

def long_short_returns(index, y_true, y_pred, q=0.2, min_n=20):
    """逐日 Top q – Bottom q 等权多空日收益（用 y_true 当日收益）"""
    idx_day = _groupby_day(index)
    df_ = (pd.DataFrame({"day": idx_day, "y": y_true, "yhat": y_pred})
             .replace([np.inf, -np.inf], np.nan)
             .dropna())
    if df_.empty:
        return pd.Series(dtype=float)
    out = {}
    for d, g in df_.groupby("day"):
        if len(g) < min_n or g["yhat"].nunique() < 3:
            continue
        top_th = g["yhat"].quantile(1 - q)
        bot_th = g["yhat"].quantile(q)
        long_leg  = g[g["yhat"] >= top_th]["y"]
        short_leg = g[g["yhat"] <= bot_th]["y"]
        if len(long_leg) < 3 or len(short_leg) < 3:
            continue
        out[d] = float(long_leg.mean() - short_leg.mean())
    return pd.Series(out).sort_index()

def sharpe_of_series(ret_series, ann_factor=252):
    if ret_series is None or len(ret_series) < 5:
        return np.nan
    mu = ret_series.mean()
    sd = ret_series.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    return float((mu / sd) * np.sqrt(ann_factor))

# ====== 训练用 fitness（签名IC均值；退化=0） ======
_train_index_for_metric = None
def set_train_index_for_metric(mi):
    """在 fit 前设置（MultiIndex: (trade_date, ts_code)）"""
    global _train_index_for_metric
    _train_index_for_metric = mi

def _fitness_daily_ic_signed_strict_zero_if_degen(y_true, y_pred, sample_weight):
    """
    目标：训练区间“逐日签名 RankIC 的均值”（严格口径，每日都计入；不达标=0）
    退化即 0 分：
      - 全局 std(y_pred) < 1e-8 （整体近常数）
      - “0 分的交易日”占比 > 50%（绝大多数日没法区分） → 0
      - 全部交易日都 0 → 0
    注：不做任何单日 IC 裁剪；不取绝对值。
    """
    import numpy as np, pandas as pd
    if _train_index_for_metric is None:
        return 0.0

    sub = (pd.DataFrame({"y": y_true, "yhat": y_pred}, index=_train_index_for_metric)
             .replace([np.inf, -np.inf], np.nan)
             .dropna())
    if sub.empty:
        return 0.0

    # 整体近常数 → 0
    if float(np.nanstd(sub["yhat"].values)) < 1e-8:
        return 0.0

    # # 严格口径逐日 IC（签名）
    # s_ic = daily_ic_series_strict(
    #     _train_index_for_metric, sub["y"].values, sub["yhat"].values,
    #     min_crosssec_n=50, uniq_ratio_th=0.05
    # )
    # 改为（传“dropna 后”的索引，长度和 y/yhat 完全一致）：
    s_ic = daily_ic_series_strict(
        sub.index, sub["y"].values, sub["yhat"].values,
        min_crosssec_n=50, uniq_ratio_th=0.05
    )

    if s_ic.empty:
        return 0.0

    zero_ratio = float((np.abs(s_ic.values) < 1e-12).mean())
    if zero_ratio > 0.5:
        return 0.0

    mean_ic = float(np.mean(s_ic.values))  # 带号均值
    # fitness 最小为 0，避免负值扰动进化（负方向会被 'neg' 算子自己翻转）
    return max(mean_ic, 0.0)

fitness_signed_strict = make_fitness(function=_fitness_daily_ic_signed_strict_zero_if_degen,
                                     greater_is_better=True)



func_set = (
    'add','sub','mul','div','log','sqrt','abs','neg','max','min',
    ts_lag1, ts_mean5, ts_mean10, cs_rank_pct
)

# ===================== 配置 & 训练 GP（以 Spearman/IC 为目标） =====================
# est = SymbolicRegressor(
#     population_size=1000,
#     generations=2,
#     tournament_size=40,
#     function_set=func_set,
#     metric='pearson',
#     parsimony_coefficient='auto',
#     p_crossover=0.85,
#     p_subtree_mutation=0.06,
#     p_hoist_mutation=0.02,
#     p_point_mutation=0.05,
#     max_samples=1.0,   # 必须（保证截面完整）
#     n_jobs=1,          # 必须（上下文为进程内全局）
#     random_state=37,
#     verbose=1,
#     stopping_criteria=0.2
# )

# # 训练
# est.set_context_index(df.loc[train_mask].index)
# est.fit(X_train, y_train, sample_weight=w_train)


# est.fit(X_train, y_train, sample_weight=w_train)

# 在创建 SymbolicRegressor 之前：把训练期的 MultiIndex 交给 fitness
set_train_index_for_metric(df.loc[train_mask].index)

est = SymbolicRegressor(
    population_size=1000,
    generations=2,
    tournament_size=40,
    function_set=func_set,              # 不删任何算子
    metric=fitness_signed_strict,       # ← 用“签名IC均值，退化=0”的严格口径
    parsimony_coefficient='auto',
    p_crossover=0.85,
    p_subtree_mutation=0.06,
    p_hoist_mutation=0.02,
    p_point_mutation=0.05,
    max_samples=1.0,                    # 保持完整截面
    n_jobs=1,                           # 必须（上下文是进程内全局）
    random_state=37,
    verbose=1,
    stopping_criteria=0.20
)

# 训练前设置“训练期”的函数上下文（你已有这行）
est.set_context_index(df.loc[train_mask].index)
est.fit(X_train, y_train, sample_weight=w_train)



# ===================== Factor Zoo：输出前10名程序（按 fitness_ 排序） =====================
def get_topk_programs(est, topk=10):
    """
    从 gplearn 的所有世代中收集 Program，按表达式去重后，
    按 fitness_（含简约惩罚）降序取前 topk
    """
    all_progs = []
    for gen in getattr(est, "_programs", []):  # 每一代的种群
        for p in gen:
            if p is None or getattr(p, "fitness_", None) is None:
                continue
            all_progs.append(p)

    # 兜底：至少包含当前最佳
    if not all_progs and getattr(est, "_program", None) is not None:
        all_progs = [est._program]

    # 按表达式字符串去重
    seen = set()
    uniq = []
    for p in all_progs:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)

    # 按 fitness_（已含 parsimony 惩罚）降序
    uniq.sort(key=lambda p: p.fitness_, reverse=True)
    return uniq[: min(topk, len(uniq))]

topk_programs = get_topk_programs(est, topk=10)

print("\n[Top 10 Factors by GP fitness]  (fitness_ includes parsimony penalty)")
for i, p in enumerate(topk_programs, 1):
    raw = getattr(p, "raw_fitness_", float("nan"))  # 某些版本可能没有 raw_fitness_
    print(f"#{i:02d}  fitness={p.fitness_:.6f}  raw={raw:.6f}  "
          f"length={getattr(p, 'length_', -1):>3}  depth={getattr(p, 'depth_', -1):>2}  :: {p}")


# ===================== 评估：整体 IC & 日度 IC =====================
def ic_spearman(y, yhat) -> float:
    r = spearmanr(y, yhat, nan_policy='omit')[0]
    return float(np.nan_to_num(r, nan=0.0))

# yhat_valid = est.predict(X_valid)
# yhat_test  = est.predict(X_test)

# 验证
est.set_context_index(df.loc[valid_mask].index)
yhat_valid = est.predict(X_valid)

# 测试
est.set_context_index(df.loc[test_mask].index)
yhat_test  = est.predict(X_test)

print(f"[VALID] IC(spearman) = {ic_spearman(y_valid, yhat_valid):.4f}")
print(f"[TEST ] IC(spearman) = {ic_spearman(y_test,  yhat_test):.4f}")

# ===================== 给前10因子分别算 VALID/TEST 的 RankIC =====================
# def _ic_for_programs(programs, X_valid, y_valid, X_test, y_test):
#     print("\n[Top 10 Factors: VALID/TEST IC]")
#     for i, p in enumerate(programs, 1):
#         # 直接用 Program.execute(X) 计算该因子的打分
#         yv = p.execute(X_valid)
#         yt = p.execute(X_test)
#         icv = ic_spearman(y_valid, yv)
#         ict = ic_spearman(y_test,  yt)
#         print(f"#{i:02d}  VALID_IC={icv:.4f}  TEST_IC={ict:.4f}")

# _ic_for_programs(topk_programs, X_valid, y_valid, X_test, y_test)


# def _ic_for_programs(programs, X_valid, y_valid, X_test, y_test,
#                      index_valid, index_test):
#     print("\n[Top 10 Factors: VALID/TEST IC]")
#     for i, p in enumerate(programs, 1):
#         # —— VALID：先设验证集上下文，再执行
#         set_function_context(build_context_from_index(index_valid))
#         yv = p.execute(X_valid)
#         icv = ic_spearman(y_valid, yv)

#         # —— TEST：切到测试集上下文，再执行
#         set_function_context(build_context_from_index(index_test))
#         yt = p.execute(X_test)
#         ict = ic_spearman(y_test, yt)

#         print(f"#{i:02d}  VALID_IC={icv:.4f}  TEST_IC={ict:.4f}")

# _ic_for_programs(
#     topk_programs,
#     X_valid, y_valid,
#     X_test,  y_test,
#     df.loc[valid_mask].index,
#     df.loc[test_mask].index
# )
def report_programs(programs, X_v, y_v, idx_v, X_t, y_t, idx_t, topn=10, ann_factor=252):
    print("\n[Top Programs: VALID / TEST metrics]")
    for i, p in enumerate(programs[:topn], 1):
        # VALID
        set_function_context(build_context_from_index(idx_v))
        yhat_v = p.execute(X_v)
        s_v = daily_ic_series_strict(idx_v, y_v, yhat_v,
                                     min_crosssec_n=50, uniq_ratio_th=0.05)
        ic_v = float(np.mean(s_v.values)) if len(s_v) else float("nan")
        zero_rt_v = float((np.abs(s_v.values) < 1e-12).mean()) if len(s_v) else float("nan")
        ret_v = long_short_returns(idx_v, y_v, yhat_v, q=0.2, min_n=20)
        sharpe_v = sharpe_of_series(ret_v, ann_factor=ann_factor)

        # TEST
        set_function_context(build_context_from_index(idx_t))
        yhat_t = p.execute(X_t)
        s_t = daily_ic_series_strict(idx_t, y_t, yhat_t,
                                     min_crosssec_n=50, uniq_ratio_th=0.05)
        ic_t = float(np.mean(s_t.values)) if len(s_t) else float("nan")
        zero_rt_t = float((np.abs(s_t.values) < 1e-12).mean()) if len(s_t) else float("nan")
        ret_t = long_short_returns(idx_t, y_t, yhat_t, q=0.2, min_n=20)
        sharpe_t = sharpe_of_series(ret_t, ann_factor=ann_factor)

        print(
          f"#{i:02d}  VALID: IC={ic_v:+.4f}  Sharpe={sharpe_v:+.2f}  [零分日占比={zero_rt_v:.2f}]"
          f"   ||   TEST: IC={ic_t:+.4f}  Sharpe={sharpe_t:+.2f}  [零分日占比={zero_rt_t:.2f}]"
        )

# 调用（替换你原来的逐因子打印）
report_programs(
    topk_programs,
    X_valid, y_valid, df.loc[valid_mask].index,
    X_test,  y_test,  df.loc[test_mask].index,
    topn=10, ann_factor=252
)






def daily_ic_mean(mask: pd.Series, yhat: np.ndarray) -> float:
    sub = df.loc[mask, :].copy()
    sub['y'] = sub['ret_fwd1'].values
    sub['yhat'] = yhat
    out = []
    n_days = sub.index.get_level_values(0).nunique()
    for _, g in tqdm(sub.groupby(level=0), total=n_days, desc="Compute daily IC"):
        if g['y'].nunique() > 1 and g['yhat'].nunique() > 1:
            out.append(spearmanr(g['y'], g['yhat']).statistic)
    return float(np.nanmean(out)) if out else 0.0

print(f"[VALID] Daily IC mean = {daily_ic_mean(valid_mask, yhat_valid):.4f}")
print(f"[TEST ] Daily IC mean = {daily_ic_mean(test_mask,  yhat_test):.4f}")
