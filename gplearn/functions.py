"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
from joblib import wrap_non_picklable_objects
import pandas as pd
from dataclasses import dataclass
from typing import Optional

__all__ = ['make_function']


class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)


def make_function(*, function, name, arity, wrap=True):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except (ValueError, TypeError):
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity)
    return _Function(function=function,
                     name=name,
                     arity=arity)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))


add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
div2 = _Function(function=_protected_division, name='div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
tan1 = _Function(function=np.tan, name='tan', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1}


# # ====== TS/CS context & operators ======
# # --------- Global context used by TS/CS operators ----------

@dataclass
class GPContext:
    day_id: np.ndarray
    prev_index: np.ndarray
    n: int

_FUNCTION_CONTEXT: Optional[GPContext] = None

def set_function_context(ctx: GPContext) -> None:
    global _FUNCTION_CONTEXT
    _FUNCTION_CONTEXT = ctx

def build_context_from_index(index) -> GPContext:
    dates = index.get_level_values(0)
    codes = index.get_level_values(1)
    day_id, _ = pd.factorize(dates, sort=False)
    day_id = day_id.astype(np.int64)
    n = len(index)
    prev_index = np.full(n, -1, dtype=np.int64)
    last_pos = {}
    for i in range(n):
        c = codes[i]; prev_index[i] = last_pos.get(c, -1); last_pos[c] = i
    return GPContext(day_id=day_id, prev_index=prev_index, n=n)

def _ts_lag1(x: np.ndarray) -> np.ndarray:
    ctx = _FUNCTION_CONTEXT
    if ctx is None:
        return np.zeros_like(x, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    idx = ctx.prev_index; valid = idx >= 0
    out[valid] = x[idx[valid]]
    return out

def _ts_mean_w(x: np.ndarray, w: int) -> np.ndarray:
    ctx = _FUNCTION_CONTEXT
    if ctx is None:
        return np.zeros_like(x, dtype=np.float64)
    acc = np.nan_to_num(x, nan=0.0).astype(np.float64).copy()
    cnt = (~np.isnan(x)).astype(np.float64)
    idx = ctx.prev_index.copy()
    for _ in range(1, w):
        valid = idx >= 0
        if not np.any(valid): break
        xv = x[idx[valid]]
        acc[valid] += np.nan_to_num(xv, nan=0.0)
        cnt[valid] += (~np.isnan(xv)).astype(np.float64)
        idx[valid] = ctx.prev_index[idx[valid]]
        idx[~valid] = -1
    out = acc / (cnt + 1e-12); out[cnt == 0] = np.nan
    return out

def _ts_mean5(x):  return _ts_mean_w(x, 5)
def _ts_mean10(x): return _ts_mean_w(x, 10)

# Temporal difference (first-order difference)
def _ts_diff1(x: np.ndarray) -> np.ndarray:
    ctx = _FUNCTION_CONTEXT
    if ctx is None:
        return np.zeros_like(x, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    idx = ctx.prev_index
    valid = idx >= 0
    out[valid] = x[valid] - x[idx[valid]]
    return out

# Time-series standard deviation (sliding window)
def _ts_std_w(x: np.ndarray, w: int) -> np.ndarray:
    ctx = _FUNCTION_CONTEXT
    if ctx is None:
        return np.zeros_like(x, dtype=np.float64)
    
    #    Time-series standard deviation (sliding window)            
    mean = _ts_mean_w(x, w)
    
    # Moving average of squared differences
    x_sq = (x - mean) **2
    acc_sq = np.nan_to_num(x_sq, nan=0.0).astype(np.float64).copy()
    cnt_sq = (~np.isnan(x_sq)).astype(np.float64)
    idx = ctx.prev_index.copy()
    
    for _ in range(1, w):
        valid = idx >= 0
        if not np.any(valid):
            break
        xv = x_sq[idx[valid]]
        acc_sq[valid] += np.nan_to_num(xv, nan=0.0)
        cnt_sq[valid] += (~np.isnan(xv)).astype(np.float64)
        idx[valid] = ctx.prev_index[idx[valid]]
        idx[~valid] = -1
    
    var = acc_sq / (cnt_sq + 1e-12)
    var[cnt_sq == 0] = np.nan
    return np.sqrt(var)

# Window standard deviation shortcut function
def _ts_std5(x): return _ts_std_w(x, 5)
def _ts_std10(x): return _ts_std_w(x, 10)

# Time series accumulation
def _ts_cumsum(x: np.ndarray) -> np.ndarray:
    ctx = _FUNCTION_CONTEXT
    if ctx is None:
        return np.zeros_like(x, dtype=np.float64)
    
    n = ctx.n
    out = np.full_like(x, np.nan, dtype=np.float64)
    codes = ctx.prev_index  # Assuming prev_index contains grouping information
    last_sum = {}
    
    for i in range(n):
        c = codes[i]
        if c == -1:
            current_sum = x[i] if not np.isnan(x[i]) else 0.0
        else:
            prev_sum = last_sum.get(c, 0.0)
            current_sum = prev_sum + (x[i] if not np.isnan(x[i]) else 0.0)
        last_sum[c] = current_sum
        out[i] = current_sum
    
    return out


# def _cs_zscore(x: np.ndarray) -> np.ndarray:
#     ctx = _FUNCTION_CONTEXT
#     if ctx is None:
#         return np.zeros_like(x, dtype=np.float64)
#     gid = ctx.day_id
#     counts = np.bincount(gid, minlength=gid.max()+1).astype(np.float64)
#     sums   = np.bincount(gid, weights=np.nan_to_num(x, nan=0.0), minlength=counts.size)
#     xsq    = np.bincount(gid, weights=np.nan_to_num(x, nan=0.0)**2, minlength=counts.size)
#     mu = sums[gid] / (counts[gid] + 1e-12)
#     var = xsq[gid] / (counts[gid] + 1e-12) - mu**2
#     var[var < 0] = 0.0
#     std = np.sqrt(var) + 1e-12
#     return (x - mu) / std

def _cs_rank_pct(x: np.ndarray) -> np.ndarray:
    ctx = _FUNCTION_CONTEXT
    if ctx is None:
        return np.zeros_like(x, dtype=np.float64)
    gid = ctx.day_id; n = x.shape[0]
    order = np.lexsort((x, gid))  # stable sort by (gid, x)
    ranks = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        j = i + 1; g = gid[order[i]]
        while j < n and gid[order[j]] == g: j += 1
        seg = x[order[i:j]]
        seg_rank = np.argsort(np.argsort(seg, kind='mergesort'), kind='mergesort')
        ranks[order[i:j]] = seg_rank
        i = j
    group_sizes = np.bincount(gid, minlength=gid.max()+1)
    denom = group_sizes[gid] + 1.0
    return (ranks + 1.0) / denom


# Cross-sectional Median
def _cs_median(x: np.ndarray) -> np.ndarray:
    ctx = _FUNCTION_CONTEXT
    if ctx is None:
        return np.zeros_like(x, dtype=np.float64)
    
    gid = ctx.day_id
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    
# Calculate the median by group
    unique_groups = np.unique(gid)
    for g in unique_groups:
        mask = gid == g
        group_data = x[mask]
        median = np.nanmedian(group_data)
        out[mask] = median
    
    return out

# Cross-sectional Extreme Ratio (Maximum/Minimum)
def _cs_range_ratio(x: np.ndarray) -> np.ndarray:
    ctx = _FUNCTION_CONTEXT
    if ctx is None:
        return np.zeros_like(x, dtype=np.float64)
    
    gid = ctx.day_id
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    
    unique_groups = np.unique(gid)
    for g in unique_groups:
        mask = gid == g
        group_data = x[mask]
        valid_data = group_data[~np.isnan(group_data)]
        if len(valid_data) < 2:
            ratio = 1.0          
        else:
            max_val = np.max(valid_data)
            min_val = np.min(valid_data)
            ratio = max_val / min_val if np.abs(min_val) > 1e-12 else 0.0
        out[mask] = ratio
    
    return out


# —— 注册为 Function 节点（本文件前面已有 make_function）
ts_lag1     = make_function(function=_ts_lag1,     name="ts_lag1",     arity=1)
ts_mean5    = make_function(function=_ts_mean5,    name="ts_mean5",    arity=1)
ts_mean10   = make_function(function=_ts_mean10,   name="ts_mean10",   arity=1)
ts_diff1 = make_function(function=_ts_diff1, name="ts_diff1", arity=1)
ts_std5 = make_function(function=_ts_std5, name="ts_std5", arity=1)
ts_std10 = make_function(function=_ts_std10, name="ts_std10", arity=1)
ts_cumsum = make_function(function=_ts_cumsum, name="ts_cumsum", arity=1)

# cs_zscore   = make_function(function=_cs_zscore,   name="cs_zscore",   arity=1)
cs_rank_pct = make_function(function=_cs_rank_pct, name="cs_rank_pct", arity=1)
cs_median = make_function(function=_cs_median, name="cs_median", arity=1)
cs_range_ratio = make_function(function=_cs_range_ratio, name="cs_range_ratio", arity=1)


# CONTEXT_AWARE_FUNCS = (ts_lag1, ts_mean5, ts_mean10, cs_zscore, cs_rank_pct)
CONTEXT_AWARE_FUNCS = (ts_lag1, ts_mean5, ts_mean10, cs_rank_pct, ts_diff1, ts_std5, ts_std10, ts_cumsum, cs_median, cs_range_ratio)
