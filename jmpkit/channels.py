from __future__ import annotations
import numpy as np, pandas as pd
from types import SimpleNamespace
from .io import Dataset
from .utils import sanitize

def _is_dataset(obj): return isinstance(obj, Dataset)
def _get_df(obj): return obj.df if _is_dataset(obj) else obj

def _refresh_namespace(obj, df):
    if _is_dataset(obj):
        obj.df = df; obj.columns = list(df.columns)
        obj.col = SimpleNamespace(**{ sanitize(c): df[c] for c in df.columns })

def add_channel_expr(data_or_df, new_name: str, expression: str, *, overwrite: bool = False,
                     engine: str = "python", extra_env: dict | None = None) -> pd.Series:
    df = _get_df(data_or_df)
    if new_name in df.columns and not overwrite:
        raise ValueError(f"Column '{new_name}' already exists. Use overwrite=True.")
    env = {
        "np": np, "log": np.log, "log10": np.log10, "exp": np.exp, "sqrt": np.sqrt,
        "abs": np.abs, "where": np.where, "clip": np.clip, "maximum": np.maximum,
        "minimum": np.minimum, "isfinite": np.isfinite, "round": np.round
    }
    if extra_env: env.update(extra_env)
    out = pd.eval(expression, engine=engine, parser="pandas", local_dict=env, target=df)
    df[new_name] = out if np.isscalar(out) else pd.Series(out, index=df.index)
    _refresh_namespace(data_or_df, df)
    return df[new_name]

def add_channel_func(data_or_df, new_name: str, func, *, overwrite: bool = False) -> pd.Series:
    df = _get_df(data_or_df)
    if new_name in df.columns and not overwrite:
        raise ValueError(f"Column '{new_name}' already exists. Use overwrite=True.")
    out = func(df)
    df[new_name] = out if np.isscalar(out) else pd.Series(out, index=df.index)
    _refresh_namespace(data_or_df, df)
    return df[new_name]
