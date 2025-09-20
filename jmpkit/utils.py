from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def q(name: str) -> str:
    return f'Q("{name}")'

def as_list(x) -> List[str]:
    if x is None: return []
    if isinstance(x, (list, tuple, np.ndarray, pd.Index)): return list(x)
    return [x]

def require(cond: bool, msg: str):
    if not cond:
        raise ImportError(msg)

def sanitize(name: str) -> str:
    import re
    s = re.sub(r"\W+", "_", str(name)).strip("_")
    if s and s[0].isdigit():
        s = f"c_{s}"
    return s or "col"

def pairwise_counts(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    mask = df[cols].notna().astype(int)
    n_mat = mask.T.dot(mask)
    n_mat.index = cols; n_mat.columns = cols
    return n_mat

def safe_xy(df, xname, yname):
    x = pd.to_numeric(df[xname], errors="coerce").to_numpy()
    y = pd.to_numeric(df[yname], errors="coerce").to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    return x, y, (x.size > 0 and y.size > 0), (x.size >= 2 and y.size >= 2)

def data_ellipse(ax, x: np.ndarray, y: np.ndarray, k: float = 1.0):
    if x.size < 2: return
    X = np.column_stack([x, y])
    S = np.cov(X, rowvar=False, ddof=1)
    if not np.all(np.isfinite(S)) or np.linalg.matrix_rank(S) < 2: return
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, 0.0)
    order = np.argsort(vals)[::-1]
    vals = vals[order]; vecs = vecs[:, order]
    mu = X.mean(axis=0)
    t = np.linspace(0, 2*np.pi, 300)
    ell = mu + (vecs @ (np.diag(k*np.sqrt(vals)) @ np.vstack((np.cos(t), np.sin(t))))).T
    ax.plot(ell[:, 0], ell[:, 1], linewidth=1)

def chi2_ucl(p: int, alpha: float) -> float:
    try:
        from scipy.stats import chi2
        return float(np.sqrt(chi2.ppf(1 - alpha, df=p)))
    except Exception:
        z = 1.96
        q = p * (1 - 2/(9*p) + z*np.sqrt(2/(9*p)))**3
        return float(np.sqrt(q))

def display_dataframe_to_user(name: str, dataframe: pd.DataFrame,
                              show_index: bool = False, max_rows: int = 1000,
                              precision: int | None = None) -> None:
    df = dataframe
    truncated = False
    if len(df) > max_rows:
        df_to_show, truncated = df.head(max_rows), True
    else:
        df_to_show = df
    try:
        from IPython.display import display, HTML
        if precision is not None:
            old = pd.get_option("display.precision"); pd.set_option("display.precision", precision)
        try:
            caption = f"{name} — {len(df):,} rows × {df.shape[1]}"
            styler = df_to_show.style.set_caption(caption)
            if not show_index: styler = styler.hide(axis="index")
            display(styler)
            if truncated:
                display(HTML(f"<em>Showing first {max_rows} rows of {len(df):,}.</em>"))
        finally:
            if precision is not None: pd.set_option("display.precision", old)
    except Exception:
        print(f"\n{name} — {len(df)} rows × {df.shape[1]}")
        print(df_to_show.to_string(index=show_index))
        if truncated: print(f"... showing first {max_rows} rows.")
