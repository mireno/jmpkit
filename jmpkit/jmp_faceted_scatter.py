# jmp_faceted_scatter.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import flier_mask, flier_color_vector  # NEW import

_HAVE_SM = False
try:
    import statsmodels.api as sm
    _HAVESM = _HAVE_SM = True
except Exception:
    sm = None

def plot_xy_by_group(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str | None = None,
    *,
    ncols: int | None = None,
    height: float = 4.0,
    width: float = 4.0,
):
    """
    Faceted scatter of y vs x, split by `group` with a best-fit line.
    All panels share the SAME y-axis (one scale), while each keeps its own x-axis.
    Tagged rows for X or Y are colored with the per-channel color.
    Returns (fig, axes).
    """
    data = df[[c for c in [x, y, group] if c is not None]].dropna()

    # overall y-limits (same for every facet)
    y_min, y_max = float(data[y].min()), float(data[y].max())
    if y_min == y_max:
        pad = 1.0 if y_max == 0 else 0.05 * abs(y_max)
        y_min, y_max = y_min - pad, y_max + pad

    if group is None:
        groups = [None]
    else:
        groups = list(pd.unique(data[group]))

    n = len(groups)
    if ncols is None:
        ncols = n  # single row like JMP; separate x-axes, shared y
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * width, nrows * height),
        squeeze=False,
        sharex=False,
        sharey=True
    )

    for i, g in enumerate(groups):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        sub = data if g is None else data[data[group] == g]
        if sub.empty:
            ax.set_visible(False)
            continue

        # Color by registry (per-channel colors)
        mfl, cols = flier_color_vector(sub, [x, y])
        if mfl.any():
            ax.scatter(sub[x][~mfl], sub[y][~mfl], s=14, alpha=0.8)
            ax.scatter(sub[x][ mfl], sub[y][ mfl], s=14, alpha=0.95, c=cols)
        else:
            ax.scatter(sub[x], sub[y], s=14, alpha=0.8)

        # Best-fit line (+ 95% CI if statsmodels available)
        if len(sub) >= 2:
            xs = np.linspace(sub[x].min(), sub[x].max(), 100)
            if _HAVE_SM:
                X = sm.add_constant(sub[x].to_numpy())
                res = sm.OLS(sub[y].to_numpy(), X).fit()
                Xp = sm.add_constant(xs)
                yhat = res.predict(Xp)
                try:
                    sf = res.get_prediction(Xp).summary_frame(alpha=0.05)
                    ax.fill_between(xs, sf["mean_ci_lower"], sf["mean_ci_upper"], alpha=0.2)
                except Exception:
                    pass
                ax.plot(xs, yhat, linewidth=2)
            else:
                m, b = np.polyfit(sub[x].to_numpy(), sub[y].to_numpy(), 1)
                ax.plot(xs, m * xs + b, linewidth=2)

        ax.set_ylim(y_min, y_max)
        ax.set_title(f"{y} vs {x}" if g is None else str(g))
        ax.set_xlabel(x)
        if c == 0:
            ax.set_ylabel(y)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    return fig, axes
