from __future__ import annotations
from typing import Dict, Any, List
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from .utils import (
    pairwise_counts, data_ellipse, safe_xy, chi2_ucl,
    display_dataframe_to_user, flier_color_vector
)

def jmp_scatterplot_matrix(df: pd.DataFrame, columns=None, k_sd_ellipse=1.0, marker_size=15,
                           upper="scatter", diag="label"):
    cols = list(df.select_dtypes(include=[np.number]).columns) if columns is None \
           else [c for c in columns if c in df.columns]
    p = len(cols)
    if p < 2:
        raise ValueError("Need at least two numeric columns.")

    fig, axes = plt.subplots(p, p, figsize=(2.6*p, 2.6*p), constrained_layout=True)

    for i in range(p):
        for j in range(p):
            ax = axes[i, j]

            if i == j:
                ax.axis("off")
                if diag == "label":
                    ax.text(0.5, 0.5, cols[i], ha="center", va="center", fontsize=11)
                continue

            x, y, ok_scatter, ok_ellipse = safe_xy(df, cols[j], cols[i])
            if not ok_scatter:
                ax.axis("off")
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
            else:
                # Color points by tag/color for either variable in this cell
                mini = pd.DataFrame({
                    cols[j]: pd.to_numeric(df[cols[j]], errors="coerce"),
                    cols[i]: pd.to_numeric(df[cols[i]], errors="coerce")
                }).dropna()
                xv, yv = mini[cols[j]].to_numpy(), mini[cols[i]].to_numpy()
                mfl, cols_vec = flier_color_vector(mini, [cols[j], cols[i]])

                if mfl.any():
                    ax.scatter(xv[~mfl], yv[~mfl], s=marker_size)
                    ax.scatter(xv[mfl],  yv[mfl],  s=marker_size, c=cols_vec)
                else:
                    ax.scatter(xv, yv, s=marker_size)

                if ok_ellipse:
                    data_ellipse(ax, x, y, k=k_sd_ellipse)

            if i < p-1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(cols[j])
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(cols[i])

    fig.suptitle("Scatterplot Matrix", y=1.02, fontsize=14)
    return fig, cols

def jmp_multivariate_panel_full(df: pd.DataFrame,
                                columns: List[str] | None = None,
                                alpha: float = 0.025,
                                upper: str = "scatter",
                                diag: str = "label") -> Dict[str, Any]:
    # 1) Scatterplot matrix + correlations/pairwise N
    fig_matrix, cols = jmp_scatterplot_matrix(df, columns=columns, upper=upper, diag=diag)
    corr_df = df[cols].corr(method="pearson", min_periods=2).round(4)
    n_pairs_df = pairwise_counts(df, cols)
    display_dataframe_to_user("Multivariate — Pearson Correlations (r)", corr_df)
    display_dataframe_to_user("Multivariate — Pairwise N", n_pairs_df)

    # 2) Complete-case numeric matrix for MD
    X = pd.DataFrame(df[cols]).apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    # remove columns without variability or too many NaNs after cleaning
    if not X.empty:
        var = X.var(axis=0, ddof=1)
        X = X.loc[:, var > 0]
    n = int(X.shape[0]); p = int(X.shape[1])

    if p < 2 or n < 2:
        md_table = pd.DataFrame({"row_index": [], "Mahalanobis_D": [], "Outlier": []})
        md_fig = plt.figure(); ax = plt.gca(); ax.axis("off")
        msg = []
        if p < 2: msg.append("Mahalanobis requires ≥2 usable numeric columns after cleaning.")
        if n < 2: msg.append("Not enough complete rows (need ≥2).")
        ax.text(0.5, 0.5, "\n".join(msg), ha="center", va="center")

        display_dataframe_to_user("Mahalanobis Distances — Table", md_table)
        return {
            "fig_matrix": fig_matrix, "corr_df": corr_df, "n_pairs_df": n_pairs_df,
            "md_table": md_table, "md_fig": md_fig, "columns": list(X.columns), "UCL": np.nan
        }

    # 3) Classical Mahalanobis Distance
    mu = X.mean(axis=0).to_numpy(dtype=float)
    A = X.to_numpy(dtype=float) - mu
    S = np.cov(A, rowvar=False, ddof=1)
    Sinv = np.linalg.pinv(S)
    D2 = np.einsum("ij,jk,ik->i", A, Sinv, A, optimize=True)
    MD = np.sqrt(np.maximum(D2, 0.0))

    UCL = chi2_ucl(p, alpha)

    md_table = pd.DataFrame({
        "row_index": X.index,
        "Mahalanobis_D": MD,
        "Outlier": MD > UCL
    })

    # 4) Mahalanobis plot — NOW colored with per-channel tag colors
    md_fig = plt.figure(); ax = plt.gca()
    x_positions = np.arange(1, n + 1)

    # Use the *cleaned* matrix rows (X) to check tag membership across the columns actually used
    # If a row is tagged in ANY of these columns, color it with that channel's color
    mfl, cols_vec = flier_color_vector(X, list(X.columns))

    if mfl.any():
        ax.scatter(x_positions[~mfl], MD[~mfl], s=25)
        ax.scatter(x_positions[mfl],  MD[mfl],  s=25, c=cols_vec)
    else:
        ax.scatter(x_positions, MD, s=25)

    ax.axhline(UCL, linewidth=1)
    ax.text(n + max(1, int(0.02 * n)), UCL, f"UCL={UCL:.3f}", va="center")
    ax.set_title("Mahalanobis Distances")
    ax.set_xlabel("Row Number")
    ax.set_ylabel("Distance")
    plt.tight_layout()

    display_dataframe_to_user("Mahalanobis Distances — Table", md_table)

    return {
        "fig_matrix": fig_matrix, "corr_df": corr_df, "n_pairs_df": n_pairs_df,
        "md_table": md_table, "md_fig": md_fig, "columns": list(X.columns), "UCL": UCL
    }
