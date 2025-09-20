from __future__ import annotations
from typing import Dict, Any, List
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from .utils import pairwise_counts, data_ellipse, safe_xy, chi2_ucl, display_dataframe_to_user

def jmp_scatterplot_matrix(df: pd.DataFrame, columns=None, k_sd_ellipse=1.0, marker_size=15,
                           upper="scatter", diag="label"):
    cols = list(df.select_dtypes(include=[np.number]).columns) if columns is None \
           else [c for c in columns if c in df.columns]
    p = len(cols)
    if p < 2: raise ValueError("Need at least two numeric columns.")
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
                ax.axis("off"); ax.text(0.5, 0.5, "No data", ha="center", va="center")
            else:
                ax.scatter(x, y, s=marker_size)
                if ok_ellipse: data_ellipse(ax, x, y, k=k_sd_ellipse)
            if i < p-1: ax.set_xticklabels([])
            else:       ax.set_xlabel(cols[j])
            if j > 0:   ax.set_yticklabels([])
            else:       ax.set_ylabel(cols[i])
    fig.suptitle("Scatterplot Matrix", y=1.02, fontsize=14)
    return fig, cols

def jmp_multivariate_panel_full(df: pd.DataFrame,
                                columns: List[str] | None = None,
                                alpha: float = 0.025,
                                upper: str = "scatter",
                                diag: str = "label") -> Dict[str, Any]:
    # 1) Scatterplot matrix + correlations/pairwise N as before
    fig_matrix, cols = jmp_scatterplot_matrix(df, columns=columns, upper=upper, diag=diag)
    corr_df = df[cols].corr(method="pearson", min_periods=2).round(4)
    n_pairs_df = pairwise_counts(df, cols)
    display_dataframe_to_user("Multivariate — Pearson Correlations (r)", corr_df)
    display_dataframe_to_user("Multivariate — Pairwise N", n_pairs_df)

    # 2) Build complete-case numeric matrix for MD — robust handling
    #    (never let this collapse to a Series or 1-D)
    X = pd.DataFrame(df[cols]).apply(pd.to_numeric, errors="coerce")

    # Drop rows with any NaN across the selected columns
    X = X.dropna(axis=0, how="any")

    # Drop columns that are all-NaN or have < 2 non-NaN values (already unlikely after dropna)
    X = X.loc[:, X.notna().sum(axis=0) >= 2]

    # Drop columns with zero variance (constant), which make Σ singular and MD meaningless
    if not X.empty:
        var = X.var(axis=0, ddof=1)
        X = X.loc[:, var > 0]

    n = int(X.shape[0])
    p = int(X.shape[1])

    # If we no longer have at least 2 usable columns, show a friendly message
    if p < 2 or n < 2:
        md_table = pd.DataFrame({"row_index": [], "Mahalanobis_D": [], "Outlier": []})
        md_fig = plt.figure()
        ax = plt.gca()
        ax.axis("off")
        msg = []
        if p < 2:
            msg.append("Mahalanobis requires ≥2 usable numeric columns after cleaning.")
        if n < 2:
            msg.append("Not enough complete rows (need ≥2).")
        ax.text(0.5, 0.5, "\n".join(msg), ha="center", va="center")
        plt.show()
        display_dataframe_to_user("Mahalanobis Distances — Table", md_table)
        return {
            "fig_matrix": fig_matrix,
            "corr_df": corr_df,
            "n_pairs_df": n_pairs_df,
            "md_table": md_table,
            "md_fig": md_fig,
            "columns": list(X.columns),  # actually used (maybe fewer than requested)
            "UCL": np.nan,
        }

    # 3) Classical MD on the cleaned matrix
    mu = X.mean(axis=0).to_numpy(dtype=float)
    A = X.to_numpy(dtype=float) - mu
    # Σ with ddof=1; use pinv for safety
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

    md_fig = plt.figure()
    ax = plt.gca()
    ax.scatter(np.arange(1, n + 1), MD, s=25)
    ax.axhline(UCL, linewidth=1)
    ax.text(n + max(1, int(0.02 * n)), UCL, f"UCL={UCL:.3f}", va="center")
    ax.set_title("Mahalanobis Distances")
    ax.set_xlabel("Row Number")
    ax.set_ylabel("Distance")
    plt.show()

    display_dataframe_to_user("Mahalanobis Distances — Table", md_table)

    return {
        "fig_matrix": fig_matrix,
        "corr_df": corr_df,
        "n_pairs_df": n_pairs_df,
        "md_table": md_table,
        "md_fig": md_fig,
        "columns": list(X.columns),  # columns actually used for MD
        "UCL": UCL,
    }

