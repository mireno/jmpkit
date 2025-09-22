from __future__ import annotations
from typing import Dict, Any
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from math import sqrt, log, pi
from .utils import display_dataframe_to_user

def _normal_pdf(x, mu, sigma):
    sigma = max(float(sigma), 1e-12)
    return (1.0 / (sigma * sqrt(2 * pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def jmp_distribution_report(df: pd.DataFrame, column: str, *, tag_fliers: bool = False) -> Dict[str, Any]:
    s = pd.to_numeric(df[column], errors="coerce")
    x = s.dropna().to_numpy()
    n, n_missing = x.size, int(s.isna().sum())
    if n == 0:
        raise ValueError(f"'{column}' has no non-missing numeric values.")

    # --- core summary stats ---
    mean = float(np.mean(x))
    std_sample = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    se_mean = float(std_sample / sqrt(n)) if n > 1 else float("nan")

    if n > 1:
        try:
            from scipy.stats import t
            tcrit = float(t.ppf(0.975, df=n-1))
        except Exception:
            tcrit = 1.96
        ci = (mean - tcrit*se_mean, mean + tcrit*se_mean)
    else:
        ci = (float("nan"), float("nan"))

    summary_df = pd.DataFrame({
        "Measure": ["Mean","Std Dev","Std Err Mean","Upper 95% Mean","Lower 95% Mean","N","N Missing"],
        "Value":   [mean,std_sample,se_mean,ci[1],ci[0],int(n),int(n_missing)]
    })

    # --- quantiles (JMP-like order) ---
    qs = [1.0,0.995,0.975,0.95,0.90,0.75,0.50,0.25,0.10,0.05,0.025,0.01,0.005,0.0]
    labels = ["maximum","99.5%","97.5%","95.0%","90.0%","quartile","median","quartile",
              "10.0%","5.0%","2.5%","1.0%","0.5%","minimum"]
    quantiles_df = pd.DataFrame({"Quantile": labels, "Value": pd.Series(x).quantile(q=qs).to_numpy()})

    # --- normal MLE fit (mu, sigma) ---
    mu_hat = mean
    sigma_mle = float(np.sqrt(np.mean((x - mu_hat) ** 2))) or 1e-12
    se_mu, se_sigma = sigma_mle/np.sqrt(n), sigma_mle/np.sqrt(2*n)
    z = 1.96
    fit_params_df = pd.DataFrame({
        "Parameter":["Location μ","Dispersion σ"],
        "Estimate":[mu_hat,sigma_mle],
        "Std Error":[se_mu,se_sigma],
        "Lower 95%":[mu_hat - z*se_mu, max(0.0, sigma_mle - z*se_sigma)],
        "Upper 95%":[mu_hat + z*se_mu, sigma_mle + z*se_sigma],
    })

    const = -0.5 * log(2 * pi * (sigma_mle ** 2))
    quad  = -0.5 * ((x - mu_hat) ** 2) / (sigma_mle ** 2)
    loglik = float(np.sum(const + quad)); k = 2
    fit_stats_df = pd.DataFrame({
        "Measure": ["-2*LogLikelihood", "AICc", "BIC"],
        "Value":   [-2*loglik,
                    (2*k - 2*loglik) + (2*k*(k+1))/(n - k - 1) if n > (k+1) else np.nan,
                    (k*np.log(n) - 2*loglik)]
    })

    # --- IQR-based fliers (JMP box-plot rule, 1.5*IQR) ---
    q1, q3 = np.nanpercentile(x, 25), np.nanpercentile(x, 75)
    iqr = q3 - q1
    lo, hi = (q1 - 1.5*iqr), (q3 + 1.5*iqr)
    f_mask = (s < lo) | (s > hi)
    f_indices = list(s.index[f_mask.fillna(False)])

    # --- plots ---
    # BEFORE (warns in non-interactive backends)
    # fig_hist = plt.figure()
    # plt.hist(x, bins="auto", density=True)
    # grid = np.linspace(x.min(), x.max(), 200)
    # plt.plot(grid, _normal_pdf(grid, mu_hat, sigma_mle))
    # plt.axvline(float(np.median(x)), linewidth=1)
    # plt.title(f"Distribution — {column}"); plt.xlabel(column); plt.ylabel("Density"); plt.show()

    # fig_box = plt.figure()
    # plt.boxplot(x, vert=True, whis=1.5, showfliers=True)
    # plt.title(f"Box Plot — {column}"); plt.ylabel(column); plt.show()

    # AFTER (backend-safe, no plt.show)
    fig_hist, axh = plt.subplots()
    axh.hist(x, bins="auto", density=True)
    grid = np.linspace(x.min(), x.max(), 200)
    axh.plot(grid, _normal_pdf(grid, mu_hat, sigma_mle))
    axh.axvline(float(np.median(x)), linewidth=1)
    axh.set_title(f"Distribution — {column}")
    axh.set_xlabel(column)
    axh.set_ylabel("Density")

    fig_box, axb = plt.subplots()
    axb.boxplot(x, vert=True, whis=1.5, showfliers=True)
    axb.set_title(f"Box Plot — {column}")
    axb.set_ylabel(column)

    # --- Normality goodness-of-fit: Shapiro–Wilk & Anderson–Darling ---
    have_scipy = True
    shapiro_W = np.nan
    shapiro_p = np.nan
    shapiro_note = ""
    ad_stat = np.nan
    ad_note = ""
    ad_levels = [15.0, 10.0, 5.0, 2.5, 1.0]
    ad_crit = [np.nan]*len(ad_levels)
    ad_reject_at_5 = np.nan

    try:
        from scipy.stats import shapiro, anderson
        try:
            W, p = shapiro(x)
            shapiro_W, shapiro_p = float(W), float(p)
            shapiro_note = ""
        except Exception as e:
            shapiro_note = f"Shapiro–Wilk not computed: {e}"

        try:
            ad_res = anderson(x, dist='norm')
            ad_stat = float(ad_res.statistic)
            ad_levels = list(map(float, ad_res.significance_level))
            ad_crit = list(map(float, ad_res.critical_values))
            if 5.0 in ad_levels:
                idx5 = ad_levels.index(5.0)
                ad_reject_at_5 = bool(ad_stat > ad_crit[idx5])
            else:
                ad_reject_at_5 = np.nan
            ad_note = ""
        except Exception as e:
            ad_note = f"Anderson–Darling not computed: {e}"

    except Exception:
        have_scipy = False
        shapiro_note = "SciPy not available; install scipy to compute Shapiro–Wilk."
        ad_note = "SciPy not available; install scipy to compute Anderson–Darling."

    # Summarized decision table at alpha = 0.05
    alpha = 0.05
    shapiro_reject = (shapiro_p <= alpha) if np.isfinite(shapiro_p) else np.nan
    gof_summary_df = pd.DataFrame([
        {"Test": "Shapiro–Wilk", "Statistic": shapiro_W, "p_value": shapiro_p,
         "Alpha": alpha, "Reject H0 (Normal)": shapiro_reject,
         "Note": shapiro_note},
        {"Test": "Anderson–Darling", "Statistic": ad_stat, "p_value": np.nan,
         "Alpha": alpha, "Reject H0 (Normal)": ad_reject_at_5,
         "Note": ad_note or "Reject if A² > critical value at 5%"}
    ])

    # Detailed AD critical values table (if available)
    ad_details_df = pd.DataFrame({
        "Significance %": ad_levels,
        "Critical Value": ad_crit,
        "Reject H0 (A² > crit)": [ (ad_stat > cv) if np.isfinite(ad_stat) and np.isfinite(cv) else np.nan
                                   for cv in ad_crit ]
    })

    # --- show dataframes in Jupyter contexts (no-op in Streamlit) ---
    display_dataframe_to_user(f"{column} — Quantiles", quantiles_df)
    display_dataframe_to_user(f"{column} — Summary Statistics", summary_df)
    display_dataframe_to_user(f"{column} — Fitted Normal Parameters", fit_params_df)
    display_dataframe_to_user(f"{column} — Fit Statistics", fit_stats_df)
    display_dataframe_to_user(f"{column} — Normality Tests (Shapiro & Anderson–Darling)", gof_summary_df)
    display_dataframe_to_user(f"{column} — Anderson–Darling Critical Values", ad_details_df)

    # Optional: persist fliers for this channel
    if tag_fliers and len(f_indices):
        try:
            from .utils import tag_fliers as _tag
            _tag(column, f_indices, mode="set")
        except Exception:
            pass

    return dict(
        fig_hist=fig_hist, fig_box=fig_box,
        quantiles_df=quantiles_df, summary_df=summary_df,
        fit_params_df=fit_params_df, fit_stats_df=fit_stats_df,
        gof_summary_df=gof_summary_df, ad_details_df=ad_details_df,
        flier_bounds=(lo, hi), flier_index=f_indices
    )

def jmp_distribution_all(df: pd.DataFrame, columns=None):
    cols = list(df.select_dtypes(include=[np.number]).columns) if columns is None \
           else [c for c in columns if c in df.columns]
    out = {}
    for c in cols:
        try:
            out[c] = jmp_distribution_report(df, c)
        except Exception as e:
            print(f"Skipping {c}: {e}")
    return out
