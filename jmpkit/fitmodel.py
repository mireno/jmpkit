from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import numpy as np, pandas as pd
from math import sqrt
from scipy import stats as _scipy_stats  # if SciPy may be missing, you can guard this import like others

_HAVE_SM = _HAVE_SK = _HAVE_LIFE = False
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.multivariate.manova import MANOVA
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    _HAVE_SM = True
except Exception:
    sm = smf = MANOVA = OrderedModel = None
try:
    from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.cross_decomposition import PLSRegression
    _HAVE_SK = True
except Exception:
    ElasticNetCV = LogisticRegressionCV = StandardScaler = make_pipeline = PLSRegression = None
try:
    from lifelines import CoxPHFitter, WeibullAFTFitter
    try:
        from lifelines import LogNormalAFTFitter, LogLogisticAFTFitter
    except Exception:
        LogNormalAFTFitter = LogLogisticAFTFitter = None
    _HAVE_LIFE = True
except Exception:
    CoxPHFitter = WeibullAFTFitter = LogNormalAFTFitter = LogLogisticAFTFitter = None

from .utils import q, as_list, require

def _rhs_from_effects(effects: List[str], degree: int = 1, cross: bool = False, add_poly: bool = False) -> str:
    if not effects: return "1"
    base = " + ".join(effects); parts = [base]
    if cross and len(effects) > 1:
        inter = " + ".join([f"{a}:{b}" for i, a in enumerate(effects) for b in effects[i+1:]])
        parts.append(inter)
    if add_poly and degree and degree > 1:
        parts.append(" + ".join([f"I({e}**{k})" for e in effects for k in range(2, degree+1)]))
    return " + ".join([p for p in parts if p]) or "1"

def _aft_fitter_from_name(name: str):
    n = (name or "weibull").lower()
    mapping = {
        "weibull": WeibullAFTFitter,
        "lognormal": LogNormalAFTFitter,
        "loglogistic": LogLogisticAFTFitter,
        "exponential": WeibullAFTFitter,
    }
    cls = mapping.get(n)
    if cls is None:
        raise ValueError(f"AFT distribution '{name}' not supported by this lifelines version.")
    return cls

@dataclass
class FitResult:
    personality: str
    formula: Optional[str]
    models: Dict[str, Any]
    summaries: Dict[str, Any]
    metrics: Dict[str, Any]

# --- ADD THESE HELPERS ANYWHERE ABOVE fit_model() ---
def _anova_tables_from_res(res):
    # Core sums of squares
    ssr = float(getattr(res, "ssr", np.nan))       # SSE (residual)
    ess = float(getattr(res, "ess", np.nan))       # SSR (model/explained)
    tss = float(getattr(res, "centered_tss", ess + ssr if np.isfinite(ess+ssr) else np.nan))

    df_m = float(res.df_model)
    df_e = float(res.df_resid)
    df_t = df_m + df_e

    ms_m = ess / df_m if df_m > 0 else np.nan
    ms_e = ssr / df_e if df_e > 0 else np.nan
    fval = ms_m / ms_e if (np.isfinite(ms_m) and np.isfinite(ms_e) and ms_e > 0) else np.nan
    try:
        pval = float(1.0 - _scipy_stats.f.cdf(fval, df_m, df_e))
    except Exception:
        pval = float(getattr(res, "f_pvalue", np.nan))

    anova = {
        "Model": {"DF": df_m, "Sum of Squares": ess, "Mean Square": ms_m, "F Ratio": fval, "Prob > F": pval},
        "Error": {"DF": df_e, "Sum of Squares": ssr, "Mean Square": ms_e},
        "C. Total": {"DF": df_t, "Sum of Squares": tss},
    }
    return anova, {"SSR": ess, "SSE": ssr, "SST": tss, "DFM": df_m, "DFE": df_e, "DFT": df_t, "MSM": ms_m, "MSE": ms_e, "F": fval, "p": pval}

def _lack_of_fit_table(df_used: pd.DataFrame, yname: str, xnames: List[str], sse: float, dfe: float, tss: float):
    """
    JMP-like Lack of Fit vs Pure Error. Works for multiple regression by grouping on all X columns.
    Returns (table_dict, max_rsq, note)
    """
    # If no regressors or no duplication -> LOF undefined
    if not xnames or len(xnames) == 0:
        return None, None, "No regressors — lack-of-fit not defined."

    # Build a grouping key across all X columns (exact match on values)
    # Only rows used in the fit (df_used) are considered.
    g = df_used.groupby(xnames, dropna=False, sort=False)

    # Count replicates per unique X row
    sizes = g.size()
    if (sizes <= 1).all():
        return None, None, "No replicated X rows — lack-of-fit not computable."

    # Pure Error SS: within-group deviations from each group's mean of Y
    # y - y_bar_group
    def _pure_ss(sub):
        yy = pd.to_numeric(sub[yname], errors="coerce").to_numpy()
        if yy.size == 0: return 0.0
        mu = float(np.nanmean(yy))
        return float(np.nansum((yy - mu) ** 2))

    pure_ss = float(g.apply(_pure_ss).sum())
    df_pure = float((sizes - 1).clip(lower=0).sum())

    # Lack-of-Fit SS = Residual SS - Pure Error SS
    lof_ss = max(sse - pure_ss, 0.0)
    df_lof = max(dfe - df_pure, 0.0)

    ms_lof = (lof_ss / df_lof) if df_lof > 0 else np.nan
    ms_pure = (pure_ss / df_pure) if df_pure > 0 else np.nan
    F = (ms_lof / ms_pure) if (np.isfinite(ms_lof) and np.isfinite(ms_pure) and ms_pure > 0) else np.nan
    try:
        p = float(1.0 - _scipy_stats.f.cdf(F, df_lof, df_pure)) if np.isfinite(F) else np.nan
    except Exception:
        p = np.nan

    # Max R^2 shown by JMP
    max_rsq = float(1.0 - (pure_ss / tss)) if (tss > 0 and np.isfinite(pure_ss)) else np.nan

    tbl = {
        "Lack Of Fit": {"DF": df_lof, "Sum of Squares": lof_ss, "Mean Square": ms_lof, "F Ratio": F, "Prob > F": p},
        "Pure Error": {"DF": df_pure, "Sum of Squares": pure_ss, "Mean Square": ms_pure},
        "Total Error": {"DF": dfe, "Sum of Squares": sse},
    }
    return tbl, max_rsq, None

def _effect_tests_type3(res, drop_intercept=True):
    """Type III effect tests via statsmodels anova_lm typ=3, when available."""
    try:
        from statsmodels.stats.anova import anova_lm
        a3 = anova_lm(res, typ=3)  # DataFrame
        # Clean up: optionally drop Intercept row to mimic JMP "Effect Tests"
        if drop_intercept:
            a3 = a3.loc[[i for i in a3.index if str(i).lower() != "intercept"]]
        out = {}
        for name, row in a3.iterrows():
            out[str(name)] = {
                "Nparm": 1 if np.isfinite(row.get("df", np.nan)) else np.nan,  # JMP shows df as Nparm for simple terms
                "DF": float(row.get("df", np.nan)),
                "Sum of Squares": float(row.get("sum_sq", np.nan)),
                "Mean Square": float(row.get("mean_sq", float(row.get("sum_sq", np.nan)) / float(row.get("df", np.nan)) if float(row.get("df", 0)) else np.nan)),
                "F Ratio": float(row.get("F", np.nan)),
                "Prob > F": float(row.get("PR(>F)", np.nan)),
            }
        return out
    except Exception:
        return None

def _params_table(res):
    params = res.params
    bse = res.bse
    tvals = res.tvalues
    pvals = res.pvalues
    out = []
    for name in params.index:
        out.append({
            "Term": str(name),
            "Estimate": float(params[name]),
            "Std Error": float(bse.get(name, np.nan)),
            "t Ratio": float(tvals.get(name, np.nan)),
            "Prob>|t|": float(pvals.get(name, np.nan)),
        })
    return out

def _summary_of_fit(res, y_series: pd.Series):
    n = int(res.nobs)
    rsq = float(getattr(res, "rsquared", np.nan))
    rsq_adj = float(getattr(res, "rsquared_adj", np.nan))
    rmse = float(sqrt(getattr(res, "mse_resid", np.nan))) if np.isfinite(getattr(res, "mse_resid", np.nan)) else float(np.nan)
    mean_y = float(pd.to_numeric(y_series, errors="coerce").dropna().mean())
    return {"RSquare": rsq, "RSquare Adj": rsq_adj, "Root Mean Square Error": rmse, "Mean of Response": mean_y, "Observations": n}

def _df_used_for_fit(res, original_df):
    # Try to recover the exact frame used by patsy/statsmodels (includes dummy coding etc.)
    try:
        design = res.model.data.frame
        # Merge back Y from original df by index if needed
        return design
    except Exception:
        # Fall back: use original df rows that had non-missing on endog/exog columns
        try:
            endog_name = res.model.endog_names
            exog_names = list(res.model.exog_names)
            cols = [c for c in [endog_name] + exog_names if c in original_df.columns]
            return original_df.loc[:, cols].dropna()
        except Exception:
            return original_df


def fit_model(df: pd.DataFrame, y: Union[str, List[str]], effects: List[str], *,
              personality: str = "standard_least_squares", degree: int = 1,
              cross: bool = False, add_poly: bool = False,
              weight: Optional[str] = None, freq: Optional[str] = None, by: Optional[str] = None,
              family: Optional[str] = None, link: Optional[str] = None,
              groups: Optional[str] = None, re_formula: str = "1",
              stepwise_direction: str = "both", max_steps: int = 50,
              l1_ratio: Optional[float] = None,
              duration_col: Optional[str] = None, event_col: Optional[str] = None,
              aft_distribution: str = "weibull",
              n_components: Optional[int] = None) -> FitResult:

    ys = as_list(y)
    rhs = _rhs_from_effects(effects, degree=degree, cross=cross, add_poly=add_poly)
    w = df[weight] if weight else None
    if freq is not None: print("Note: 'freq' is acknowledged but not expanded in this implementation.")

    models: Dict[str, Any] = {}; summaries: Dict[str, Any] = {}; metrics: Dict[str, Any] = {}

    if personality in ("standard_least_squares", "ols", "least_squares"):
        require(_HAVE_SM, "statsmodels is required for OLS.")
        for yi in ys:
            formula = f'{q(yi)} ~ ' + " + ".join([q(e) for e in effects]) if effects else f"{q(yi)} ~ 1"
            used_formula = True
            try:
                res = smf.ols(formula, data=df).fit()
                ok = hasattr(res, "model") and hasattr(res.model, "exog") and res.model.exog.shape[1] > 0
            except Exception:
                ok, res = False, None
            if not ok:
                # Fallback numeric/dummy path (unchanged)
                used_formula = False
                X = df[effects].copy() if effects else pd.DataFrame(index=df.index)
                non_num = X.select_dtypes(exclude=[np.number]).columns
                X = pd.get_dummies(X, drop_first=False, dtype=float) if len(non_num) else X.apply(pd.to_numeric, errors="coerce")
                yv = pd.to_numeric(df[yi], errors="coerce")
                mask = yv.notna() & (X.notna().all(1) if not X.empty else True)
                X, yv = X.loc[mask], yv.loc[mask]
                X = sm.add_constant(X, hasconst="add")
                res = sm.OLS(yv, X).fit()

            models[yi] = res
            summaries[yi] = res.summary()
            metrics[yi] = {"AIC": float(getattr(res, "aic", np.nan)),
                           "BIC": float(getattr(res, "bic", np.nan)),
                           "R2":  float(getattr(res, "rsquared", np.nan))}

            # --- NEW: JMP-like tables ---
            try:
                # Data actually used in fit
                df_used = _df_used_for_fit(res, df).copy()
                # Ensure Y column present for helper calcs
                if yi not in df_used.columns:
                    df_used[yi] = df[yi]

                # Core ANOVA
                anova_tbl, core = _anova_tables_from_res(res)

                # Lack-of-Fit (only if replicates exist)
                lof_tbl, max_rsq, note = _lack_of_fit_table(
                    df_used=df_used,
                    yname=yi,
                    xnames=[c for c in effects] if effects else [],
                    sse=core["SSE"], dfe=core["DFE"], tss=core["SST"]
                )

                # Effect Tests (Type III) — only reliable when the model came from a formula
                eff_tests = _effect_tests_type3(res, drop_intercept=True) if used_formula else None

                jmp_block = {
                    "Summary of Fit": _summary_of_fit(res, df[yi]),
                    "Analysis of Variance": anova_tbl,
                    "Lack Of Fit": (lof_tbl if lof_tbl is not None else {"note": note}),
                    "Max RSquare": (max_rsq if max_rsq is not None else np.nan),
                    "Parameter Estimates": _params_table(res),
                    "Effect Tests": (eff_tests if eff_tests is not None else {"note": "Type III requires a formula fit; unavailable in matrix-fit fallback."})
                }
                summaries[f"{yi}::jmp"] = jmp_block
            except Exception as _e:
                # Keep the fit even if JMP block fails; expose a minimal note
                summaries[f"{yi}::jmp"] = {"note": f"JMP-style tables unavailable: {_e}"}

        return FitResult("OLS", f"{' + '.join(ys)} ~ {rhs}", models, summaries, metrics)

    if personality == "stepwise":
        require(_HAVE_SM, "statsmodels is required for stepwise OLS.")
        def aic_of(formula): return smf.ols(formula, data=df, weights=w).fit().aic
        selected, candidates, best_aic, steps = [], effects.copy(), np.inf, 0
        while steps < max_steps and candidates:
            steps += 1; improved = False
            fwd = sorted([(aic_of(f"{ys[0]} ~ " + ("+".join(selected + [c]) if selected else c)), c) for c in candidates])
            if fwd and fwd[0][0] + 1e-9 < best_aic:
                best_aic, c = fwd[0]; selected.append(c); candidates.remove(c); improved = True
            if stepwise_direction in ("both","backward") and selected:
                bwd = sorted([(aic_of(f"{ys[0]} ~ " + ("+".join([z for z in selected if z != c]) or "1")), c) for c in list(selected)])
                if bwd and bwd[0][0] + 1e-9 < best_aic:
                    best_aic, worst = bwd[0]; selected.remove(worst); improved = True
            if not improved: break
        final_rhs = "+".join(selected) or "1"
        for yi in ys:
            res = smf.ols(f"{yi} ~ {final_rhs}", data=df, weights=w).fit()
            models[yi] = res; summaries[yi] = res.summary()
            metrics[yi] = {"AIC": float(res.aic), "BIC": float(res.bic), "R2": float(res.rsquared)}
            # NEW: attach JMP-like block to stepwise final model as well
            try:
                df_used = _df_used_for_fit(res, df)
                anova_tbl, core = _anova_tables_from_res(res)
                lof_tbl, max_rsq, note = _lack_of_fit_table(df_used, yi, selected, core["SSE"], core["DFE"], core["SST"])
                eff_tests = _effect_tests_type3(res, drop_intercept=True)
                summaries[f"{yi}::jmp"] = {
                    "Summary of Fit": _summary_of_fit(res, df[yi]),
                    "Analysis of Variance": anova_tbl,
                    "Lack Of Fit": (lof_tbl if lof_tbl is not None else {"note": note}),
                    "Max RSquare": (max_rsq if max_rsq is not None else np.nan),
                    "Parameter Estimates": _params_table(res),
                    "Effect Tests": (eff_tests if eff_tests is not None else {"note": "Effect tests unavailable."})
                }
            except Exception as _e:
                summaries[f"{yi}::jmp"] = {"note": f"JMP-style tables unavailable: {_e}"}
        return FitResult("Stepwise(OLS)", f"{' + '.join(ys)} ~ {final_rhs}", models, summaries, metrics)

    if personality == "glm":
        require(_HAVE_SM, "statsmodels is required for GLM.")
        fam_map = {"gaussian": sm.families.Gaussian, "binomial": sm.families.Binomial,
                   "poisson": sm.families.Poisson, "gamma": sm.families.Gamma,
                   "inverse_gaussian": sm.families.InverseGaussian,
                   "nb": getattr(sm.families, "NegativeBinomial", None)}
        require(family in fam_map and fam_map[family] is not None, f"Unsupported GLM family '{family}'.")
        fam = fam_map[family](); 
        if link:
            links = {"identity": sm.families.links.identity(), "log": sm.families.links.log(),
                     "logit": sm.families.links.logit(), "probit": sm.families.links.probit(),
                     "cloglog": sm.families.links.cloglog()}
            require(link in links, f"Unsupported GLM link '{link}'."); fam.link = links[link]
        for yi in ys:
            res = smf.glm(f"{yi} ~ {rhs}", data=df, family=fam).fit()
            models[yi] = res; summaries[yi] = res.summary()
            metrics[yi] = {"AIC": float(res.aic),
                           "BIC": float(res.aic + (np.log(len(res.fittedvalues))-2)*res.df_model)}
        return FitResult("GLM", f"{' + '.join(ys)} ~ {rhs}", models, summaries, metrics)

    if personality in ("generalized_regression","penalized"):
        require(_HAVE_SK, "scikit-learn is required for penalized regression.")
        X = df[effects].to_numpy()
        for yi in ys:
            yv = df[yi].to_numpy()
            if np.array_equal(np.unique(yv), [0,1]) or len(np.unique(yv)) <= 3:
                clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, cv=5, max_iter=2000, penalty="l2",
                                                                          solver="lbfgs", multi_class="auto")).fit(X, yv)
                models[yi] = clf; summaries[yi] = {"coef": getattr(clf[-1], "coef_", None),
                                                   "intercept": getattr(clf[-1], "intercept_", None)}
                metrics[yi] = {"CV_score_mean": float(np.mean(clf[-1].scores_[list(clf[-1].scores_.keys())[0]].mean(axis=0)))}
            else:
                enet = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=(l1_ratio or np.linspace(0.05,0.95,7)),
                                                                   alphas=None, cv=5, max_iter=5000)).fit(X, yv)
                models[yi] = enet; summaries[yi] = {"alpha": float(enet[-1].alpha_), "l1_ratio": float(enet[-1].l1_ratio_)}
                metrics[yi] = {"R2": float(enet.score(X, yv))}
        return FitResult("Generalized Regression", None, models, summaries, metrics)

    if personality == "mixed_model":
        require(_HAVE_SM, "statsmodels is required for MixedLM."); require(groups is not None, "Provide 'groups'.")
        for yi in ys:
            res = smf.mixedlm(f"{yi} ~ {rhs}", data=df, groups=df[groups], re_formula=re_formula).fit()
            models[yi] = res; summaries[yi] = res.summary(); metrics[yi] = {"AIC": float(res.aic), "BIC": float(res.bic)}
        return FitResult("MixedLM", f"{' + '.join(ys)} ~ {rhs} || groups={groups}", models, summaries, metrics)

    if personality == "manova":
        require(_HAVE_SM, "statsmodels is required for MANOVA."); require(len(ys) >= 2, "MANOVA needs 2+ Y.")
        formula = f"{' + '.join(ys)} ~ {rhs}"; mv = MANOVA.from_formula(formula, data=df)
        return FitResult("MANOVA", formula, {"MANOVA": mv}, {"MANOVA": mv.mv_test()}, {})

    if personality == "nominal_logistic":
        require(_HAVE_SM, "statsmodels is required for multinomial logistic regression.")
        yi = ys[0]; res = smf.mnlogit(f"{yi} ~ {rhs}", data=df).fit()
        return FitResult("Multinomial Logit", f"{yi} ~ {rhs}", {yi: res}, {yi: res.summary()}, {"AIC": float(res.aic)})

    if personality == "ordinal_logistic":
        require(_HAVE_SM, "statsmodels is required for ordinal logistic regression.")
        yi = ys[0]; endog = df[yi]; exog = sm.add_constant(df[effects], has_constant="add")
        res = OrderedModel(endog, exog, distr="logit").fit(method="bfgs")
        return FitResult("Ordered Logit", f"{yi} ~ {rhs}", {yi: res}, {yi: res.summary()}, {"AIC": float(res.aic)})

    if personality == "proportional_hazards":
        require(_HAVE_LIFE, "lifelines is required for CoxPH."); require(duration_col and event_col, "Provide duration_col & event_col.")
        df_surv = df[[duration_col, event_col] + effects].dropna()
        cph = CoxPHFitter().fit(df_surv, duration_col=duration_col, event_col=event_col, formula=" + ".join(effects))
        return FitResult("CoxPH", f"Survival({duration_col},{event_col}) ~ {rhs}",
                         {"CoxPH": cph}, {"CoxPH": cph.summary}, {"concordance": float(cph.concordance_index_)})

    if personality == "partial_least_squares":
        require(_HAVE_SK, "scikit-learn is required for PLSRegression.")
        X = df[effects].to_numpy()
        Y = df[ys].to_numpy() if len(ys) > 1 else df[ys[0]].to_numpy().reshape(-1,1)
        n_comp = n_components or min(X.shape[1], (Y.shape[1] if Y.ndim>1 else 1), 2)
        pls = PLSRegression(n_components=n_comp).fit(X, Y)
        return FitResult("PLSRegression", None, {"PLS": pls},
                         {"PLS": {"x_weights": pls.x_weights_, "y_weights": pls.y_weights_}},
                         {"R2_X": float(np.var(pls.x_scores_, ddof=1).sum() / np.var(X, ddof=1).sum())})

    if personality == "response_screening":
        require(_HAVE_SM, "statsmodels is required for screening with OLS.")
        rows = []
        for yi in ys:
            for x in effects:
                try:
                    res = smf.ols(f"{yi} ~ {x}", data=df).fit()
                    rows.append({"Y": yi, "X": x, "slope": res.params.get(x, np.nan),
                                 "p": res.pvalues.get(x, np.nan), "R2": res.rsquared, "AIC": res.aic})
                except Exception:
                    rows.append({"Y": yi, "X": x, "slope": np.nan, "p": np.nan, "R2": np.nan, "AIC": np.nan})
        tbl = pd.DataFrame(rows).sort_values(["Y","p","AIC"])
        return FitResult("Response Screening", None, {"screen": tbl}, {"screen": tbl}, {})

    raise ValueError(f"Unsupported personality: {personality}")

def fit_ols(df, y, effects, **kw):  return fit_model(df, y, effects, personality="standard_least_squares", **kw)
def fit_stepwise(df, y, effects, direction="both", max_steps=50, **kw):
    return fit_model(df, y, effects, personality="stepwise",
                     stepwise_direction=direction, max_steps=max_steps, **kw)
def fit_glm(df, y, effects, *, family, link=None, **kw):
    return fit_model(df, y, effects, personality="glm", family=family, link=link, **kw)
def fit_generalized_regression(df, y, effects, l1_ratio=None, **kw):
    return fit_model(df, y, effects, personality="generalized_regression", l1_ratio=l1_ratio, **kw)
def fit_mixed_model(df, y, effects, groups, re_formula="1", **kw):
    return fit_model(df, y, effects, personality="mixed_model", groups=groups, re_formula=re_formula, **kw)
def fit_manova(df, y_list, effects, **kw): return fit_model(df, y_list, effects, personality="manova", **kw)
def fit_nominal_logistic(df, y, effects, **kw): return fit_model(df, y, effects, personality="nominal_logistic", **kw)
def fit_ordinal_logistic(df, y, effects, **kw): return fit_model(df, y, effects, personality="ordinal_logistic", **kw)
def fit_proportional_hazards(df, effects, *, duration_col, event_col, **kw):
    return fit_model(df, [], effects, personality="proportional_hazards", duration_col=duration_col, event_col=event_col, **kw)
def fit_parametric_survival(df, effects, *, duration_col, event_col, aft_distribution="weibull", **kw):
    # stub hooks into fit_model if you add parametric AFT later
    return fit_model(df, [], effects, personality="parametric_survival",
                     duration_col=duration_col, event_col=event_col, aft_distribution=aft_distribution, **kw)
def fit_pls(df, y, effects, n_components=None, **kw):
    return fit_model(df, y, effects, personality="partial_least_squares", n_components=n_components, **kw)
def fit_response_screening(df, y_list, effects, **kw):
    return fit_model(df, y_list, effects, personality="response_screening", **kw)
