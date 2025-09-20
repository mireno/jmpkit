from __future__ import annotations
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from .fitmodel import fit_model
from .utils import q, flier_color_vector  # NEW

def jmp_fit_and_plots(df: pd.DataFrame, y, effects, *,
                      personality: str = "standard_least_squares",
                      regressor=None, degree: int = 1, cross: bool = False, add_poly: bool = False, **fit_kwargs):
    ys = [y] if isinstance(y, str) else list(y)
    fitres = fit_model(df, y=ys, effects=effects, personality=personality,
                       degree=degree, cross=cross, add_poly=add_poly, **fit_kwargs)
    models = fitres.models
    results = {}

    def _yhat_resid(yi, model):
        if hasattr(model, "fittedvalues") and hasattr(model, "resid") and hasattr(model, "model"):
            y_true = pd.Series(model.model.endog, index=model.model.data.row_labels)
            y_hat  = pd.Series(model.fittedvalues, index=model.model.data.row_labels)
            resid  = pd.Series(model.resid, index=model.model.data.row_labels)
            return y_true, y_hat, resid
        X = df[effects]
        y_true = pd.to_numeric(df[yi], errors="coerce")
        y_hat = pd.Series(model.predict(X), index=X.index)
        m = y_true.notna() & np.isfinite(y_hat)
        return y_true[m], y_hat[m], (y_true[m] - y_hat[m])

    def _baseline():
        base = {}
        for e in effects:
            s = df[e]
            base[e] = float(pd.to_numeric(s, errors="coerce").mean()) if pd.api.types.is_numeric_dtype(s) \
                      else (s.mode().iloc[0] if not s.mode().empty else s.dropna().iloc[0] if s.notna().any() else np.nan)
        return base

    def _line(model, xname, x_min, x_max, n=200):
        grid = np.linspace(float(x_min), float(x_max), n)
        base = _baseline()
        frame = pd.DataFrame({k: [v]*n for k, v in base.items()}); frame[xname] = grid
        try:
            y_line = np.asarray(model.predict(frame), dtype=float)
        except Exception:
            beta0 = float(getattr(model.params, "get", lambda k, d=0.0: d)("Intercept")) if hasattr(model, "params") else 0.0
            key = q(xname) if hasattr(model, "params") and q(xname) in getattr(model, "params", {}).index else xname
            beta = float(model.params[key]) if hasattr(model, "params") and key in getattr(model, "params", {}).index else 0.0
            contrib = 0.0
            if hasattr(model, "params"):
                for e in effects:
                    if e == xname: continue
                    k2 = q(e) if q(e) in model.params.index else e
                    if k2 in model.params.index and pd.api.types.is_numeric_dtype(df[e]):
                        contrib += float(model.params[k2]) * float(pd.to_numeric(df[e], errors="coerce").mean())
            y_line = beta0 + contrib + beta * grid
        return grid, y_line

    if regressor is None:
        num_effects = [e for e in effects if pd.api.types.is_numeric_dtype(df[e])]
        chosen_regs = [num_effects[0]] if num_effects else [effects[0]]
    elif isinstance(regressor, str):
        chosen_regs = [regressor]
    else:
        chosen_regs = list(regressor)

    for yi in ys:
        model = models[yi]
        y_true, y_hat, resid = _yhat_resid(yi, model)
        figs_reg = []

        for xr in chosen_regs:
            x = pd.to_numeric(df[xr], errors="coerce")
            m = x.notna() & y_true.notna()
            x_sc, y_sc = x[m], y_true[m]
            fig = plt.figure(); ax = plt.gca()
            if x_sc.size:
                # per-channel color for any tagged row in {xr, yi}
                sub_df = pd.DataFrame({xr: x_sc, yi: y_sc}, index=y_sc.index)
                mfl, cols = flier_color_vector(sub_df, [xr, yi])
                if mfl.any():
                    ax.scatter(x_sc[~mfl], y_sc[~mfl], s=25)
                    ax.scatter(x_sc[ mfl], y_sc[ mfl], s=25, c=cols)
                else:
                    ax.scatter(x_sc, y_sc, s=25)

                grid, y_line = _line(model, xr, float(x_sc.min()), float(x_sc.max()))
                ax.plot(grid, y_line, linewidth=1)
                ax.set_xlabel(xr); ax.set_ylabel(yi)
                ax.set_title(f"Regression Plot — {yi} vs {xr}\n(other predictors held at mean)")
            else:
                ax.axis("off"); ax.text(0.5,0.5,f"No data for {xr}",ha="center",va="center")
            plt.tight_layout(); plt.show()
            figs_reg.append(fig)

        # Actual by Predicted (tag by response channel color)
        fig_abp = plt.figure(); ax = plt.gca()
        sub_df = pd.DataFrame({yi: y_true}, index=y_true.index)
        mfl, cols = flier_color_vector(sub_df, [yi])
        if len(y_hat):
            if mfl.any():
                ax.scatter(y_hat.to_numpy()[~mfl], y_true.to_numpy()[~mfl], s=25)
                ax.scatter(y_hat.to_numpy()[ mfl], y_true.to_numpy()[ mfl], s=25, c=cols)
            else:
                ax.scatter(y_hat.to_numpy(), y_true.to_numpy(), s=25)
            mn = float(min(y_hat.min(), y_true.min()))
            mx = float(max(y_hat.max(), y_true.max()))
            ax.plot([mn, mx], [mn, mx], linewidth=1)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Actual by Predicted — {yi}")
        plt.tight_layout(); plt.show()

        # Residual by Predicted (tag by response channel color)
        fig_rbp = plt.figure(); ax = plt.gca()
        sub_df = pd.DataFrame({yi: y_true}, index=y_true.index)
        mfl, cols = flier_color_vector(sub_df, [yi])
        if len(y_hat):
            if mfl.any():
                ax.scatter(y_hat.to_numpy()[~mfl], resid.to_numpy()[~mfl], s=25)
                ax.scatter(y_hat.to_numpy()[ mfl], resid.to_numpy()[ mfl], s=25, c=cols)
            else:
                ax.scatter(y_hat.to_numpy(), resid.to_numpy(), s=25)
        ax.axhline(0.0, linewidth=1)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Residual"); ax.setTitle = f"Residual by Predicted — {yi}"
        plt.tight_layout(); plt.show()

        results[yi] = {
            "fig_regression": figs_reg,
            "fig_actual": fig_abp,
            "fig_residual": fig_rbp
        }

    return results
