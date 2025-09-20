# streamlit run app.py --server.port 8501 --server.address 0.0.0.0
from __future__ import annotations
import io
import json
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

# ---- Import your toolkit (place the jmpkit/ folder next to this file) ----
from jmpkit import (
    Dataset, load_any, add_channel_expr, add_channel_func,
    jmp_distribution_report, jmp_multivariate_panel_full,
    fit_model, jmp_fit_and_plots, plot_xy_by_group
)

# ------------------ Small helpers for Streamlit ------------------

def _load_uploaded(file) -> Dataset:
    """Load uploaded file (Excel/CSV/Parquet/Feather/JSON/PKL) into Dataset."""
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file, sheet_name=0, header=0)
    elif name.endswith((".csv", ".tsv", ".txt")):
        df = pd.read_csv(file, sep=None, engine="python", header=0)
    elif name.endswith(".parquet"):
        df = pd.read_parquet(file)
    elif name.endswith(".feather"):
        df = pd.read_feather(file)
    elif name.endswith(".json"):
        try:
            df = pd.read_json(file, orient="records")
        except ValueError:
            file.seek(0)
            df = pd.read_json(file, orient="table")
    elif name.endswith(".pkl"):
        df = pd.read_pickle(file)
    else:
        st.error(f"Unsupported file type: {name}")
        st.stop()
    return Dataset(df.convert_dtypes(), source=name)

def _df() -> pd.DataFrame:
    return st.session_state["data"].df

def _download_link(df: pd.DataFrame, *, to: str = "xlsx"):
    if to == "xlsx":
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="data")
        bio.seek(0)
        st.download_button(
            "Download Excel (.xlsx)", data=bio, file_name="dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="dataset.csv", mime="text/csv")

def _safe_num_cols(df: pd.DataFrame) -> List[str]:
    return list(df.select_dtypes(include=[np.number]).columns)

def _show_fig(fig, *, width_px: int, dpi: int, caption: str | None = None):
    """
    Display a Matplotlib figure at a user-controlled pixel width and DPI
    without modifying the figure object (no jmpkit edits).
    """
    buf = io.BytesIO()
    # Save at requested DPI; st.image will scale to width_px, preserving aspect ratio.
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, caption=caption, width=width_px)
    buf.close()

# ------------------ App layout ------------------

st.set_page_config(page_title="JMP-like Analysis — Web UI", layout="wide")
st.title("JMP-like Analysis — Web Interface")

with st.sidebar:
    st.header("1) Upload Data")
    up = st.file_uploader(
        "Excel/CSV/Parquet/Feather/JSON/PKL",
        type=["xlsx","xls","csv","tsv","txt","parquet","feather","json","pkl"]
    )
    if up is not None:
        st.session_state["data"] = _load_uploaded(up)
        st.success(f"Loaded: {up.name}  →  {len(_df()):,} rows × {_df().shape[1]} cols")

    if "data" in st.session_state:
        st.divider()
        st.header("2) Channels")
        st.caption("Add derived columns using expressions. Use backticks for names with spaces: `temp(f)`")
        col1, col2 = st.columns([1,2])
        with col1:
            new_name = st.text_input("New column name", key="channel_name")
        with col2:
            expr = st.text_input("Expression (pandas.eval)", placeholder="e.g., log(`credit_score` + 1)", key="channel_expr")
        if st.button("Add Channel", type="primary", disabled=not new_name or not expr):
            try:
                add_channel_expr(st.session_state["data"], new_name, expr, overwrite=True)
                st.success(f"Added/updated channel: {new_name}")
            except Exception as e:
                st.error(f"Failed to add channel: {e}")

        st.divider()
        st.header("3) Display Size (no changes to jmpkit)")
        width_px = st.slider("Plot width (pixels)", 300, 2000, 900, 50)
        dpi      = st.slider("Render DPI", 72, 300, 120, 4)
        st.caption("These only affect how figures are rendered here; the underlying figure objects are unchanged.")

        st.divider()
        st.header("4) Download")
        _download_link(_df(), to="xlsx")
        _download_link(_df(), to="csv")

# Halt early if no data
if "data" not in st.session_state:
    st.info("Upload a file to get started. Once loaded, you can add channels and run analyses.")
    st.stop()

# ------------------ Main tabs ------------------

tab_data, tab_dist, tab_multi, tab_fit, tab_plot_scatter = st.tabs(["Data", "Distribution", "Multivariate", "Fit Model", "Fit Y by X (faceted)"])

# ---- Data tab ----
with tab_data:
    st.subheader("Preview")
    st.dataframe(_df().head(100), use_container_width=True)
    st.subheader("Summary")
    st.dataframe(st.session_state["data"].summary(), use_container_width=True)

# ---- Distribution tab ----
with tab_dist:
    st.subheader("Distribution Report")
    cols = st.multiselect("Columns (numeric suggested)", options=list(_df().columns), default=_safe_num_cols(_df())[:1])
    if st.button("Run Distribution", disabled=len(cols) == 0):
        for c in cols:
            st.markdown(f"#### {c}")
            try:
                # Call jmpkit as-is; only display size is controlled here
                res = jmp_distribution_report(_df(), c)
                _show_fig(res["fig_hist"], width_px=width_px, dpi=dpi, caption="Histogram + Fitted Normal")
                _show_fig(res["fig_box"],  width_px=width_px, dpi=dpi, caption="Box Plot")
                st.dataframe(res["quantiles_df"], use_container_width=True)
                st.dataframe(res["summary_df"], use_container_width=True)
                st.dataframe(res["fit_params_df"], use_container_width=True)
                st.dataframe(res["fit_stats_df"], use_container_width=True)
                st.dataframe(res["gof_summary_df"], use_container_width=True)
                st.dataframe(res["ad_details_df"], use_container_width=True)
            except Exception as e:
                st.warning(f"Skipping {c}: {e}")

# ---- Multivariate tab ----
with tab_multi:
    st.subheader("Multivariate Panel")
    mcols = st.multiselect("Columns (≥2)", options=list(_df().columns), default=_safe_num_cols(_df())[:3])
    alpha = st.number_input("Alpha (tail prob for UCL)", min_value=0.0001, max_value=0.2, value=0.025, step=0.001, format="%.3f")
    upper = st.selectbox("Upper triangle", options=["scatter", "corr"], index=0)
    diag = st.selectbox("Diagonal", options=["label", "none"], index=0)
    if st.button("Run Multivariate", disabled=len(mcols) < 2):
        try:
            out = jmp_multivariate_panel_full(_df(), columns=mcols, alpha=alpha, upper=upper, diag=diag)
            _show_fig(out["fig_matrix"], width_px=width_px, dpi=dpi, caption="Scatterplot Matrix")
            st.dataframe(out["corr_df"], use_container_width=True)
            st.dataframe(out["n_pairs_df"], use_container_width=True)
            _show_fig(out["md_fig"], width_px=width_px, dpi=dpi, caption="Mahalanobis Distances")
            st.dataframe(out["md_table"], use_container_width=True)
            st.caption(f"UCL = {out['UCL']:.3f}")
        except Exception as e:
            st.error(f"Multivariate failed: {e}")

# ---- Fit Model tab ----
with tab_fit:
    st.subheader("Fit Model")
    persona = st.selectbox(
        "Personality",
        [
            "standard_least_squares","stepwise","glm","generalized_regression",
            "mixed_model","manova","nominal_logistic","ordinal_logistic",
            "proportional_hazards","parametric_survival","partial_least_squares","response_screening"
        ],
        index=0
    )
    # Common selectors
    y_all = list(_df().columns)
    eff_all = list(_df().columns)
    multi_y = persona in ("manova","partial_least_squares","response_screening")

    y_sel = st.multiselect("Y variables" if multi_y else "Y variable",
                           options=y_all, default=(y_all[:2] if multi_y else y_all[:1]))
    if not multi_y and y_sel:
        y_sel = y_sel[0]

    effects = st.multiselect("Construct Model Effects", options=eff_all, default=eff_all[:2])

    # Construct options
    colA, colB, colC = st.columns(3)
    with colA:
        degree = st.number_input("Polynomial degree (per effect)", 1, 5, 1, 1)
    with colB:
        cross = st.checkbox("Include pairwise interactions", value=False)
    with colC:
        add_poly = st.checkbox("Add polynomial terms", value=False)

    # Persona-specific options
    extra_kwargs = {}

    if persona == "stepwise":
        col1, col2 = st.columns(2)
        with col1:
            extra_kwargs["stepwise_direction"] = st.selectbox("Direction", ["forward","backward","both"], index=2)
        with col2:
            extra_kwargs["max_steps"] = st.number_input("Max steps", 1, 200, 50, 1)

    if persona == "glm":
        col1, col2 = st.columns(2)
        with col1:
            family = st.selectbox("Family", ["gaussian","binomial","poisson","gamma","inverse_gaussian","nb"], index=0)
            extra_kwargs["family"] = family
        with col2:
            link = st.selectbox("Link (optional)", ["", "identity","log","logit","probit","cloglog"], index=0)
            extra_kwargs["link"] = (None if link == "" else link)

    if persona == "mixed_model":
        if len(eff_all) > 0:
            extra_kwargs["groups"] = st.selectbox("Grouping column (random intercept)", eff_all)

    if persona in ("proportional_hazards","parametric_survival"):
        dur = st.selectbox("Duration column", options=eff_all)
        evt = st.selectbox("Event column (1=event)", options=eff_all)
        extra_kwargs["duration_col"] = dur
        extra_kwargs["event_col"] = evt
        if persona == "parametric_survival":
            extra_kwargs["aft_distribution"] = st.selectbox("AFT distribution", ["weibull","lognormal","loglogistic","exponential"], index=0)

    if persona == "partial_least_squares":
        extra_kwargs["n_components"] = st.number_input("PLS components", 1, 10, 2, 1)

    # Run fit
    if st.button("Fit"):
        try:
            result = fit_model(
                _df(), y=y_sel, effects=effects,
                personality=persona, degree=degree, cross=cross, add_poly=add_poly,
                **extra_kwargs
            )
            st.success(f"Fitted: {result.personality}")
            st.code(result.formula or "(no formula string)", language="text")  # monospace formula

            # --- REPLACE your current "Print summaries" loop with this ---
            for k, v in result.summaries.items():
                st.markdown(f"#### Summary — {k}")

                # If it's our JMP-style dict, render each section as a table
                if isinstance(v, dict):
                    for section, content in v.items():
                        st.markdown(f"**{section}**")

                        if isinstance(content, list) and content and isinstance(content[0], dict):
                            df = pd.DataFrame(content)
                            # nice index if 'Term' exists
                            if "Term" in df.columns:
                                df = df.set_index("Term")
                            st.dataframe(df, use_container_width=True)

                        elif isinstance(content, dict):
                            # dict-of-dicts -> rows; dict-of-scalars -> single row
                            first_val = next(iter(content.values()), {})
                            if isinstance(first_val, dict):
                                df = pd.DataFrame(content).T
                            else:
                                df = pd.DataFrame([content])
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.code(str(content), language="text")   # or st.write(content)
                    continue

                # Fallback: statsmodels Summary objects etc.
                try:
                    html = v.as_html()
                    st.code(html, height=520, scrolling=True)
                except Exception:
                    st.code(str(v), language="text")


            # Show metrics
            if result.metrics:
                st.markdown("#### Metrics")
                try:
                    pretty = {k: {m: (float(vv) if hasattr(vv, "__float__") else vv)
                                  for m, vv in d.items()}
                              for k, d in result.metrics.items()}
                    st.code(json.dumps(pretty, indent=2), language="json")
                except Exception:
                    st.code(str(result.metrics), language="text")

            # Diagnostic plots (only single-Y regression-like)
            if persona in ("standard_least_squares","glm","generalized_regression","stepwise") and (not multi_y):
                st.markdown("#### Diagnostic Plots")
                regressor = effects[0] if effects else None
                figs = jmp_fit_and_plots(
                    _df(),
                    y=y_sel if isinstance(y_sel, str) else y_sel[0],
                    effects=effects, personality=persona,
                    regressor=regressor
                )
                one = next(iter(figs.values()))
                for fig in one["fig_regression"]:
                    _show_fig(fig, width_px=width_px, dpi=dpi, caption="Regression Plot")
                _show_fig(one["fig_actual"], width_px=width_px, dpi=dpi, caption="Actual by Predicted")
                _show_fig(one["fig_residual"], width_px=width_px, dpi=dpi, caption="Residual by Predicted")

        except Exception as e:
            st.error(f"Fit failed: {e}")

# ---- Plot Scatter tab ----
with tab_plot_scatter:
    st.subheader("Fit Y by X (faceted)")

    cols_all = list(_df().columns)
    x = st.selectbox("X (predictor)", options=cols_all, index=0 if cols_all else None)
    y_opts = [c for c in cols_all if c != x]
    y = st.selectbox("Y (response)", options=y_opts, index=0 if y_opts else None)

    group = st.selectbox("Group (optional)", options=["(none)"] + cols_all, index=0)
    grp = None if group == "(none)" else group

    # Optional layout knobs (keep minimal; you can remove these lines if not needed)
    ncols = st.slider("Facets per row", 1, 4, 3)
    h = st.slider("Facet height (inches)", 3.0, 6.0, 4.0)
    w = st.slider("Facet width (inches)", 3.0, 6.0, 4.0)

    run_disabled = (x is None) or (y is None)
    if st.button("Run XY Plot", disabled=run_disabled):
        fig, _ = plot_xy_by_group(_df(), x=x, y=y, group=grp, ncols=ncols, height=h, width=w)

        # If you already use _show_fig elsewhere:
        # _show_fig(fig, width_px=width_px, dpi=dpi, caption=f"{y} vs {x} by {group if grp else 'All'}")

        # Or simply:
        st.pyplot(fig, use_container_width=True)