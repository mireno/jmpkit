from jmpkit import (
    load_any, add_channel_func,
    jmp_distribution_all, jmp_multivariate_panel_full,
    fit_ols, fit_glm, jmp_fit_and_plots
)

DATA_PATH = "/Users/mireno/Downloads/DataScience/mortgage.xlsx"

def main():
    data = load_any(DATA_PATH)

    # Derived channel example
    if "loan_houseprice_ratio" in data.df.columns:
        add_channel_func(data, "ratio100", lambda d: d["loan_houseprice_ratio"] * 100)

    # Distribution for all numeric
    _ = jmp_distribution_all(data.df)

    # Multivariate
    cols = [c for c in ["credit_score","loan_houseprice_ratio","ratio100","income_class"] if c in data.df.columns]
    if len(cols) >= 2:
        _ = jmp_multivariate_panel_full(data.df, columns=cols, alpha=0.025)

    # OLS example
    y = "credit_score"
    effects = [c for c in ["loan_houseprice_ratio","CA","income_class"] if c in data.df.columns]
    if y in data.df.columns and effects:
        res = fit_ols(data.df, y=y, effects=effects)
        print(res.summaries[y])
        _ = jmp_fit_and_plots(data.df, y=y, effects=effects, regressor=effects[0])

    # GLM example (logistic)
    if "default" in data.df.columns and "credit_score" in data.df.columns:
        res_glm = fit_glm(data.df, y="default", effects=["credit_score"], family="binomial", link="logit")
        print(res_glm.summaries["default"])

if __name__ == "__main__":
    main()
