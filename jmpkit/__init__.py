from .io import Dataset, load_any
from .channels import add_channel_expr, add_channel_func
from .distribution import jmp_distribution_report, jmp_distribution_all
from .multivariate import jmp_scatterplot_matrix, jmp_multivariate_panel_full
from .fitmodel import (
    fit_model, FitResult,
    fit_ols, fit_stepwise, fit_glm, fit_generalized_regression, fit_mixed_model,
    fit_manova, fit_nominal_logistic, fit_ordinal_logistic,
    fit_proportional_hazards, fit_parametric_survival, fit_pls, fit_response_screening
)
from .plots import jmp_fit_and_plots
from .utils import display_dataframe_to_user
from .jmp_faceted_scatter import plot_xy_by_group

# NEW: expose tagging + colors to the app
from .utils import tag_fliers, get_fliers, set_flier_color, get_flier_color
