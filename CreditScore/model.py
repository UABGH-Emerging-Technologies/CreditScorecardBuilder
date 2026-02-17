# CreditScore/model.py
"""
High-level modelling utilities used by the Streamlit UI.

Changes in this revision
------------------------
1. Uses the new plotter hierarchy:
       • FeatureImportancePlotter  (horizontal bar plot)
2. No more direct call to the old plot_feature_importance helper.
3. Figure objects returned by the plotter are displayed via Streamlit and
   stored in `self.feat_import_fig`, preserving the previous behaviour for
   ZIP downloads.
"""

from __future__ import annotations

import copy
from io import BytesIO
from typing import List, Tuple

import pandas as pd
import streamlit as st

from CreditScore.data import DataPipeline, DataValidator
from CreditScore.Plotters.feature_importance import FeatureImportancePlotter
from CreditScore.utils import CacheMixin


# --------------------------------------------------------------------------- #
#                         ──  M O D E L L I N G  ──                           #
# --------------------------------------------------------------------------- #
class ModellingComponent(CacheMixin):
    """
    Fit a (statsmodels) logistic regression, present coefficients and create
    an Excel calculator.

    Public attributes
    -----------------
    feat_import_fig : matplotlib.figure.Figure
        Set after calling `show_coefficients` – useful for ZIP download.
    """

    def __init__(self) -> None:
        self.excel_bytes = BytesIO(initial_bytes=b"")  # Excel calculator
        self.feat_import_fig = None  # will be set later

    # ------------------------------------------------------------------ #
    # utility                                                            #
    # ------------------------------------------------------------------ #
    def get_excel_calc(self) -> BytesIO:
        """Return the Excel calculator byte buffer."""
        return self.excel_bytes

    # ------------------------------------------------------------------ #
    # data preparation                                                   #
    # ------------------------------------------------------------------ #
    def data_prep(
        self,
        df: pd.DataFrame,
        target: str,
        features: List[str],
    ) -> Tuple[pd.DataFrame, pd.Series] | None:
        """
        Run the preprocessing pipeline + basic sanity checks.

        Returns
        -------
        X : pd.DataFrame
        y : pd.Series
        """
        pipe = DataPipeline()
        X, y = pipe.prepare_data(df, target, features)

        # ── VIF warning -------------------------------------------------
        vif_df = self.compute_vif(X)
        if not vif_df[vif_df.VIF > 4].empty:
            st.warning(
                "⚠️ High multicollinearity!\n\n" + vif_df[vif_df.VIF > 4].to_markdown(index=False)
            )

        # ── binary target check ----------------------------------------
        if not DataValidator().validate_binary_target(y):
            st.error("The selected outcome must be binary.")
            return None

        return X, y

    # ------------------------------------------------------------------ #
    # modelling                                                          #
    # ------------------------------------------------------------------ #
    def modelling(self, X, y, target):
        """
        Fit a logistic model (cached) and perform optional threshold
        optimisation.
        """
        with st.spinner("Training …"):
            mdl_cached, coef_df = self.fit_logit_model(X, y, target)

        model = copy.deepcopy(mdl_cached)

        # optional threshold optimisation
        if hasattr(model, "optimize_threshold"):
            from sklearn.model_selection import train_test_split

            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
            model.optimize_threshold(X_te, y_te, metric="precision")
            st.info(f"Optimised threshold: {model.optimal_threshold:.4f}")

        # warn when regularised
        if getattr(model, "used_regularization", False):
            st.warning(
                "⚠️ **Regularised Model** – p-values / CI not available "
                "because the estimates are biased."
            )

        return model, coef_df

    # ------------------------------------------------------------------ #
    # visualisation + Excel export                                       #
    # ------------------------------------------------------------------ #
    def show_coefficients(self, X, data_features, model, coef_df):
        """
        Display coefficients, feature-importance plot and diagnostic stats.
        """
        # ── table -------------------------------------------------------
        st.subheader("Coefficients")
        st.dataframe(coef_df[["Feature", "Coefficient"]].round(4))

        # ── feature-importance plot ------------------------------------
        plot_df = (
            coef_df.loc[coef_df["Feature"] != "Intercept", ["Feature", "Coefficient"]]
            .assign(abs_coef=lambda d: d["Coefficient"].abs())
            .sort_values("abs_coef", ascending=False)
        )
        features = plot_df["Feature"].to_list()
        magnitudes = plot_df["abs_coef"].to_numpy()

        fi_plotter = FeatureImportancePlotter()
        self.feat_import_fig = fi_plotter.plot_importance(features, magnitudes)
        st.pyplot(self.feat_import_fig)  # display in Streamlit

        # ── detailed summary -------------------------------------------
        if not getattr(model, "used_regularization", False):
            fmt = {c: "{:.4f}" for c in coef_df.columns if coef_df[c].dtype != "O"}
            st.write("#### Detailed Model Summary")
            st.dataframe(coef_df.style.format(fmt))
            st.download_button(
                "📥 Download Detailed Model Summary (CSV)",
                self.to_csv_bytes(coef_df),
                "detailed_model_summary.csv",
                "text/csv",
            )

        # ── model statistics -------------------------------------------
        stats = model.get_model_stats()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AIC", f"{stats['aic']:.2f}")
        col2.metric("BIC", f"{stats['bic']:.2f}")
        col3.metric("Pseudo-R²", f"{stats['pseudo_r2']:.3f}")
        converged = stats.get("converged", True)
        col4.metric("Converged", "✓" if converged else "✗")

        # ── Excel calculator -------------------------------------------
        st.write("---")
        if st.button("📊 Create Excel Calculator", key="excel_export"):
            self._build_excel_calculator(X, data_features, coef_df)

        # ── convergence diagnostics ------------------------------------
        if not converged:
            with st.expander("⚠️ Convergence Diagnostics", expanded=True):
                self._show_convergence_diagnostics(model)

    # ------------------------------------------------------------------ #
    # helpers – Excel & diagnostics                                      #
    # ------------------------------------------------------------------ #
    def _build_excel_calculator(self, X, features, coef_df):
        """
        Generate the downloadable Excel prediction sheet.
        """
        try:
            from CreditScore.excel_export import create_logistic_excel
        except ImportError:
            st.error("Excel export helper not found.")
            return

        # prepare coefficient dictionary
        coef_dict = {
            ("const" if f == "Intercept" else f): c
            for f, c in zip(coef_df["Feature"], coef_df["Coefficient"])
        }

        # gather feature statistics for scaling hints inside Excel
        feature_stats = {
            f: {"mean": float(X[f].mean()), "std": float(X[f].std())}
            for f in features
            if X[f].dtype.kind in {"i", "f"}
        }

        # build Excel bytes
        self.excel_bytes = create_logistic_excel(
            coefficients=coef_dict,
            feature_names=["const"] + features,
            feature_stats=feature_stats,
        )
        st.session_state["excel_calc"] = self.excel_bytes

        # download button
        st.download_button(
            "📥 Download Logistic Regression Excel",
            data=self.excel_bytes,
            file_name="logistic_regression_calculator.xlsx",
            mime=("application/vnd.openxmlformats-officedocument." "spreadsheetml.sheet"),
            help="Excel file with built-in formulas for new predictions.",
        )
        st.info(
            "Excel file generated!  Blue cells = input, remaining cells are "
            "protected (password: 'logistic')."
        )

    def _show_convergence_diagnostics(self, model):
        """Display Streamlit guidance when optimization fails to converge."""
        diagnostics = model.diagnose_convergence()
        st.warning("The model did not converge properly. Estimates might be unreliable.")
        if diagnostics["condition_number"]:
            st.write(f"**Condition Number:** {diagnostics['condition_number']:.2e}")
            if diagnostics["condition_number"] > 1000:
                st.info("High condition number suggests multicollinearity.")
        if diagnostics["warnings"]:
            st.write("**Warnings:**")
            for w in diagnostics["warnings"]:
                st.write(f"- {w}")
        st.info(
            "**Possible solutions:**\n"
            "- Remove highly correlated features\n"
            "- Combine similar features\n"
            "- Use regularisation\n"
            "- Collect more data\n"
            "- Check for perfect separation"
        )
