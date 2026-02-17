# components/power_analysis.py
import pandas as pd
import streamlit as st

from CreditScore.evaluate import StatsmodelsEvaluator
from CreditScore.utils import CacheMixin


class PowerAnalysisComponent(CacheMixin):
    """Render a Streamlit power analysis workflow for fitted models."""

    def render(self, X, y, model, features):
        """Display power analysis controls and exportable results."""
        if model is None:
            return

        if getattr(model, "used_regularization", False):
            st.info(
                "🔋 Power analysis not available for regularised models "
                "because standard errors are not defined."
            )
            return

        st.write("🔋  Power analysis")

        evaluator = StatsmodelsEvaluator(model)

        feature_list = ["Intercept"] + features
        coeff = st.selectbox("Coefficient to test", feature_list, index=None, key="coeff")

        if not coeff:  # Nothing selected yet
            return

        tails = st.selectbox("Tail(s)", ["One", "Two"], key="tails")
        alpha = st.slider("α", 0.01, 0.20, 0.05, 0.01, key="alpha")
        power_target = st.slider("Target power", 0.50, 0.99, 0.80, 0.01, key="power_target")

        compute_clicked = st.button("Compute", key="compute")

        if compute_clicked:
            with st.spinner("Processing..."):
                # Wald-test based sample-size
                needed_n = evaluator.power_analysis_wald(
                    coeff, alpha=alpha, power=power_target, tails=tails
                )

                # Parameters for G*Power
                gpower = evaluator.get_gpower_parameters(coeff, X, y)

                x_col = X[coeff] if coeff != "Intercept" else None
                x_distribution = (
                    "Binomial" if x_col is not None and set(x_col.unique()) <= {0, 1} else "Normal"
                )

                md_table = f"""
                **Module:** Z tests → *Logistic regression* &nbsp;&nbsp;
                **Analysis type:** *A priori*

                | Parameter | Value |
                |-----------|-------|
                | Tail(s) | {tails} |
                | Odds ratio | {gpower['odds_ratio']:.3f} |
                | Pr(Y=1\\|X=1) H₀ | {gpower['pr_y1_x1_h0']:.3f} |
                | α err prob | {alpha:.2f} |
                | Power (1-β err prob) | {power_target:.2f} |
                | R² other X | {gpower['r2_other_x']:.3f} |
                | X distribution | {x_distribution} |
                | X parm π | {gpower['x_param_1']:.3f} |
                | Number of tested predictors | 1 |
                | Total number of predictors | {gpower['n_total_predictors']} |
                """

                # CSV dataframe (single row for now)
                csv_row = {
                    "Coefficient": coeff,
                    "tails": tails,
                    "alpha": alpha,
                    "target_power": power_target,
                    "required_n": needed_n,
                    "odds_ratio": gpower["odds_ratio"],
                    "pr_y1_x1_h0": gpower["pr_y1_x1_h0"],
                    "r2_other_x": gpower["r2_other_x"],
                    "x_distribution": x_distribution,
                    "x_param_1": gpower["x_param_1"],
                    "x_param_2": gpower.get("x_param_2", pd.NA),
                    "n_tested_predictors": 1,
                    "n_total_predictors": gpower["n_total_predictors"],
                }
                csv_df = pd.DataFrame([csv_row])

                st.session_state.power_result = {
                    "needed_n": needed_n,
                    "md_table": md_table,
                    "csv_df": csv_df,
                }

        if "power_result" in st.session_state:
            res = st.session_state.power_result

            st.success(
                f"≈ **{res['needed_n']:,}** observations required " f"for {power_target:.0%} power"
            )

            st.write("#### Parameters for G*Power Verification")
            st.markdown(res["md_table"])

            st.info(
                "_Note: Some difference between our calculation and G*Power "
                "is expected. G*Power uses a single-predictor approximation "
                "while we use the full Fisher information matrix from your "
                "multi-predictor model._"
            )

            st.download_button(
                "Download result (CSV)",
                self.to_csv_bytes(res["csv_df"]),
                "power_analysis.csv",
            )
