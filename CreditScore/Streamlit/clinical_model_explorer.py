"""
Logistic-Regression + Power-Analysis Streamlit page
"""

from __future__ import annotations

from typing import List

import pandas as pd

from config.config import Config
from CreditScore.analysis import PowerAnalysisComponent
from CreditScore.model import ModellingComponent
from CreditScore.Streamlit.streamlit_interface import BaseHandler
from CreditScore.utils import ensure_dataframe
from llm_utils.aiweb_common.streamlit.page_renderer import StreamlitUIHelper


class ClinicalModelExplorer(BaseHandler):
    """Logistic-regression explorer (coefficients + power analysis)."""

    # ------------------------------------------------------------------ init
    def __init__(self, ui: StreamlitUIHelper) -> None:
        super().__init__(ui)
        self.page_title = "Logistic Regression + Power Analysis"
        self.page_icon = "🧪"
        self.file_types = ("csv", "xlsx")
        self.default_features = Config.DEFAULT_FEATURES
        self.plot_options = {"feat_fig": "Feature Importance"}
        self.table_options = {
            "coef_df": "Model Coefficients",
            "power_analysis": "Power Analysis Results",
        }

    # ------------------------------------------------------------------ render
    def render(self) -> None:
        """Render the Streamlit page and run analysis when submitted."""
        self._init_page()
        self._render_sidebar()

        df = self._load_data()
        if df.empty:
            return

        target, feats = self._select_columns(df)
        if not (target and feats):
            return

        ss, ns = self.ui.session_state, self._key

        if self.ui.button("Submit model", key=ns("submit")):
            ss[ns("submitted")] = True

        if not ss.get(ns("submitted"), False):
            self.ui.info("Press Submit model to run analysis.")
            return

        # once submitted, run full analysis & build report immediately
        artifacts = self._run_analysis(df, target, feats)
        self._build_report(**artifacts)

        # allow re-running with a *Reset* button
        if self.ui.button("Reset page", key=ns("reset_btn")):
            for k in list(ss.keys()):
                if k.startswith(ns("")):
                    ss.pop(k)
            self.ui.rerun()

    # ==================================================================
    # internal analysis
    # ==================================================================
    def _run_analysis(self, data: pd.DataFrame, target: str, features: List[str]):
        """Run model training, diagnostics, and report artifact assembly."""
        modeller = ModellingComponent()
        X, y = modeller.data_prep(data, target, features)

        self.ui.write("### Data preview")
        self.ui.dataframe(pd.concat([X, y.rename(target)], axis=1))

        # ----------------------- data quality
        from CreditScore.data import DataValidator

        issues = DataValidator().check_data_quality(data, target, features)
        if issues["missing_values"] or issues["constant_features"]:
            with self.ui.expander("Data Quality Report"):
                self.ui.text(DataValidator.report_data_issues(issues))

        # ----------------------- modelling
        model, coef_df = modeller.modelling(X, y, target)
        modeller.show_coefficients(X, features, model, coef_df)

        ss = self.ui.session_state
        ss[self._key("feat_fig")] = modeller.feat_import_fig
        ss[self._key("coef_df")] = coef_df

        # ----------------------- power analysis
        tab_lr, tab_power = self.ui.tabs(["Logistic Regression", "Power Analysis"])
        with tab_power:
            PowerAnalysisComponent().render(X, y, model, features)

        # ----------------------- prepare report parts
        tables = {
            "Coefficients.csv": ensure_dataframe(coef_df),
        }

        power_key = "power_result"
        if power_key in ss and ss[power_key].get("csv_df") is not None:
            tables["Power_Analysis.csv"] = ensure_dataframe(ss[power_key]["csv_df"])

        figures = {}
        if self._key("feat_fig") in ss:
            figures["Feature_Importance"] = ss[self._key("feat_fig")]

        summary_md = ""
        if power_key in ss and "md_table" in ss[power_key]:
            summary_md = ss[power_key]["md_table"]

        return dict(
            tables=tables,
            figures=figures,
            summary=summary_md,
            zip_name="lrpa_analysis_report.zip",
            extra_files=None,
        )
