"""
Clinical Risk Scorecard Streamlit page
"""

from __future__ import annotations

import re
import time
from io import BytesIO
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.config import Config
from CreditScore.evaluate import ScorecardEvaluator
from CreditScore.excel_export import create_scorecard_excel
from CreditScore.Plotters.performance import PerformancePlotter

from CreditScore.Plotters.score_distribution import ScoreDistributionPlotter
from CreditScore.Plotters.shap import SHAPPlotter
from CreditScore.Plotters.threshold import ThresholdMetricPlotter
from CreditScore.Streamlit.streamlit_interface import (
    BaseHandler,
    ensure_dataframe,
)
from CreditScore.train import OptbinningScorecardModel
from CreditScore.utils import (
    LogisticWrapper,
    ModelMetrics,
    generate_model_summary,
)
from aiweb_common.streamlit.page_renderer import StreamlitUIHelper


# ────────────────────────────────────────────────────────────────────────────
class ScorecardBuilder(BaseHandler):
    """🩺 Clinical risk-scorecard builder."""

    # ------------------------------------------------------------------ init
    def __init__(self, ui: StreamlitUIHelper) -> None:
        super().__init__(ui)
        self.page_title = "Clinical Risk Scorecard Builder"
        self.page_icon = "🤖"
        self.file_types = ("csv", "xlsx")
        self.default_features = [c for c in Config.DEFAULT_FEATURES if c != "thirty_day_mortality"]
        self.plot_options = {
            "roc": "ROC Curve",
            "prc": "Precision-Recall Curve",
            "cap": "Cumulative Accuracy Profile",
            "ks": "KS Statistic",
            "threshold": "Threshold Plot",
            "shap": "SHAP Decision Plots",
            "psi": "Population Stability Index",
            "distribution": "Score Distribution",
        }
        self.table_options = {
            "scorecard": "Scorecard Table",
            "psi_table": "PSI Table",
            "metrics_comparison_df": "Detailed Metrics",
            # "train_metrics": "Train Metrics",
            # "test_metrics": "Test Metrics",
        }

    # ------------------------------------------------------------------ render
    def render(self) -> None:  # noqa: C901  (UI-heavy)
        """Render the Streamlit scorecard builder UI and actions."""
        self._init_page()
        self._render_sidebar()

        df = self._load_data()
        if df.empty:
            return

        ss, ns = self.ui.session_state, self._key
        if ss.get(ns("show_data_preview"), False):
            with self.ui.expander("🗂️ See data preview", expanded=False):
                self.ui.dataframe(df)

        target, feats = self._select_columns(df)
        feature_filter = self.ui.session_state[self._key("feature_category")]
        if not (target and feats):
            return

        ss.setdefault(ns("trained"), False)
        ss.setdefault(ns("training"), False)

        # Train / re-train button
        if self.ui.button(
            "🚈 Train / Re-train model", use_container_width=True, key=ns("train_btn")
        ):
            ss[ns("training")] = True
            ss[ns("trained")] = False
            self.ui.rerun()

        # -------------------------------------------------------- training
        if ss.get(ns("training")):
            self._train_pipeline(df, target, feats, feature_filter=feature_filter)
            ss[ns("training")] = False
            ss[ns("trained")] = True
            self.ui.rerun()

        # -------------------------------------------------------- results
        if ss.get(ns("trained")):
            self._show_results()

            if self.ui.button("🏗️ Build full report ZIP", key=ns("zip_btn")):
                self._compile_and_download_report()

    # ==================================================================
    # internal helpers
    # ==================================================================
    def _train_pipeline(
        self, data: pd.DataFrame, target: str, feats: List[str], feature_filter: str
    ) -> None:
        """
        Full score-card training pipeline.
        Stores artefacts in session_state (namespaced).
        """
        from CreditScore.data import DataPreprocessor, DataSplitter
        from CreditScore.variable_selection import ScorecardVariableSelector

        ss, ns = self.ui.session_state, self._key

        pre = DataPreprocessor()
        X, y = pre.prepare_modeling_data(data, target, feats)
        y = pre.encode_target(y)
        X_tr, X_te, y_tr, y_te = DataSplitter().split_data(X, y, test_size=0.2, random_state=42)

        sel = ScorecardVariableSelector()
        with self.ui.spinner("Training model…", show_time=True, width="stretch"):
            start_time = time.time()
            sc_model, sel_vars, sel_rep = sel.select_variables_full_pipeline(
                X_train=X_tr,
                y_train=y_tr,
                scorecard_model_class=OptbinningScorecardModel,
                initial_binning_config=Config.binning_args(),
                scorecard_config=Config.scorecard_args(),
                fit_config=Config.get_scorecard_fit_config(),
            )
            stop_time = time.time()
            time_elapsed = stop_time - start_time
            sc_model.optimize_threshold(X_te[sel_vars], y_te)

            threshold_strategy = ss.get(ns("threshold_choice"), "Optimized")

            if threshold_strategy == "Event rate":
                event_rate = y_tr.mean()
                sc_model.optimal_threshold = event_rate
                sc_model.threshold_strategy_label = "Event rate"
                ss[ns("event_rate_threshold")] = event_rate

            elif threshold_strategy == "Default (0.5)":
                sc_model.optimal_threshold = 0.5
                sc_model.threshold_strategy_label = "Default (0.5)"

            else:
                # Assuming optimized threshold was already set via optimize_threshold
                sc_model.threshold_strategy_label = "Optimized"

            # Save the final threshold in use (for UI/display/export)
            ss[ns("active_threshold")] = sc_model.optimal_threshold
            threshold_summary = f"Trained with threshold strategy: {threshold_strategy} → threshold = {sc_model.optimal_threshold:.4f}"
            ss[ns("threshold_strategy_summary")] = threshold_summary
            self.ui.success(f"Model trained in {round(time_elapsed/60, 2)} minutes")
            selection_text_report = sel.selector.generate_selection_report()
            ss[ns("selection_text_report")] = selection_text_report

        # ---------- metrics & plots
        tr_pred = sc_model.predict_proba(X_tr)[:, 1]
        te_pred = sc_model.predict_proba(X_te)[:, 1]
        met_tr = ModelMetrics.calculate_metrics(
            y_tr, tr_pred, y_pred_class=sc_model.predict_class(X_tr)
        )
        met_te = ModelMetrics.calculate_metrics(
            y_te, te_pred, y_pred_class=sc_model.predict_class(X_te)
        )

        use_baseline = ss.get(ns("use_baseline_rate"), False)
        perf_plotter = PerformancePlotter()
        figs_perf = perf_plotter.from_evaluator(
            ScorecardEvaluator(sc_model),
            X_train=X_tr[sel_vars],
            y_train=y_tr,
            X_test=X_te[sel_vars],
            y_test=y_te,
            use_baseline_rate=use_baseline,
        )
        thr_plotter = ThresholdMetricPlotter()
        fig_thr = thr_plotter.plot(
            y_true=y_te,
            y_proba=sc_model.predict_proba(X_te[sel_vars])[:, 1],
            optimal_threshold=sc_model.optimal_threshold,
        )
        # use_baseline = ss.get(ns("use_baseline_rate"), False)
        plotter = SHAPPlotter(sc_model, LogisticWrapper)
        figs_shap = plotter.decision_plots(
            X_tr[sel_vars],
            X_te[sel_vars],
            y_tr,
            use_baseline_rate=use_baseline,
        )
        plotter2 = ScoreDistributionPlotter()
        plotter2.model = sc_model
        fig_score = plotter2.plot_score_distribution(
            X_te[sel_vars],
            y_te,
        )
        evaluator = ScorecardEvaluator(sc_model)
        evaluator.fit_monitoring(
            X_te[sel_vars],
            y_te,
            X_tr[sel_vars],
            y_tr,
        )
        fig_psi = evaluator.plot_psi()
        psi_table = evaluator.get_psi_report()
        metrics_comparison_df = ModelMetrics.create_metrics_comparison_df(
            {
                "Train": met_tr,
                "Test": met_te,
            }
        )

        # ---------- Excel helper files
        scorecard_df = sc_model.get_scorecard_table()

        # # # Masking for Missing/Special columns. No longer in use.
        # # # if "Points" in scorecard_df.columns:
        # # #     if pd.api.types.is_numeric_dtype(scorecard_df["Points"]) and scorecard_df["Points"].notna().any():
        # # #         missing_mask = scorecard_df["Bin"].isna() | scorecard_df["Bin"].astype(str).str.lower().isin(["missing", "special"])
        # # #         scorecard_df.loc[missing_mask, "Points"] = "No Value"
        # # #         self.ui.info("Missing and special bins set to Points = 'No Value' for clinical interpretability.")
        # # #     else:
        # # #         self.ui.warning("Points column is not numeric or is empty. Skipping override.")
        # # # else:
        # # #     self.ui.warning("Points column not found in scorecard table. Override skipped.")

        # Masking for ASA bin label  ([np.int64(3)]) -> [3], not currently working right.
        def clean_asa_bin_labels(bin_value):
            if isinstance(bin_value, list):
                return ", ".join(str(int(x)) for x in bin_value)
            elif isinstance(bin_value, np.integer):
                return str(int(bin_value))
            elif isinstance(bin_value, str):
                if "≤" in bin_value or "<" in bin_value:
                    return "1 or 2"
                return re.sub(r"np\.int64\((\d+)\)", r"\1", bin_value)
            else:
                return str(bin_value)

        # --- after creating Bin_Display ---
        asa_mask = scorecard_df["Variable"] == "AsaStatusClassification_Value_Code"
        scorecard_df["Bin_Display"] = scorecard_df["Bin"]
        scorecard_df.loc[asa_mask, "Bin_Display"] = scorecard_df.loc[asa_mask, "Bin"].apply(
            clean_asa_bin_labels
        )

        # --- stash a safe copy into session_state (so downstream mutations can't touch it) ---
        ss[ns("score_df")] = scorecard_df.copy()

        # --- prepare an export-only copy where Bin is replaced with human-readable labels ---
        scorecard_export_df = scorecard_df.copy()
        scorecard_export_df["Bin"] = scorecard_export_df["Bin_Display"]
        basic_xlsx_df = scorecard_export_df.drop(columns=["Bin_Display"], errors="ignore")

        # Write Excel for user download using the export copy
        basic_xlsx = BytesIO()
        with pd.ExcelWriter(basic_xlsx, engine="openpyxl") as wr:
            basic_xlsx_df.to_excel(wr, sheet_name="Risk_Scorecard", index=False)
        basic_xlsx.seek(0)
        ss[ns("scorecard_xlsx")] = basic_xlsx

        # IMPORTANT: pass a COPY into create_scorecard_excel so it cannot mutate our session DataFrame
        excel_calc = create_scorecard_excel(
            sc_model,
            sel_vars,
            X_sample=X_tr[sel_vars],
            y_sample=y_tr,
            override_scorecard_df=scorecard_export_df.copy(),
        )

        # keep other session state saves unchanged

        # ---------- stash into session_state
        ss[ns("model")] = sc_model
        ss[ns("sel_vars")] = sel_vars
        ss[ns("score_df")] = scorecard_df
        ss[ns("train_met")] = met_tr
        ss[ns("test_met")] = met_te
        ss[ns("psi_table")] = psi_table
        ss[ns("metrics_comparison_df")] = metrics_comparison_df
        ss[ns("figs")] = {
            "roc": figs_perf["roc"],
            "prc": figs_perf["prc"],
            "cap": figs_perf["cap"],
            "ks": figs_perf["ks"],
            **figs_shap,
            "threshold": fig_thr,
            "psi": fig_psi,
            "distribution": fig_score,
        }
        ss[ns("summary")] = generate_model_summary(
            y_tr,
            y_te,
            tr_pred,
            te_pred,
            sc_model,
            threshold_summary=threshold_summary,
            feature_filter=feature_filter,
        )
        ss[ns("basic_xlsx")] = basic_xlsx
        ss[ns("excel_calc")] = excel_calc
        ss[ns("var_sel_rep")] = sel_rep  # raw dict for later if needed
        plt.close("all")

    # ---------------------------------------------------------------- show
    def _show_results(self) -> None:
        """Render plots, tables, and summary after training."""
        ss, ns = self.ui.session_state, self._key
        plot_visibility = ss[ns("plot_visibility")]
        table_visibility = ss[ns("table_visibility")]

        if ss.get(ns("show_var_sel_report"), False):
            report = ss.get(ns("selection_text_report"))
            if report:
                with self.ui.expander("🧠 Variable Selection Report", expanded=False):
                    self.ui.markdown(f"```\n{report}\n```")

        met_tr, met_te = ss[ns("train_met")], ss[ns("test_met")]

        self.ui.subheader("Model performance")
        col1, col2 = self.ui.columns(2)
        col1.metric("Train AUC", f"{met_tr['auc_roc']:.3f}")
        col2.metric("Test  AUC", f"{met_te['auc_roc']:.3f}")

        selected = [t for t in ("roc", "prc", "cap", "ks") if plot_visibility.get(t)]
        if selected:
            with self.ui.expander("📊 Performance curves", expanded=False):
                for tag in selected:
                    self.ui.markdown(f"**{self.plot_options[tag]}**")
                    self.ui.pyplot(ss[ns("figs")][tag])

        if plot_visibility.get("threshold"):
            with self.ui.expander("📊 Threshold optimisation", expanded=False):
                self.ui.pyplot(ss[ns("figs")]["threshold"])

        if plot_visibility.get("shap"):
            with self.ui.expander("📊 SHAP decision plots", expanded=False):
                for tag in ss[ns("figs")]:
                    if tag.startswith("shap_decision_"):
                        self.ui.pyplot(ss[ns("figs")][tag])

        if table_visibility.get("scorecard"):
            with self.ui.expander("📋 Scorecard table", expanded=False):
                scorecard_df_display = ss[ns("score_df")].copy()
                if "Bin_Display" in scorecard_df_display.columns:
                    scorecard_df_display["Bin"] = scorecard_df_display["Bin_Display"].astype(str)
                    scorecard_df_display = scorecard_df_display.drop(columns=["Bin_Display"])
                self.ui.dataframe(scorecard_df_display)

        if plot_visibility.get("distribution"):
            with self.ui.expander("📊 Score Distribution", expanded=False):
                self.ui.pyplot(ss[ns("figs")]["distribution"])

        if plot_visibility.get("psi") or table_visibility.get("psi_table"):
            with self.ui.expander("📊 PSI Monitoring"):
                if plot_visibility.get("psi"):
                    self.ui.pyplot(ss[ns("figs")]["psi"])
                if table_visibility.get("psi_table"):
                    self.ui.dataframe(ss[ns("psi_table")])

        if table_visibility.get("metrics_comparison_df"):
            with self.ui.expander("📋 Detailed Metrics Comparison", expanded=False):
                df = ss[ns("metrics_comparison_df")]
                self.ui.dataframe(df)

        self.ui.info(ss[ns("summary")])

    # ------------------------------------------------------ ZIP + download
    def _compile_and_download_report(self) -> None:
        """Package visible artifacts into a report zip and trigger download."""
        ss, ns = self.ui.session_state, self._key
        plot_visibility = ss[ns("plot_visibility")]
        table_visibility = ss[ns("table_visibility")]

        # Map table keys to filenames
        table_key_to_session_and_filename = {
            "scorecard": ("score_df", "scorecard_table.csv"),
            "metrics_comparison_df": ("metrics_comparison_df", "detailed_metrics.csv"),
            "psi_table": ("psi_table", "psi_table.csv"),
        }

        # Filter and collect visible tables
        tables = {}
        for vis_key, (session_key, filename) in table_key_to_session_and_filename.items():
            if not table_visibility.get(vis_key, False):
                continue

            df = ensure_dataframe(ss[ns(session_key)]).copy()

            # Special case: scorecard table → use display labels, hide helper column
            if session_key == "score_df":
                if "Bin_Display" in df.columns:
                    # Show cleaned labels under the "Bin" column
                    df["Bin"] = df["Bin_Display"].astype(str)
                    # Hide helper column from exported CSV
                    df = df.drop(columns=["Bin_Display"], errors="ignore")

            tables[filename] = df

        # Filter and collect visible figures
        all_figs = ss[ns("figs")]
        figures = {key: fig for key, fig in all_figs.items() if plot_visibility.get(key, False)}

        # Include SHAP decision plots if SHAP is visible
        if plot_visibility.get("shap", False):
            figures.update(
                {key: fig for key, fig in all_figs.items() if key.startswith("shap_decision_")}
            )

        # Include summary
        summary = ss[ns("summary")]

        # Handle variable selection report
        selection_text_report = (
            ss[ns("selection_text_report")] if ss.get(ns("show_var_sel_report"), False) else None
        )
        selection_text_bytes = (
            BytesIO(selection_text_report.encode("utf-8")) if selection_text_report else None
        )

        # Extra files (conditionally include variable selection text)
        extra = [
            ("basic_scorecard.xlsx", ss[ns("basic_xlsx")]),
            ("interactive_calculator.xlsx", ss[ns("excel_calc")]),
        ]
        if selection_text_bytes:
            extra.append(("variable_selection.txt", selection_text_bytes))

        # Build the report
        self._build_report(
            tables=tables,
            figures=figures,
            summary=summary,
            zip_name="clinical_scorecard_report.zip",
            extra_files=extra,
            variable_selection_text=selection_text_report,
        )
