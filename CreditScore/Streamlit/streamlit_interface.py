"""
streamlit_interface
-------------------
A common, DRY UI layer for all Streamlit pages of the CreditScore package.

This file defines `BaseHandler`, which bundles:
  • page initialisation / side-bar plumbing
  • file upload and DataFrame-loader helpers
  • a generic _build_report() that collects tables, figures + extra files
    into a DOCX and ZIP via ReportBuilder
  • namespaced access to Streamlit session_state
"""

from __future__ import annotations

import io
import json
import warnings
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from aiweb_common.file_operations.upload_manager import StreamlitUploadManager

from CreditScore.data import _filter_features_by_category
from CreditScore.report_builder import ReportBuilder, compile_report_bytes
from llm_utils.aiweb_common.streamlit.page_renderer import StreamlitUIHelper

warnings.filterwarnings("ignore", category=FutureWarning)

# ────────────────────────────────────────────────────────────────────────────
# Optional small utilities kept here so children don't need to re-import them
# ────────────────────────────────────────────────────────────────────────────


def ensure_dataframe(obj: Any, col_name: str = "value") -> pd.DataFrame:
    """
    Convert various simple Python objects to a minimally useful DataFrame
    (fallback helper for report tables).
    """
    from pandas import DataFrame, Series

    if isinstance(obj, DataFrame):
        return obj
    if isinstance(obj, Series):
        return obj.to_frame().reset_index()
    if isinstance(obj, dict):
        # if dict of equal-length lists       → normal DF
        # else: single-row DF
        vals = list(obj.values())
        if all(isinstance(v, (list, tuple)) for v in vals) and len({len(v) for v in vals}) == 1:
            return pd.DataFrame(obj)
        return pd.DataFrame([obj])
    if isinstance(obj, (list, tuple)):
        if len(obj) and isinstance(obj[0], dict):
            return pd.DataFrame(obj)
        return pd.DataFrame({col_name: list(obj)})
    return pd.DataFrame({col_name: [obj]})


def dict_to_markdown(d: Dict[str, Any], indent: int = 0) -> str:
    """Pretty print nested dicts as Markdown for inclusion in DOCX."""
    md = ""
    for k, v in d.items():
        if isinstance(v, dict):
            md += "  " * indent + f"**{k}:**\n"
            md += dict_to_markdown(v, indent + 1)
        elif isinstance(v, list):
            md += "  " * indent + f"**{k}:**\n"
            for item in v:
                md += "  " * (indent + 1) + f"- {item}\n"
        else:
            md += "  " * indent + f"**{k}:** {v}\n"
    return md


# ────────────────────────────────────────────────────────────────────────────
class BaseHandler(ABC):
    """
    Parent class for all Streamlit pages.

    Sub-classes should:
      • set self.page_title / icon / defaults in __init__
      • implement       render()     – orchestrates the UI flow
      • optionally use  _key(), _load_data(), _build_report()
    """

    # ------------------------------------------------------------------ init
    def __init__(self, ui: StreamlitUIHelper) -> None:
        self.ui = ui

        # overridable defaults
        self.page_title: str = "Base Page"
        self.page_icon: str = "🏷️"
        self.file_types: Tuple[str, ...] = ("csv", "xlsx")
        self.default_features: List[str] = []
        self.default_target_idx: int = 0
        self.plot_options: Dict[str, str] = {}  # e.g. {"roc": "ROC Curve"}
        self.table_options: Dict[str, str] = {}  # e.g. {"scorecard": "Scorecard Table"}
        self.show_sidebar: bool = True

    # =================================================================
    # generic helpers – sub-classes usually call these
    # =================================================================
    # --------------------------------------------------------- UI setup
    def _init_page(self) -> None:
        """Initialize Streamlit page settings and title."""
        try:
            self.ui.setup_page(
                page_title=self.page_title,
                page_icon=self.page_icon,
                hide_branding=True,
            )
        except Exception:  # pragma: no cover
            pass
        self.ui.title(self.page_title)

    # --------------------------------------------------------- load data
    def _load_data(self) -> pd.DataFrame:
        """Standard file-uploader → pandas.DataFrame helper."""
        up_file = self.ui.file_uploader(
            f"Upload data ({', '.join(self.file_types)})",
            type=list(self.file_types),
            key=self._key("upload"),
        )
        if not up_file:
            self.ui.info("Please upload a file to continue.")
            return pd.DataFrame()

        # 1) try aiweb_common upload-manager (handles Excel sheets nicely)
        try:
            df, _meta = StreamlitUploadManager(
                up_file, accept_multiple_files=False
            ).process_upload()
            return df
        except Exception:
            pass

        # 2) fallback to generic loader
        from CreditScore.data import DataLoader

        loader = DataLoader()
        return loader.load_file(up_file.name, file_obj=up_file)

    # --------------------------------------------------------- col select
    def _select_columns(self, data: pd.DataFrame) -> Tuple[str, List[str]]:
        """Pick target/feature columns with category filtering."""
        from config.config import default_features, default_target

        cols = data.columns.to_list()
        self.default_target_idx = cols.index(default_target) if default_target in cols else 0

        # Target first (from full data)
        target = self.ui.selectbox(
            "Select outcome / target column",
            cols,
            index=self.default_target_idx,
            key=self._key("target"),
        )

        # Timing dropdown
        timing_category = self.ui.selectbox(
            "Choose feature category",
            ["All", "Preop", "Intraop", "Postop"],
            index=0,
            key=self._key("feature_category"),
        )

        # Filter data
        filtered_data = _filter_features_by_category(data, timing_category, target)

        # Feature selection
        feats = self.ui.select_columns(
            default_columns=default_features,
            data=filtered_data,
            select_type="features",
            exclude_column=target,
            key=self._key("feats"),
        )

        return target, feats

    # --------------------------------------------------------- build rep
    def _build_report(
        self,
        tables: Dict[str, pd.DataFrame],
        figures: Dict[str, Any],
        summary: str = "",
        zip_name: str = "analysis_report.zip",
        extra_files: Optional[List[Tuple[str, bytes]]] = None,
        variable_selection_text: Optional[str] = None,
    ) -> None:
        """
        Build DOCX + ZIP with tables, figures, summary + optional extra files,
        then offer a Streamlit `download_button`.
        """

        # Hand tables + figures to the Word-doc creator
        docx_bytes = compile_report_bytes(
            summary=summary,
            results={name: {"": df} for name, df in tables.items()},
            figures=figures,
            variable_selection_text=variable_selection_text,
        )

        # Assemble ZIP
        with ReportBuilder() as rb:
            # add CSVs
            for fname, df in tables.items():
                rb.add_dataframe(df, fname)

            # add DOCX
            if isinstance(docx_bytes, io.BytesIO):
                docx_bytes.seek(0)
            rb.add_bytes(docx_bytes, "report.docx")

            # add stand-alone figure PNGs
            for fig_name, fig in figures.items():
                if isinstance(fig, plt.Figure):
                    rb.add_figure(fig, f"{fig_name}.png")
                    plt.close(fig)  # <--- ADD THIS LINE! This is the most likely culprit
                else:
                    print(
                        f"Warning: Skipping figure '{fig_name}' of type {type(fig)} from ZIP as it's not a Matplotlib Figure."
                    )

            # any extras (e.g., Excel calculators)
            if extra_files:
                for fname, file_bytes in extra_files:
                    rb.add_bytes(file_bytes, fname)

            zip_bytes = rb.build_zip()

        # Download button
        self.ui.download_button(
            "📦 Download full report",
            data=zip_bytes,
            mime="application/zip",
            file_name=zip_name,
        )

    # --------------------------------------------------------- namespace
    def _key(self, suffix: str) -> str:
        """Namespace keys → avoid collisions across multiple pages/tabs."""
        return f"{self.__class__.__name__}_{suffix}"

    def _render_sidebar(self):
        """
        Render the sidebar UI using StreamlitUIHelper's proxy.
        Child class must configure plot_options/table_options.
        """
        if not self.show_sidebar:
            return
        ss = self.ui.session_state
        ns = self._key
        self.default_visible_plots = {"roc", "prc", "psi"}
        self.default_visible_tables = {"psi_table", "metrics_comparison_df"}
        sb = self.ui.sidebar  # this is your _ContainerProxy!

        if ns("plot_visibility") not in ss:
            ss[ns("plot_visibility")] = {
                k: k in self.default_visible_plots for k in self.plot_options
            }
        if ns("table_visibility") not in ss:
            ss[ns("table_visibility")] = {
                k: k in self.default_visible_tables for k in self.table_options
            }

        sb.subheader(f"🧭 User Options for {self.page_title}")

        if self.page_title == "Clinical Risk Scorecard Builder":
            sb.selectbox(
                "Select thresholding method:",
                ["Optimized", "Event rate", "Default (0.5)"],
                # index=0,
                key=ns("threshold_choice"),
                help="Choose how classification threshold is selected",
            )
            sb.header("📋 Data Views")
            sb.checkbox(
                "Show Baseline Reference",
                value=ss.get(ns("use_baseline_rate"), True),
                key=ns("use_baseline_rate"),
            )
            # sb.checkbox(
            #     "Show Data Integrity Summary",
            #     #value=ss.get(ns("integrity_check"), False),
            #     key=ns("integrity_check")
            # )
            sb.checkbox(
                "Show data preview",
                value=ss.get(ns("show_data_preview"), True),
                key=ns("show_data_preview"),
                help="Expand to inspect the uploaded dataset",
            )
            sb.checkbox(
                "Show Variable Selection Report",
                value=ss.get(ns("show_var_sel_report"), True),
                key=ns("show_var_sel_report"),
            )
            sb.markdown("---")
        sb.header("📊📋Plot and Table Options")
        col1, col2 = sb.columns(2)
        with col1:
            if col1.button("Select All", key=ns("select_all_btn")):
                # When "Select All" is clicked, update both internal dictionaries
                # AND the individual checkbox keys in session_state
                for k in self.plot_options:  # Iterate through actual options, not just ss keys
                    ss[ns(f"plot_{k}")] = True  # Update individual checkbox state
                    ss[ns("plot_visibility")][k] = True  # Update the summary dictionary
                for k in self.table_options:
                    ss[ns(f"table_{k}")] = True  # Update individual checkbox state
                    ss[ns("table_visibility")][k] = True  # Update the summary dictionary
                # st.rerun()

        with col2:
            if col2.button("Clear All", key=ns("deselect_all_btn")):
                # Similar to "Select All", update both internal dictionaries
                # AND the individual checkbox keys in session_state
                for k in self.plot_options:
                    ss[ns(f"plot_{k}")] = False
                    ss[ns("plot_visibility")][k] = False
                for k in self.table_options:
                    ss[ns(f"table_{k}")] = False
                    ss[ns("table_visibility")][k] = False
                # st.rerun()

        if self.plot_options:
            sb.header("📊 Plots")
            for k, label in self.plot_options.items():
                # Let Streamlit manage the checkbox state via its key.
                # Initialize the individual checkbox key in session_state
                # if it doesn't exist, using the value from plot_visibility.
                # This ensures consistency on first load.
                if ns(f"plot_{k}") not in ss:
                    ss[ns(f"plot_{k}")] = ss[ns("plot_visibility")][k]

                # Render the checkbox. Its current state will be in ss[ns(f"plot_{k}")]
                checkbox_state = sb.checkbox(
                    label,
                    # value=ss[ns(f"plot_{k}")], # Use the *individual* session_state key for value
                    key=ns(f"plot_{k}"),  # This links the checkbox directly to this ss key
                    help=f"Show/hide {label.lower()}",
                )
                # Crucially: Update the 'plot_visibility' dictionary based on the checkbox's *current* state
                # This makes the individual checkbox the source of truth for the dict entry
                ss[ns("plot_visibility")][k] = checkbox_state

        if self.table_options:
            sb.header("📋 Tables")
            for k, label in self.table_options.items():
                if ns(f"table_{k}") not in ss:
                    ss[ns(f"table_{k}")] = ss[ns("table_visibility")][k]

                checkbox_state = sb.checkbox(
                    label,
                    # value=ss[ns(f"table_{k}")],
                    key=ns(f"table_{k}"),
                    help=f"Show/hide {label.lower()}",
                )
                ss[ns("table_visibility")][k] = checkbox_state

        sb.markdown("---")
