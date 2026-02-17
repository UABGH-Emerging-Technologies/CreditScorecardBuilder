"""
Unified Streamlit UI for

  • 🩺 Clinical Risk Scorecard Builder   (default tab)
  • 📊 Logistic-Regression Coefficients & Power Analysis

The heavy lifting lives in `CreditScore.Streamlit.streamlit_interface`
and corresponding children classes (scorecard_builder and clinical_model_explorer).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from aiweb_common.streamlit.page_renderer import StreamlitUIHelper

from config.config import Config
from CreditScore.Streamlit.clinical_model_explorer import ClinicalModelExplorer
from CreditScore.Streamlit.scorecard_builder import ScorecardBuilder


def main() -> None:  # pragma: no cover
    """Initialise Streamlit and dispatch the two sub-apps."""
    st.set_page_config(page_title="Clinical Analytics Suite", page_icon="🤖")
    ui = StreamlitUIHelper()

    ui.title("🩺📋💼 Clinical Analytics Suite")

    tab_scorecard, tab_lr = ui.tabs(["Clinical Risk Scorecard", "Model Coefficients & Power"])

    # ------------------------------------------------------------------ #
    # Tab 1 – Scorecard
    # ------------------------------------------------------------------ #
    with tab_scorecard:
        ui.subheader("🚑📈💓 Clinical Risk Scorecard Builder")
        ui.markdown(Config.SC_HEADER_MARKDOWN)
        ScorecardBuilder(ui).render()

    # ------------------------------------------------------------------ #
    # Tab 2 – Logistic regression + power analysis
    # ------------------------------------------------------------------ #
    with tab_lr:
        ui.subheader("🧬💻🔬 Logistic Regression and Power Analysis Builder")
        ui.markdown(Config.LR_HEADER_MARKDOWN)
        ClinicalModelExplorer(ui).render()


# ----------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    main()
