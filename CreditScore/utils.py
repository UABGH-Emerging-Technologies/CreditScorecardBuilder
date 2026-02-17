"""Shared helpers for metrics, reporting, and data cleaning."""

import io
import json
import logging
import re
import textwrap
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from docx import Document
from docx.shared import Inches, Pt
from rapidfuzz import fuzz, process
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

from CreditScore.evaluate import ModelEvaluator
from CreditScore.train import StatsmodelsLogisticModel

logger = logging.getLogger(__name__)


class ModelMetrics:
    """Common model metrics calculations."""

    @staticmethod
    def calculate_cap_curve(y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
        """
        Compute the CAP (Cumulative Accuracy Profile) score.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities

        Returns:
            CAP score (area under the CAP curve)
        """
        import numpy as np
        from sklearn.metrics import auc

        # Sort probabilities and corresponding true labels
        sorted_indices = np.argsort(-y_pred_proba)
        sorted_true_labels = y_true.iloc[sorted_indices].values

        # Compute cumulative sum of positives
        total_positives = sum(y_true)
        cumulative_positives = np.cumsum(sorted_true_labels)
        percentage_positives = cumulative_positives / total_positives

        # Compute the CAP score (area under curve)
        cap_score = auc(np.linspace(0, 1, len(percentage_positives)), percentage_positives)

        return cap_score

    @staticmethod
    def calculate_metrics(
        y_true: pd.Series, y_pred_proba: np.ndarray, y_pred_class: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate both probability-based and classification-based metrics."""

        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]

        evaluator = ModelEvaluator(model=None)
        precision, recall, auc_pr = evaluator.compute_precision_recall(y_true, y_pred_proba)
        ks_stat, _ = ks_2samp(y_pred_proba[y_true == 1], y_pred_proba[y_true == 0])
        cap_score = ModelMetrics.calculate_cap_curve(y_true, y_pred_proba)

        metrics = {
            "auc_roc": roc_auc_score(y_true, y_pred_proba),
            "auc_pr": auc_pr,
            "ks_stat": ks_stat,
            "cap_curve": cap_score,
        }

        if y_pred_class is not None:
            metrics.update(
                {
                    "accuracy": accuracy_score(y_true, y_pred_class),
                    "precision_thresholded": precision_score(y_true, y_pred_class),
                    "recall_thresholded": recall_score(y_true, y_pred_class),
                    "f1_score": f1_score(y_true, y_pred_class),
                }
            )

        return metrics

    @staticmethod
    def create_metrics_comparison_df(metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Create comparison DataFrame from metrics dictionary.

        Args:
            metrics_dict: Dictionary of model_name -> metrics

        Returns:
            DataFrame with metrics comparison
        """
        df = pd.DataFrame(metrics_dict).T
        df = df.round(4)
        return df


class VisualizationHelper:
    """Common visualization utilities."""

    @staticmethod
    def setup_plot_style():
        """Set up consistent plot styling."""
        plt.style.use("seaborn-v0_8-darkgrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 10

    @staticmethod
    def save_figure(fig: plt.Figure, save_path: str, dpi: int = 300):
        """
        Save figure with consistent settings.

        Args:
            fig: Matplotlib figure
            save_path: Path to save figure
            dpi: DPI for saved figure
        """
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    @staticmethod
    def create_subplots(
        n_plots: int, n_cols: int = 2, figsize: Optional[Tuple[float, float]] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create subplots with consistent sizing.

        Args:
            n_plots: Number of plots needed
            n_cols: Number of columns
            figsize: Figure size (auto-calculated if None)

        Returns:
            Tuple of (figure, axes)
        """
        n_rows = int(np.ceil(n_plots / n_cols))

        if figsize is None:
            figsize = (n_cols * 5, n_rows * 4)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Flatten axes for easy iteration
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Hide extra subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        return fig, axes


class ConfigurationManager:
    """Manage model configurations using JSON config files."""

    @staticmethod
    def scorecard_args() -> Dict[str, Any]:
        """Get scorecard configuration for model initialization."""
        from config.config import Config

        return Config.scorecard_args()

    @staticmethod
    def get_scorecard_fit_config() -> Dict[str, Any]:
        """Get cross-validation configuration for model fitting."""
        from config.config import Config

        config = Config.cross_validation_args()

        # Extract fit-specific parameters
        return {
            "hyperparameter_search": config.get("hyperparameter_search", True),
            "param_grid": config.get(
                "param_grid", {"C": [0.01, 0.1, 1, 10], "l1_ratio": [0.1, 0.5, 0.9]}
            ),
            "cv": config.get("cv", 5),
        }

    @staticmethod
    def binning_args() -> Dict[str, Any]:
        """Get binning configuration optimized for medical data with class imbalance."""
        from config.config import Config

        return Config.binning_args()

    @staticmethod
    def monitoring_args() -> Dict[str, Any]:
        """Get monitoring configuration."""
        from config.config import Config

        return Config.monitoring_args()

    @staticmethod
    def power_analysis_args() -> Dict[str, Any]:
        """Get power analysis configuration."""
        from config.config import Config

        config = Config.power_analysis_args()

        # Extract legacy format for backward compatibility
        return {
            "alpha": config.get("default_alpha", 0.05),
            "power": config.get("default_power", 0.80),
            "min_sample_size": config.get("simulation", {}).get("min_sample_size", 50),
            "max_sample_size": config.get("simulation", {}).get("max_sample_size", 500),
            "step_size": config.get("simulation", {}).get("step_size", 50),
            "n_simulations": config.get("simulation", {}).get("n_simulations", 1000),
        }

    @staticmethod
    def variable_selection_args() -> Dict[str, Any]:
        """Get variable selection configuration."""
        from config.config import Config

        return Config.variable_selection_args()


class CacheMixin:
    """Cached helpers for model fitting and diagnostics."""

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def fit_logit_model(X, y, target_column):
        """Fit a statsmodels logistic model and return coefficients."""
        mdl = StatsmodelsLogisticModel()
        mdl.fit(X, y, target_name=target_column)
        coef = mdl.get_coefficients()
        coef["Odds_Ratio"] = np.exp(coef["Coefficient"])
        if {"CI_Lower", "CI_Upper"}.issubset(coef.columns):
            coef["OR_CI_Lower"] = np.exp(coef["CI_Lower"])
            coef["OR_CI_Upper"] = np.exp(coef["CI_Upper"])
        return mdl, coef

    @staticmethod
    @st.cache_data(show_spinner=False)
    def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
        """Compute variance inflation factors for numeric columns."""
        x = sm.add_constant(df.astype(float), has_constant="add")
        vif = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
        out = pd.DataFrame({"Feature": x.columns, "VIF": vif})
        return out[out.Feature != "const"]

    @staticmethod
    @st.cache_data(show_spinner=False)
    def to_csv_bytes(df: pd.DataFrame) -> bytes:
        """Serialize a DataFrame to UTF-8 CSV bytes."""
        return df.to_csv(index=False).encode("utf-8")


class LogisticWrapper:
    """Expose scikit-learn coefficients for SHAP compatibility."""

    def __init__(self, model):
        self.model = model
        # Expose the model parameters so SHAP can inspect them.
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_

    def __call__(self, X):
        # Return the model's decision function (i.e., log odds)
        return self.model.decision_function(X)


def generate_model_summary(
    y_train,
    y_test,
    y_train_pred,
    y_test_pred,
    scorecard_model,
    threshold_summary: Optional[str] = None,
    feature_filter: Optional[str] = None,
):
    """Generate a structured summary of model accuracy, top risk factors, and dataset distribution."""

    # Compute AUC scores for accuracy
    train_auc = ModelMetrics.calculate_metrics(y_train, y_train_pred)["auc_roc"]
    test_auc = ModelMetrics.calculate_metrics(y_test, y_test_pred)["auc_roc"]

    # Updated accuracy classification
    if test_auc >= 0.90:
        accuracy_statement = "highly reliable for risk assessment"
    elif test_auc >= 0.80:
        accuracy_statement = "moderately reliable, suitable for risk assessment"
    elif test_auc >= 0.70:
        accuracy_statement = "somewhat reliable, may require additional validation"
    else:
        accuracy_statement = "less reliable, requiring additional validation"

    # Compare training vs test similarity
    similarity = abs(train_auc - test_auc) * 100
    similarity_statement = (
        "high similarity"
        if similarity < 5
        else "moderate similarity" if similarity < 10 else "low similarity"
    )

    # Get top risk-driving variables via coefficients
    coef_df = scorecard_model.get_coefficients().sort_values(by="Coefficient", ascending=True)
    coef_df = coef_df[coef_df["Feature"] != "Intercept"]
    top_risk_factors = coef_df.head(3)["Feature"].tolist()

    # Target variable distribution
    class_counts_train = y_train.value_counts()
    class_counts_test = y_test.value_counts()
    class_pct_train = class_counts_train / class_counts_train.sum() * 100
    class_pct_test = class_counts_test / class_counts_test.sum() * 100

    if feature_filter:
        filter_text = f"The model was trained using **{feature_filter}** feature subset."
    else:
        filter_text = "The model was trained using the **full feature set**."
    # Generate summary paragraph
    summary_text = f"""
    For research purposes only, use this model with caution.
    The model is **{accuracy_statement}**, with a test accuracy of **{test_auc:.2f} (AUC)**.
    {threshold_summary}. {filter_text}
    The highest risk factors determined were: **{', '.join(top_risk_factors)}**.
    The training set and test set are **{similarity:.1f}% alike**, signifying **{similarity_statement}** in generalization.
    The target variable distribution in the training data is **Class 0: {class_pct_train[0]:.1f}% ({class_counts_train[0]} samples)** and
    **Class 1: {class_pct_train[1]:.1f}% ({class_counts_train[1]} samples)**.  In the test data, **Class 0: {class_pct_test[0]:.1f}% ({class_counts_test[0]} samples)** and **Class 1: {class_pct_test[1]:.1f}% ({class_counts_test[1]} samples)**.
    For single patient risk please use the interactive risk calculator.
    """

    # Display in Streamlit
    st.info(summary_text)
    return summary_text


def clean_add_results_to_docx(self, doc, results):
    """Docx table renderer with custom column sizing."""
    import textwrap

    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    total_available_width = 7.5  # Approximate usable width
    reserved_widths = {
        0: 0.85,
        1: 0.25,
        2: 1,
        3: 0.6,
        5: 0.6,
        6: 0.45,
        8: 0.6,
        11: 0.6,
        12: 0.45,
    }  # first and third columns
    row_shade_color = "F2F2F2"  # light gray for alternating rows
    target_table = "scorecard_table.csv"

    for method_name, sections in results.items():
        for section_name, df in sections.items():
            if not isinstance(df, pd.DataFrame):
                continue

            df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
            if df.index.name or df.index.names[0]:
                df.reset_index(inplace=True)

            heading = f"{method_name}" if not section_name else f"{method_name} - {section_name}"
            doc.add_heading(heading, level=2)

            if method_name == target_table:
                num_cols = len(df.columns)
                fixed = [i for i in reserved_widths if i < num_cols]
                remaining_width = total_available_width - sum(reserved_widths[i] for i in fixed)
                flexible = [i for i in range(num_cols) if i not in reserved_widths]
                default_width = remaining_width / max(len(flexible), 1)

                table = doc.add_table(rows=1, cols=num_cols)
                table.style = "Table Grid"
                table.autofit = False

                # Set header widths
                for i in range(num_cols):
                    w = reserved_widths.get(i, default_width)
                    cell = table.rows[0].cells[i]
                    cell.width = Inches(w)
                    table.columns[i].width = Inches(w)
                    cell.text = textwrap.fill(str(df.columns[i]), width=20)

                # Add data rows
                for r_idx, (_, row) in enumerate(df.iterrows()):
                    cells = table.add_row().cells
                    for i, val in enumerate(row):
                        cells[i].text = textwrap.fill(str(val), width=40)
                        cells[i].width = Inches(reserved_widths.get(i, default_width))

                        # Alternate shading every other row
                        if r_idx % 2 == 1:
                            tc = cells[i]._tc
                            tcPr = tc.get_or_add_tcPr()
                            shd = OxmlElement("w:shd")
                            shd.set(qn("w:fill"), row_shade_color)
                            tcPr.append(shd)

                # Tight spacing
                for row in table.rows:
                    for cell in row.cells:
                        for para in cell.paragraphs:
                            para.paragraph_format.space_after = Pt(0)
            else:
                # 🧁 Use default formatting for other tables
                table = doc.add_table(rows=1, cols=len(df.columns))
                table.style = "Table Grid"

                for i, col in enumerate(df.columns):
                    table.rows[0].cells[i].text = str(col)

                for _, row in df.iterrows():
                    cells = table.add_row().cells
                    for i, val in enumerate(row):
                        cells[i].text = str(val)


def calculate_ranking_based_cap_score(
    y_true: pd.Series, y_pred_proba: np.ndarray, preds: np.ndarray
) -> float:
    """Score a CAP curve based on ranked predictions."""
    n = len(y_true)
    total_positives = np.sum(y_true)

    if total_positives == 0:
        return 0.0

    sorted_indices = np.argsort(-y_pred_proba)
    y_sorted = y_true.iloc[sorted_indices].values

    top_n = sum(preds)
    if top_n == 0:
        return 0.0

    y_top = y_sorted[:top_n]
    captured = np.sum(y_top) / total_positives

    return captured


def run_integrity_checks(df: pd.DataFrame):
    """Render a Streamlit data integrity summary."""
    st.subheader("🔍 Data Integrity Summary")

    # Display data types
    st.write("📘 Column Data Types:")
    st.write(df.dtypes)

    # Null value overview
    st.write("📉 Missing Value Counts:")
    st.write(df.isnull().sum())

    # Look for boolean-like columns
    bool_like = [col for col in df.columns if df[col].dropna().isin([True, False]).all()]
    st.write("🟨 Likely Boolean Columns:")
    st.write(bool_like)

    for col in bool_like:
        st.write(f"🔹 `{col}` value breakdown:")
        st.write(df[col].value_counts(dropna=False))

    # Catch suspicious object-type columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        st.write("🔍 Object-Type Column Samples:")
        for col in obj_cols:
            st.write(f"`{col}` samples: {df[col].unique()[:5]}")


def dict_to_markdown(d, indent=0):
    """Recursively format dictionary as Markdown"""
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


def ensure_dataframe(obj, col_name="value"):
    """Ensure an object is represented as a DataFrame."""
    from pandas import DataFrame, Series

    if isinstance(obj, DataFrame):
        return obj
    if isinstance(obj, Series):
        return obj.to_frame().reset_index()
    if isinstance(obj, dict):
        vals = list(obj.values())
        if all(isinstance(v, (list, tuple)) for v in vals) and len(set(len(v) for v in vals)) == 1:
            return pd.DataFrame(obj)
        return pd.DataFrame([obj])
    if isinstance(obj, (list, tuple)):
        if len(obj) and isinstance(obj[0], dict):
            return pd.DataFrame(obj)
        return pd.DataFrame({col_name: [v for v in obj]})
    return pd.DataFrame({col_name: [obj]})


class DefinitionMatcher:
    """Match feature names to phenotype definitions."""

    def __init__(self, json_path: str, threshold: int = 80, verbose: bool = False):
        """
        json_path: path to your phenotype JSON
        threshold: fuzzy match score threshold (0-100)
        verbose: if True, stores last_match info for debugging
        """
        self.threshold = threshold
        self.verbose = verbose
        self.raw: Dict[str, Dict[str, str]] = (
            {}
        )  # phenotype -> return_columns map (decoded phenotype name)
        self.return_key_index: Dict[str, Tuple[str, str]] = (
            {}
        )  # return_key -> (phenotype, return_key)
        self.composite_index: Dict[str, Tuple[str, str]] = (
            {}
        )  # "phenotype|return_key" -> (phenotype, return_key)
        self._load(json_path)
        self._build_search_index()
        self.last_match_info: Optional[Dict] = None

    # ------- Loading and indexing -------
    def _load(self, path: str) -> None:
        """Load phenotype metadata from JSON."""
        with open(path, "r") as fh:
            raw_json = json.load(fh)

        for raw_phen_name, entry in raw_json.items():
            phen_name = unquote(raw_phen_name).strip()
            return_map = entry.get("return_columns", {}) or {}
            # Normalize keys and preserve original mapping order / values
            decoded_map: Dict[str, str] = {}
            for k, v in return_map.items():
                decoded_k = unquote(k).strip()
                # Skip empty keys that are just "" : ""
                if decoded_k == "" and (v == "" or v is None):
                    continue
                decoded_map[decoded_k] = v
            if decoded_map:
                self.raw[phen_name] = decoded_map

    def _build_search_index(self) -> None:
        """Build lookup indices for phenotype matching."""
        self.return_key_index = {}
        self.composite_index = {}
        self._token_index = []
        for phen, rmap in self.raw.items():
            for rkey in rmap.keys():
                self.return_key_index[rkey] = (phen, rkey)
                composite = f"{phen}|{rkey}"
                self.composite_index[composite] = (phen, rkey)
            all_keys = [phen] + list(rmap.keys())
            for k in all_keys:
                norm = self._normalize(k)
                tokens = set(norm.split())
                self._token_index.append((tokens, phen))

        # candidates: return_keys, phenotype names, composite strings
        self._fuzzy_candidates = (
            list(self.return_key_index.keys())
            + list(self.raw.keys())
            + list(self.composite_index.keys())
        )

        # normalized_map: normalized string -> list(original candidate strings)
        norm_map = defaultdict(list)
        for cand in self._fuzzy_candidates:
            norm = self._normalize(cand)
            if norm:
                norm_map[norm].append(cand)
        self._normalized_map = dict(norm_map)
        # list of normalized choices for fuzzy scorer
        self._normalized_choices = list(self._normalized_map.keys())

    # ------- Normalization -------
    def _normalize(self, text: str) -> str:
        """Normalize strings for fuzzy matching."""
        if not text:
            return ""
        s = str(text)
        s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)  # split CamelCase
        s = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", s)
        s = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", s)
        s = re.sub(r"[:\-\_\(\)\[\]\.,/\\]+", " ", s)  # punctuation to space
        s = re.sub(r"\s+", " ", s).strip().lower()
        s = re.sub(
            r"\b(value|code|vc|inmls|mls|total|median|actual|status|classification|count)\b", " ", s
        )
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # ------- Formatting output -------
    def _format_return_map(self, phen: str, rmap: Dict[str, str]) -> str:
        """
        Convert return_columns dict into a single-line definition string.
        Prefer preserving label: code (or [code] label) formats; preserve non-numeric types like "Text"/"Number".
        Example output:
            [0] Unknown Concept, [500] Inpatient, AdmissionType: Text, AdmissionType_VC: Number
        """
        parts: List[str] = []
        for label, code in rmap.items():
            # Keep the original label and value representation
            code_str = "" if code is None else str(code)
            # Distinguish typeless textual codes like "Text"/"Number"
            if code_str.lower() in {"text", "number", "[text]", "[number]"}:
                parts.append(f"{label}: {code_str}")
            else:
                parts.append(f"[{code_str}] {label}")
        return ", ".join(parts)

    # ------- Core lookup -------
    def get_definition(self, feature: str) -> str:
        """
        Returns a single string containing the matched phenotype's return_columns entries
        relevant to `feature`. Uses multiple strategies:
          1) Exact return_column key match
          2) Exact phenotype name match
          3) Exact composite "phenotype|return_key" match
          4) Token-sort fuzzy match across all candidates
        If no match above threshold, returns "No definition available".
        """
        self.last_match_info = None
        if not feature or not feature.strip():
            return "No definition available"

        feature = feature.strip()
        # 1) Exact return_column key
        if feature in self.return_key_index:
            phen, rkey = self.return_key_index[feature]
            rmap = self.raw.get(phen, {})
            # return the whole phenotype return_columns (user requested that the phenotype and its return_columns be kept together)
            out = self._format_return_map(phen, rmap)
            self.last_match_info = {
                "method": "exact_return_key",
                "phenotype": phen,
                "return_key": rkey,
                "score": 100,
            }
            return out

        # 2) Exact phenotype name
        if feature in self.raw:
            rmap = self.raw[feature]
            out = self._format_return_map(feature, rmap)
            self.last_match_info = {"method": "exact_phenotype", "phenotype": feature, "score": 100}
            return out

        # 3) Exact composite keys
        for comp in self.composite_index:
            if feature == comp or feature.lower() == comp.lower():
                phen, rkey = self.composite_index[comp]
                rmap = self.raw.get(phen, {})
                out = self._format_return_map(phen, rmap)
                self.last_match_info = {
                    "method": "exact_composite",
                    "phenotype": phen,
                    "return_key": rkey,
                    "score": 100,
                }
                return out

        # 4) Token-overlap fuzzy matching
        norm = self._normalize(feature)
        tokens = set(norm.split())

        candidates = []
        for token_set, phen in self._token_index:
            if tokens & token_set:
                candidates.append((phen, token_set))

        scored = [(phen, len(tokens & token_set)) for phen, token_set in candidates]
        # Boost phenotypes that explicitly contain "Age (Years)"
        if not scored:
            self.last_match_info = {
                "method": "token_overlap",
                "matched_phenotype": None,
                "score": 0,
            }
            return "No definition available"
        # Sort by overlap score, then prefer phenotype name matches
        scored.sort(key=lambda x: (x[1], self._normalize(x[0]).count("sex")), reverse=True)
        # Sort by overlap score, then prefer phenotype name matches
        scored.sort(key=lambda x: (x[1], self._normalize(x[0]).count(norm)), reverse=True)

        # Filter out matches where phenotype name doesn't contain any input tokens
        for phen, score in scored:
            phen_tokens = set(self._normalize(phen).split())
            rkeys = self.raw.get(phen, {}).keys()
            rkey_tokens = set()
            for rk in rkeys:
                rkey_tokens |= set(self._normalize(rk).split())
            all_tokens = phen_tokens | rkey_tokens
            if tokens & all_tokens:
                rmap = self.raw.get(phen, {})
                out = self._format_return_map(phen, rmap)
                self.last_match_info = {
                    "method": "token_overlap",
                    "matched_phenotype": phen,
                    "score": score,
                }
                return out

        # If no semantically valid match, return fallback
        self.last_match_info = {
            "method": "token_overlap",
            "matched_phenotype": None,
            "score": scored[0][1],
        }
        return "No definition available"
