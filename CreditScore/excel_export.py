# CreditScore/excel_export.py

import io
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import (
    Alignment,
    Border,
    Font,
    PatternFill,
    Protection,
    Side,
)
from openpyxl.utils import get_column_letter, quote_sheetname
from openpyxl.worksheet.datavalidation import DataValidation

from config.config import Config
from CreditScore.utils import DefinitionMatcher

sys.path.append(str(Path(__file__).parent.parent))


class ExcelModelExporter:
    """Base class for exporting models to Excel with formulas."""

    def __init__(self):
        self.wb = Workbook()
        self.ws = self.wb.active

        # Define styles
        self.header_style = {
            "font": Font(bold=True, color="FFFFFF", size=12),
            "fill": PatternFill(start_color="366092", end_color="366092", fill_type="solid"),
            "alignment": Alignment(horizontal="center", vertical="center"),
            "border": Border(
                left=Side(style="thin"),
                right=Side(style="thin"),
                top=Side(style="thin"),
                bottom=Side(style="thin"),
            ),
        }

        self.input_style = {
            "fill": PatternFill(start_color="E7F3FF", end_color="E7F3FF", fill_type="solid"),
            "border": Border(
                left=Side(style="thin"),
                right=Side(style="thin"),
                top=Side(style="thin"),
                bottom=Side(style="thin"),
            ),
            "protection": Protection(locked=False),
        }

        self.formula_style = {
            "fill": PatternFill(start_color="D9EAD3", end_color="D9EAD3", fill_type="solid"),
            "border": Border(
                left=Side(style="thin"),
                right=Side(style="thin"),
                top=Side(style="thin"),
                bottom=Side(style="thin"),
            ),
            "protection": Protection(locked=True),
        }

        self.result_style = {
            "font": Font(bold=True, size=14),
            "fill": PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid"),
            "alignment": Alignment(horizontal="center", vertical="center"),
            "border": Border(
                left=Side(style="medium"),
                right=Side(style="medium"),
                top=Side(style="medium"),
                bottom=Side(style="medium"),
            ),
        }
        self.description_style = {
            "fill": PatternFill(
                start_color="F5F5F5", end_color="F5F5F5", fill_type="solid"
            ),  # Light gray
            "border": Border(
                left=Side(style="thin"),
                right=Side(style="thin"),
                top=Side(style="thin"),
                bottom=Side(style="thin"),
            ),
            "alignment": Alignment(wrap_text=True),
            "protection": Protection(locked=True),
        }

    def _apply_cell_style(self, cell, style_dict):
        """Apply style dictionary to a cell."""
        for attr, value in style_dict.items():
            setattr(cell, attr, value)

    def _set_column_width(self, col_idx: int, width: float):
        """Set column width."""
        col_letter = get_column_letter(col_idx)
        self.ws.column_dimensions[col_letter].width = width


class ScorecardExcelExporter(ExcelModelExporter):
    """Export scorecard model to Excel with scoring formulas."""

    def __init__(self):
        super().__init__()
        # Load configurations
        self.scorecard_config = Config.scorecard_args()
        self.monitoring_config = Config.monitoring_args()
        # Store monitoring instance to reuse between calculations
        self._monitoring = None
        self._monitoring_splits = None
        self._monitoring_event_rates = None

    def _extract_scaling_parameters(self, scorecard_model) -> Dict[str, Any]:
        """Extract scaling parameters from the trained scorecard model."""
        params = {
            "scaling_method": self.scorecard_config.get("scaling_method", "min_max"),
            "scaling_method_params": self.scorecard_config.get(
                "scaling_method_params", {"min": 0, "max": 1000}
            ),
            "reverse_scorecard": self.scorecard_config.get("reverse_scorecard", False),
            "intercept_based": self.scorecard_config.get("intercept_based", False),
        }

        # Try to extract actual parameters from the model if available
        if hasattr(scorecard_model, "scorecard"):
            sc = scorecard_model.scorecard
            if hasattr(sc, "scaling_method_"):
                params["scaling_method"] = sc.scaling_method_
            if hasattr(sc, "scaling_method_params_"):
                params["scaling_method_params"] = sc.scaling_method_params_
            if hasattr(sc, "reverse_scorecard_"):
                params["reverse_scorecard"] = sc.reverse_scorecard_

        return params

    def _fit_monitoring(self, scorecard_model, X_sample, y_sample):
        """Fit monitoring instance once and cache results."""
        if (
            self._monitoring is None
            and scorecard_model is not None
            and X_sample is not None
            and y_sample is not None
        ):
            try:
                from optbinning.scorecard import ScorecardMonitoring

                # Use monitoring configuration
                psi_method = self.monitoring_config.get("psi_method", "quantile")
                psi_n_bins = self.monitoring_config.get("psi_n_bins", 5)
                psi_min_bin_size = self.monitoring_config.get("psi_min_bin_size", 0.05)

                # Handle both direct scorecard and wrapped model
                if hasattr(scorecard_model, "scorecard"):
                    # OptbinningScorecardModel wrapper
                    scorecard = scorecard_model.scorecard
                elif hasattr(scorecard_model, "table"):
                    # Direct Scorecard object
                    scorecard = scorecard_model
                else:
                    raise ValueError("Invalid scorecard model type")

                # Create and fit monitoring instance
                self._monitoring = ScorecardMonitoring(
                    scorecard=scorecard,
                    psi_method=psi_method,
                    psi_n_bins=psi_n_bins,
                    psi_min_bin_size=psi_min_bin_size,
                )
                self._monitoring.fit(X_sample, y_sample, X_sample, y_sample)

                # Cache the results
                self._monitoring_splits = list(self._monitoring.psi_splits)
                tests_df = self._monitoring.tests_table()
                self._monitoring_event_rates = tests_df["Event rate E"].values

            except Exception as e:
                raise Exception(f"Failed to fit ScorecardMonitoring: {str(e)}")

    def _parse_categorical_bin(self, bin_str: str) -> List[str]:
        """Parse a categorical bin string like \"['Male']\" or \"['Undetermined' 'Female']\" into list of values."""
        bin_str = str(bin_str).strip()

        # Handle special values
        if bin_str.lower() in ["missing", "special", "none", "nan", ""]:
            return []

        # Check if it looks like an array representation
        if bin_str.startswith("[") and bin_str.endswith("]") and "'" in bin_str:
            # Remove outer brackets
            inner = bin_str[1:-1]
            # Match quoted strings
            matches = re.findall(r"['\"]([^'\"]+)['\"]", inner)
            return matches

        # If not array format, return the string itself as single item
        return [bin_str]

    def _format_bin_display(self, bin_str: str) -> str:
        """Format bin string for user-friendly display using ≤ and ≥ notation."""
        bin_str = str(bin_str).strip()

        # Handle special cases
        if bin_str.lower() == "missing":
            return "Missing"
        elif bin_str.lower() in ["special", "none", "nan"]:
            return bin_str.capitalize()

        # Handle categorical bins (array format)
        if bin_str.startswith("[") and "'" in bin_str:
            cat_values = self._parse_categorical_bin(bin_str)
            if len(cat_values) == 1:
                return cat_values[0]
            else:
                return " / ".join(cat_values)

        # For numeric bins, use clean mathematical notation
        # Handle negative infinity cases
        if "(-inf" in bin_str or "[-inf" in bin_str:
            match = re.search(r"\(-inf,\s*(-?\d+\.?\d*)\)", bin_str)
            if match:
                return f"< {match.group(1)}"
            match = re.search(r"\(-inf,\s*(-?\d+\.?\d*)\]", bin_str)
            if match:
                return f"≤ {match.group(1)}"

        # Handle positive infinity cases
        elif "inf)" in bin_str or "inf]" in bin_str:
            match = re.search(r"\((-?\d+\.?\d*),\s*inf\)", bin_str)
            if match:
                return f"> {match.group(1)}"
            match = re.search(r"\[(-?\d+\.?\d*),\s*inf\)", bin_str)
            if match:
                return f"≥ {match.group(1)}"

        # Handle regular numeric ranges with clean notation
        range_pattern = r"([\[\(])(-?\d+\.?\d*),\s*(-?\d+\.?\d*)([\]\)])"
        match = re.match(range_pattern, bin_str)
        if match:
            start_bracket, start, end, end_bracket = match.groups()

            # Use mathematical notation: 2.5 ≤ x < 5.27
            start_symbol = "≤" if start_bracket == "[" else "<"
            end_symbol = "≤" if end_bracket == "]" else "<"

            return f"{start} {start_symbol} x {end_symbol} {end}"

        # For any other format that might contain commas, replace commas with semicolons
        if "," in bin_str:
            return bin_str.replace(",", ";")

        # For regular formats without commas, keep original
        return bin_str

    def _create_psi_lookup_sheet(self, scorecard_model, X_sample, y_sample):
        """Create a PSI lookup table sheet for score-to-probability mapping."""
        # Create new sheet
        psi_ws = self.wb.create_sheet("PSI Lookup")

        # Headers
        headers = ["Min Score", "Max Score", "Event Rate", "Risk Bin"]
        for col, header in enumerate(headers, 1):
            cell = psi_ws.cell(1, col, header)
            self._apply_cell_style(cell, self.header_style)

        # Get PSI table data
        if self._monitoring_splits is None or self._monitoring_event_rates is None:
            raise Exception("Monitoring not fitted - cannot create PSI lookup table")

        splits = self._monitoring_splits
        event_rates = self._monitoring_event_rates
        n_bins = len(event_rates)

        # Get score range from scaling params
        scaling_params = self._extract_scaling_parameters(scorecard_model)
        min_score = 0
        max_score = 1000
        if scaling_params.get("scaling_method") == "min_max":
            params = scaling_params.get("scaling_method_params", {})
            min_score = params.get("min", 0)
            max_score = params.get("max", 1000)

        # Build lookup table rows
        row = 2
        for i in range(n_bins):
            # Determine bin boundaries
            if i == 0:
                bin_min = min_score
                bin_max = splits[0] if len(splits) > 0 else max_score
            elif i == n_bins - 1:
                bin_min = splits[i - 1]
                bin_max = max_score
            else:
                bin_min = splits[i - 1]
                bin_max = splits[i]

            # Write row
            psi_ws.cell(row, 1, bin_min).number_format = "0.0"
            psi_ws.cell(row, 2, bin_max).number_format = "0.0"
            psi_ws.cell(row, 3, event_rates[i]).number_format = "0.00%"
            psi_ws.cell(row, 4, f"Bin {i+1}/{n_bins}")

            row += 1

        # Set column widths
        psi_ws.column_dimensions["A"].width = 12
        psi_ws.column_dimensions["B"].width = 12
        psi_ws.column_dimensions["C"].width = 12
        psi_ws.column_dimensions["D"].width = 15

        # Add note about VLOOKUP usage
        note_row = row + 2
        psi_ws.cell(note_row, 1, "Note:")
        psi_ws.cell(note_row, 1).font = Font(bold=True, italic=True)
        psi_ws.merge_cells(f"B{note_row}:D{note_row+1}")
        psi_ws.cell(
            note_row,
            2,
            "This table is used by VLOOKUP formulas in the main calculator sheet. "
            + "The Min Score column is used for approximate matching to find the appropriate bin.",
        )
        psi_ws.cell(note_row, 2).alignment = Alignment(wrap_text=True)
        psi_ws.cell(note_row, 2).font = Font(italic=True, size=10)

    def export_scorecard(
        self,
        scorecard_model,
        feature_names: List[str],
        X_sample,
        y_sample,
        output_file: Optional[str] = None,
        override_scorecard_df: Optional[pd.DataFrame] = None,
    ) -> io.BytesIO:
        """Export a scorecard calculator workbook with lookup tables."""
        scorecard_df = (
            override_scorecard_df
            if override_scorecard_df is not None
            else scorecard_model.get_scorecard_table()
        )
        """
        Export scorecard to Excel with formulas.

        Args:
            scorecard_model: Trained scorecard model
            feature_names: List of feature names
            X_sample: Sample data (typically training data) for monitoring calibration
            y_sample: Target data for monitoring calibration
            output_file: Optional output file path

        Returns:
            BytesIO object containing Excel file
        """
        # Get scorecard table and scaling parameters
        scorecard_table = scorecard_df
        scaling_params = self._extract_scaling_parameters(scorecard_model)

        # Fit monitoring to prepare PSI table
        self._fit_monitoring(scorecard_model, X_sample, y_sample)

        # Set worksheet name
        self.ws.title = "Scorecard Calculator"

        # Add title
        self.ws.merge_cells("A1:D1")
        title_cell = self.ws["A1"]
        title_cell.value = "Clinical Risk Scorecard Calculator"
        title_cell.font = Font(bold=True, size=16)
        title_cell.alignment = Alignment(horizontal="center", vertical="center")

        # Add instructions
        self.ws.merge_cells("A3:D4")
        inst_cell = self.ws["A3"]
        inst_cell.value = (
            "Select the appropriate bin for each feature from the dropdown in the blue cells below. "
            "The bins show the ranges or categories that apply. "
            "Select from the dropdown or leave blank if the value is not available. "
            "The scorecard will automatically calculate the points based on your selections."
            "Hover over the cells with red triangles for input descriptions"
        )
        self.ws.row_dimensions[3].height = 30
        inst_cell.alignment = Alignment(wrap_text=True, horizontal="left", vertical="top")

        # Create input section
        input_row = 6
        self.ws.cell(input_row, 1, "Feature").font = Font(bold=True)
        self.ws.cell(input_row, 2, "Select Bin").font = Font(bold=True)
        self.ws.cell(input_row, 3, "Points").font = Font(bold=True)
        self.ws.cell(input_row, 4, "Description").font = Font(bold=True)

        # Apply header style
        for col in range(1, 5):
            self._apply_cell_style(self.ws.cell(input_row, col), self.header_style)

        # Create lookup tables on a separate sheet (not hidden for debugging)
        lookup_ws = self.wb.create_sheet("Lookups")
        # Don't hide the sheet initially - we'll see if this helps with preservation

        # Add headers to lookup sheet - simplified structure
        lookup_ws.cell(1, 1, "Feature").font = Font(bold=True)
        lookup_ws.cell(1, 2, "Bin").font = Font(bold=True)
        lookup_ws.cell(1, 3, "Points").font = Font(bold=True)

        # Build feature rows and lookup tables
        current_row = input_row + 1
        lookup_row = 2  # Start after headers

        matcher = DefinitionMatcher("config/mpog_phenotypes.json", threshold=80, verbose=True)
        for feature in feature_names:
            # Get bins for this feature
            feature_bins = scorecard_table[scorecard_table["Variable"] == feature]

            if feature_bins.empty:
                continue

            # Add feature name to main sheet
            self.ws.cell(current_row, 1, feature)

            # Create dropdown list of bins
            bin_descriptions = []
            lookup_start = lookup_row

            use_display_col = "Bin_Display" in scorecard_df.columns  # or feature_bins.columns

            for _, bin_row in feature_bins.iterrows():
                points = bin_row["Points"]

                # safe fetch (Bin_Display might be list/ndarray/string/NaN/None)
                disp_val = bin_row.get("Bin_Display", None)
                use_display = (
                    use_display_col and (disp_val is not None) and (str(disp_val).strip() != "")
                )

                if use_display:
                    # Use cleaned label verbatim (no extra formatting that could mangle it)
                    display_bin = str(disp_val)
                else:
                    # Fall back to formatting the raw Bin
                    bin_str = str(bin_row["Bin"])
                    display_bin = self._format_bin_display(bin_str)

                bin_descriptions.append(display_bin)

                # Add to lookup table
                lookup_ws.cell(lookup_row, 1, feature)
                lookup_ws.cell(lookup_row, 2, display_bin)
                lookup_ws.cell(lookup_row, 3, points)
                lookup_row += 1

            # Input cell with dropdown
            input_cell = self.ws.cell(current_row, 2)
            self._apply_cell_style(input_cell, self.input_style)
            # input_style already includes Protection(locked=False)

            # Set default value to "Missing" if present
            missing_display = "Missing"
            if missing_display in bin_descriptions:
                input_cell.value = missing_display

            # Create data validation dropdown with all bins
            # Create data validation dropdown with all bins
            if bin_descriptions:
                # If any item contains a comma, or the list is long, use a range reference
                has_commas = any(("," in s) for s in bin_descriptions)
                inline_ok = (len(",".join(bin_descriptions)) < 200) and (not has_commas)

                if inline_ok:
                    # Use inline list only when safe (no commas)
                    dv = DataValidation(
                        type="list",
                        formula1='"' + ",".join(bin_descriptions).replace('"', '""') + '"',
                        allow_blank=True,
                    )
                else:
                    # Use range reference so entries like "1, 2" are a single option
                    dv = DataValidation(
                        type="list",
                        formula1=f"{quote_sheetname('Lookups')}!$B${lookup_start}:$B${lookup_row-1}",
                        allow_blank=True,
                    )

                dv.error = "Please select a bin from the dropdown"
                dv.errorTitle = "Invalid Selection"
                dv.prompt = "Please select from the list"
                dv.promptTitle = "Bin Selection"

                self.ws.add_data_validation(dv)
                dv.add(input_cell)

            # Points cell - simple lookup based on selected bin
            points_cell = self.ws.cell(current_row, 3)
            points_cell.value = (
                f'=IF(OR(B{current_row}="", ISBLANK(B{current_row})), 0, '
                f"IFERROR(VLOOKUP(B{current_row}, "
                f"Lookups!B{lookup_start}:C{lookup_row-1}, 2, FALSE), 0))"
            )
            self._apply_cell_style(points_cell, self.formula_style)

            # Below are manually set variable definitions for numerical
            # variables that did not match to phenotype list
            manual_defs = {
                "Non-Surgical Anesthesia time",
                "FFP_VolumeInMLs",
                "PRBC_TotalVolumeInMLs",
                "ColloidEquivalent_Value",
                "Crystalloids_TotalValue",
                "PreopBicarbonate_Value",
                "PreopCreatinine_Value",
                "PreopHemoglobin_Value",
                "BPFirstInRoom_BP_Dias",
                "BPFirstInRoom_BP_Sys",
            }
            # Description cell (Column 4)
            if feature in manual_defs:
                description_text = "Value range"
            else:
                description_text = matcher.get_definition(feature)
                if "[" not in description_text and "]" not in description_text:
                    description_text = "Value range"
                else:
                    description_text = re.sub(r"\s*[\r\n]+\s*", ", ", description_text).strip()
            def_cell = self.ws.cell(current_row, 4)
            def_cell.alignment = Alignment(wrap_text=True)
            self.ws.column_dimensions["D"].width = 20
            def_cell.value = (
                description_text[:40] + "..." if len(description_text) > 40 else description_text
            )
            if description_text != "Value range":
                comment = Comment(description_text, "Definition")
                char_count = len(description_text)
                line_count = description_text.count(",") + 1
                comment.width = min(400, 80 + char_count * 4)
                comment.height = min(200, 20 + line_count * 12)
                def_cell.comment = comment
            self._apply_cell_style(def_cell, self.description_style)
            self.ws.row_dimensions[current_row].height = 15

            current_row += 1

        # Add intercept if present
        intercept_rows = scorecard_table[scorecard_table["Variable"] == "Intercept"]
        if not intercept_rows.empty:
            intercept_points = intercept_rows.iloc[0]["Points"]
            self.ws.cell(current_row, 1, "Intercept (Base Score)")
            self.ws.cell(current_row, 3, intercept_points)
            self._apply_cell_style(self.ws.cell(current_row, 3), self.formula_style)
            current_row += 1

        # Add total score
        current_row += 1
        self.ws.cell(current_row, 2, "TOTAL SCORE:")
        self.ws.cell(current_row, 2).font = Font(bold=True, size=12)

        score_cell = self.ws.cell(current_row, 3)
        score_cell.value = f"=SUM(C{input_row+1}:C{current_row-2})"
        self._apply_cell_style(score_cell, self.result_style)

        # Add probability and risk category lookups
        current_row += 2
        self.ws.cell(current_row, 2, "Risk Probability:")
        self.ws.cell(current_row, 2).font = Font(bold=True)

        prob_cell = self.ws.cell(current_row, 3)
        # Use VLOOKUP to find probability from PSI table
        score_cell_ref = f"C{current_row-2}"
        prob_cell.value = f"=VLOOKUP({score_cell_ref},'PSI Lookup'!A:C,3,TRUE)"
        prob_cell.number_format = "0.00%"
        self._apply_cell_style(prob_cell, self.formula_style)

        # Add risk category
        current_row += 1
        self.ws.cell(current_row, 2, "Risk Category:")
        self.ws.cell(current_row, 2).font = Font(bold=True)

        risk_cell = self.ws.cell(current_row, 3)
        # Use VLOOKUP to find risk bin from PSI table
        risk_cell.value = f"=VLOOKUP({score_cell_ref},'PSI Lookup'!A:D,4,TRUE)"
        self._apply_cell_style(risk_cell, self.formula_style)
        # Add note to the right of the risk bin
        self.ws.cell(current_row, 4, "higher bin is more risky")
        self.ws.cell(current_row, 4).font = Font(italic=True, color="888888")

        # Insert simple summary text below results (no merging, no formulas for top 3)
        summary_row = current_row + 2
        self.ws.cell(summary_row, 2, "Summary:")
        self.ws.cell(summary_row, 2).font = Font(bold=True, italic=True)

        # Probability cell reference
        prob_cell_ref = f"C{current_row-1}"
        # Baseline event rate (from PSI Lookup table, C2)
        baseline_cell_ref = "'PSI Lookup'!C2"
        # Ratio formula
        ratio_formula = f'=IFERROR({prob_cell_ref}/{baseline_cell_ref},"")'

        # Write summary lines
        self.ws.cell(summary_row + 1, 2, "Based on the parameters entered:")
        self.ws.cell(summary_row + 2, 2, f"  - Probability of 30-day mortality:").font = Font(
            italic=True
        )
        self.ws.cell(summary_row + 2, 3, f"={prob_cell_ref}")
        self.ws.cell(summary_row + 2, 3).number_format = "0.00%"

        self.ws.cell(summary_row + 3, 2, f"  - Baseline probability:").font = Font(italic=True)
        self.ws.cell(summary_row + 3, 3, f"={baseline_cell_ref}")
        self.ws.cell(summary_row + 3, 3).number_format = "0.00%"

        self.ws.cell(summary_row + 4, 2, f"  - Times baseline:").font = Font(italic=True)
        # Round to two decimal places
        rounded_ratio_formula = f'=IFERROR(ROUND({prob_cell_ref}/{baseline_cell_ref},2),"")'
        self.ws.cell(summary_row + 4, 3, rounded_ratio_formula)
        self.ws.cell(summary_row + 4, 3).number_format = "0.00"

        self.ws.cell(
            summary_row + 6,
            2,
            "The three biggest factors in determining this score were the features with the largest points above.",
        )

        # Set column widths for main sheet
        self._set_column_width(1, 50)  # Feature names
        self._set_column_width(2, 30)  # Select bin (wider for dropdowns)
        self._set_column_width(3, 15)  # Points
        self._set_column_width(4, 40)  # Definitions (wider for readability)

        # Set column widths for Lookups sheet
        lookup_ws.column_dimensions["A"].width = 25
        lookup_ws.column_dimensions["B"].width = 30
        lookup_ws.column_dimensions["C"].width = 15

        # Now create the PSI lookup table
        self._create_psi_lookup_sheet(scorecard_model, X_sample, y_sample)

        # Protect worksheet to prevent accidental formula changes
        # Input cells are already marked as unlocked via Protection(locked=False)
        # Use simple protection approach like the old implementation
        self.ws.protection.sheet = True
        self.ws.protection.enable()

        # Save to BytesIO
        output = io.BytesIO()
        self.wb.save(output)
        output.seek(0)

        if output_file:
            with open(output_file, "wb") as f:
                f.write(output.getvalue())
            output.seek(0)  # Reset position after writing

        return output


def create_scorecard_excel(
    scorecard_model,
    feature_names: List[str],
    X_sample,
    y_sample,
    override_scorecard_df: Optional[pd.DataFrame] = None,
) -> io.BytesIO:
    """Convenience wrapper for exporting the scorecard calculator."""
    exporter = ScorecardExcelExporter()
    return exporter.export_scorecard(
        scorecard_model,
        feature_names,
        X_sample=X_sample,
        y_sample=y_sample,
        override_scorecard_df=override_scorecard_df,  # new argument
    )
