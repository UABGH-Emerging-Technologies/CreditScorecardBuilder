"""
Light-weight, object-oriented helper that collects artifacts
(DataFrames, matplotlib figures, bytes, existing files …),
writes them to a temporary directory and finally zips everything
into a BytesIO object for easy download (e.g. from Streamlit).

Usage
-----
from CreditScore.report_builder import ReportBuilder

with ReportBuilder() as rb:
    rb.add_dataframe(coef_df,     "coefficients.csv")
    rb.add_dataframe(power_df,    "power_analysis.csv")
    rb.add_figure(fig,            "power_curve.png")
    rb.add_file("some/path/note.txt")         # existing file
    rb.add_bytes(pdf_bytes,       "report.pdf")

zip_bytes = rb.build_zip()  # ready for st.download_button
"""

import tempfile
import types
import zipfile
from contextlib import AbstractContextManager
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from aiweb_common.file_operations.docx_creator import StreamlitDocxCreator
from docx import Document
from docx.shared import Inches

from CreditScore.utils import clean_add_results_to_docx


class SafeStreamlitDocxCreator(StreamlitDocxCreator):
    """Docx creator with preserved summary/results bindings."""

    def __init__(self, summary, results, figures, variable_selection_text=None):
        super().__init__(summary, results, figures)
        self.summary = summary  # manually assign the third value
        self.results = results  # reassign to guard against reassignment
        self.figures = figures
        self.variable_selection_text = variable_selection_text


def patched_create_docx_report(self):
    """Build a DOCX report with summary, tables, and figures."""
    doc = Document()
    section = doc.sections[0]
    section.left_margin = Inches(0.5)
    section.right_margin = Inches(0.5)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)

    doc.add_heading("Model Evaluation Report", 0)

    if self.summary:
        doc.add_heading("Summary", level=1)
        for line in self.summary.splitlines():
            doc.add_paragraph(line.strip())

    # Original results logic
    if self.results:
        self._add_results_to_docx(doc, self.results)

    if self.variable_selection_text:
        doc.add_heading("Variable Selection Report", level=1)
        for line in self.variable_selection_text.splitlines():
            doc.add_paragraph(line)

    # 👇 Injected figure logic
    if self.figures:
        doc.add_heading("Figures", level=1)
        for name, fig in self.figures.items():
            if isinstance(fig, plt.Figure):
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                doc.add_paragraph(f"Figure: {name}")
                doc.add_picture(buf, width=Inches(6))
                plt.close(fig)

    return doc


def compile_report_bytes(summary: str, results: dict, figures: dict, variable_selection_text=None):
    """Create a DOCX report and return it as a BytesIO stream."""
    # ------------------------------------------------------------------ #
    # 💡  VERBOSE DEBUGGING
    # ------------------------------------------------------------------ #
    import numbers
    import textwrap

    import pandas as pd

    def _describe(obj, indent=0):
        pad = " " * indent
        if isinstance(obj, pd.DataFrame):
            print(f"{pad}↳ DataFrame shape={obj.shape}")
        elif isinstance(obj, pd.Series):
            print(f"{pad}↳ Series   len={len(obj)}  name={obj.name!r}")
        elif isinstance(obj, (list, tuple)):
            print(f"{pad}↳ {type(obj).__name__}  len={len(obj)}")
        elif isinstance(obj, dict):
            print(f"{pad}↳ dict keys={list(obj.keys())}")
        elif isinstance(obj, (str, bytes, numbers.Number)):
            preview = textwrap.shorten(str(obj), width=60, placeholder=" …")
            print(f"{pad}↳ {type(obj).__name__}  value={preview!r}")
        else:
            print(f"{pad}↳ {type(obj)}")

    print("\n" + "=" * 60 + "\nDEBUG :: compile_report_bytes\n" + "=" * 60)

    print("• summary :", end=" ")
    _describe(summary)

    print("• results :", end=" ")
    _describe(results)
    if isinstance(results, dict):
        for meth, sections in results.items():
            print(f"  └─ method {meth!r} :", end=" ")
            _describe(sections, indent=4)
            if isinstance(sections, dict):
                for sec, data in sections.items():
                    print(f"      └─ section {sec!r} :", end=" ")
                    _describe(data, indent=8)

    print("• figures :", end=" ")
    _describe(figures)
    if isinstance(figures, dict):
        for name, fig in figures.items():
            print(f"  └─ figure {name!r} :", end=" ")
            _describe(fig, indent=4)

    print("=" * 60 + "\n")

    SafeStreamlitDocxCreator.create_docx_report = patched_create_docx_report
    SafeStreamlitDocxCreator._add_results_to_docx = types.MethodType(
        clean_add_results_to_docx, SafeStreamlitDocxCreator
    )
    creator = SafeStreamlitDocxCreator(
        summary, results, figures, variable_selection_text=variable_selection_text
    )
    creator.create_docx_report = types.MethodType(patched_create_docx_report, creator)
    doc = creator.create_docx_report()
    docx_buffer = BytesIO()
    doc.save(docx_buffer)
    docx_buffer.seek(0)
    return docx_buffer


class ReportBuilder(AbstractContextManager):
    """Collect artifacts → produce in-memory ZIP."""

    _FIG_EXTS = {"png", "jpg", "jpeg", "svg", "gif", "tif", "tiff", "bmp", "webp"}
    _CSV_EXTS = {"csv"}  # easy to extend later

    def __init__(self, tmp_prefix: str = "report_", auto_cleanup: bool = True):
        self._tmpdir_ctx = tempfile.TemporaryDirectory(prefix=tmp_prefix)
        self.tmp_path = Path(self._tmpdir_ctx.name)  # Real Path
        self.auto_cleanup = auto_cleanup
        self._closed = False

    # ------------------------------------------------------------------ helpers
    def _check_closed(self):
        """Raise if the builder has already been closed."""
        if self._closed:
            raise RuntimeError("ReportBuilder is already closed.")

    def _add_readme(self):
        """Copy config/Report_README.md into temp dir root if present."""
        readme_path = Path("config/Report_README.md")
        if readme_path.exists():
            dest = self.tmp_path / "Report_README.md"
            dest.write_bytes(readme_path.read_bytes())

    def _safe_path(self, filename: str) -> Path:
        """
        Decide where a file should live (figures/, data/, or root) and return a
        full path *inside* the temp directory.  The needed sub-folders are
        created on the fly.
        """
        name = Path(filename).name  # strip any upstream path
        ext = name.split(".")[-1].lower()

        if ext in self._CSV_EXTS:
            subdir = "data"
        elif ext in self._FIG_EXTS:
            subdir = "figures"
        else:
            subdir = ""  # everything else in root

        target = self.tmp_path / subdir / name if subdir else self.tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    # ------------------------------------------------------------------ adders
    def add_dataframe(self, df: pd.DataFrame, filename: str):
        """Save DataFrame as CSV inside the bundle."""
        self._check_closed()
        df.to_csv(self._safe_path(filename), index=False, float_format="%.4f")

    def add_figure(self, fig, filename: str, dpi: int = 300):
        """Save a matplotlib figure into the report bundle."""
        print("adding (only matplotlib, for now) figure to report")
        self._check_closed()
        path = self._safe_path(filename)
        # This only works for Matplotlib
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        except BaseException:
            raise TypeError(f"Unsupported figure type: {type(fig)}")

    def add_bytes(self, data: Union[bytes, BytesIO], filename: str):
        """Write raw bytes or a BytesIO stream into the bundle."""
        self._check_closed()
        with open(self._safe_path(filename), "wb") as f:
            if isinstance(data, BytesIO):  # already a stream
                print("adding BytesIO stream to report")
                data.seek(0)
                f.write(data.read())
            elif isinstance(data, bytes):  # raw bytes
                print("adding bytes to report")
                f.write(data)
            else:
                raise TypeError(f"Unsupported data type: {type(data)}")

    def add_file(self, file_path: Union[str, Path], filename: Optional[str] = None):
        """Copy an existing file into the bundle."""
        self._check_closed()
        src = Path(file_path)
        if not src.exists():
            raise FileNotFoundError(src)
        dest_name = filename or src.name
        dest = self._safe_path(dest_name)
        dest.write_bytes(src.read_bytes())

    # ------------------------------------------------------------------ build
    def build_zip(self, zip_name: str = "report.zip") -> BytesIO:
        """Create an in-memory ZIP and return it as BytesIO."""
        self._check_closed()
        self._add_readme()
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in self.tmp_path.rglob("*"):
                if file.is_file():
                    # keep relative path so folders appear in the archive
                    zf.write(file, arcname=file.relative_to(self.tmp_path))
        zip_buffer.seek(0)
        return zip_buffer

    # ------------------------------------------------------------------ context
    def close(self):
        """Close and optionally clean up the temp directory."""
        if not self._closed and self.auto_cleanup:
            self._tmpdir_ctx.cleanup()
        self._closed = True

    def __exit__(self, exc_type, exc, tb):
        self.close()
        # propagate exception (if any)
        return False
