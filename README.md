# Credit Scorecard Builder

Credit Scorecard Builder is a Streamlit application for building and exploring clinical risk scorecards. It accepts CSV and Excel data, trains an automatically binned elastic-net logistic regression model, displays model diagnostics, and exports a report with scorecard artifacts.

## Quickstart with uv

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/) for Python installation, dependency locking, and command execution. Install Git and uv first, then run:

```bash
git clone https://github.com/UABGH-Emerging-Technologies/CreditScorecardBuilder.git
cd CreditScorecardBuilder
uv sync --locked
uv run streamlit run UserInterface/credit_score_app.py
```

`uv sync --locked` creates `.venv`, installs the Python version recorded in `.python-version` if necessary, and installs the exact dependencies in `uv.lock`. Open the local URL printed by Streamlit (normally <http://localhost:8501>).

The equivalent convenience commands are `make sync` and `make run`.

## Create and download a scorecard

Prepare a `.csv` or `.xlsx` dataset with one row per observation. It must contain a binary outcome column and the candidate predictor columns. Uploaded data is processed by the Streamlit server running locally in your environment. Do not upload identifiable patient data unless your local use is covered by the appropriate institutional approvals.

In the application:

1. Stay on the **Clinical Risk Scorecard** tab and upload the dataset.
2. Select the outcome/target column.
3. Choose a feature category (**All**, **Preop**, **Intraop**, or **Postop**) and select the predictor columns. The target must not be included as a predictor.
4. In the sidebar, choose the thresholding method and the plots, tables, data preview, and variable-selection report you want to see or export.
5. Select **Train / Re-train model** and wait for training to finish. The page will show train/test AUCs and the selected diagnostics.
6. Select **Build full report ZIP**, then **Download full report**.

The downloaded `clinical_scorecard_report.zip` contains a Word report, the selected tables as CSV files, the selected plots as PNG files, `basic_scorecard.xlsx`, and `interactive_calculator.xlsx`. If enabled in the sidebar, it also contains the variable-selection report.

## Dev Container option

The dev container is optional; uv remains the dependency manager inside it. It runs the official Astral uv image for Python 3.12, executes `uv sync --locked` after creation, and forwards Streamlit port 8501.

Prerequisites are Docker and VS Code with the **Dev Containers** extension.

```bash
git clone https://github.com/UABGH-Emerging-Technologies/CreditScorecardBuilder.git
cd CreditScorecardBuilder
code .
```

In VS Code, run **Dev Containers: Reopen in Container** from the Command Palette. When setup completes, use the integrated terminal:

```bash
uv run streamlit run UserInterface/credit_score_app.py --server.address 0.0.0.0
```

Open the forwarded **Streamlit UI** port when prompted. Upload CSV or Excel data through the browser just as in the uv-only setup.

## Development commands

```bash
make test       # run pytest in the locked environment
make style      # run Black, isort, and Flake8
make docs       # serve the MkDocs site locally
make lock       # intentionally update uv.lock after dependency changes
```

Run `uv sync --locked` after pulling a lockfile change. When changing dependencies, edit `pyproject.toml`, run `uv lock`, and commit both files.
