# Credit Scorecard Builder

Tools for building and exploring clinical risk scorecards. The main entry point
is the Streamlit UI in `UserInterface/credit_score_app.py`.

## Quickstart

### 1) Create an environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies
From the repository root (`CreditScorecardBuilder`):
```bash
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```
`requirements.txt` installs the local `llm_utils` package via `./llm_utils`.

### 3) Configure data directory
The app looks for data in `/data` (Docker) or via a devcontainer mount. If
running locally, set a path explicitly before launch by editing
`config/config.py` and setting `Config._USER_DATA_DIR` to a valid directory.

### 4) Run the Streamlit app
```bash
streamlit run UserInterface/credit_score_app.py
```

## Documentation

### Preview locally
```bash
mkdocs serve
```

### Publish to GitHub Pages
```bash
mkdocs gh-deploy
```
Set `site_url` in `mkdocs.yml` to your GitHub Pages URL:
`https://<github-username>.github.io/<repo-name>/`.
