# preprocess_add_outcomes_corrected.py
"""
Preprocess the AdditionalOutcomesANDWhileUnderAnesCare extract for scorecard modeling.

Corrected form of preprocess_add_outcomes.py. Every output goes to its own timestamped
directory under data/intermediate/, so nothing existing is overwritten.
"""

import json
from pathlib import Path

import pandas as pd

from CreditScore.data import DataLoader, DataPreprocessor, DataSplitter, DataValidator

RAW = Path("/home/agryshyna/raw/AdditionalOutcomesANDWhileUnderAnesCare.xlsx")
OUT = Path(__file__).resolve().parents[1] / "data" / "intermediate" / "add_outcomes_20260901"

# Known only once the encounter is over, or identifier-like with too many levels to bin.
# Same list postpacu/config/datasets.py drops for this extract.
LEAKAGE_AND_ID = [
    "AdmitDateTime",
    "DischargeDateTime",
    "DeceasedDateTime",
    "DeceasedDuringEncounter",
    "LOS_Hours",
    "DeathDaysAfterAnesEnd",
    "Subsequent30DayProcedures",
    "ZipCode",
    "PRIMARYSURGEON",
    "SurgeonSpecialtyByName",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}

    # Step 1 -- load
    df = DataLoader.load_file(RAW)
    print(f"loaded {df.shape[0]:,} rows x {df.shape[1]} cols")
    summary["raw_shape"] = list(df.shape)

    # Step 1b -- restrict to the inpatient cohort. The workbook holds every case; the
    # published analysis models inpatients only. Without this the denominator is wrong
    # and the escalation rate comes out about half what it should be.
    df = df.loc[df["InpatientStat"] == 1].copy()
    print(f"inpatient cohort: {df.shape[0]:,} rows")
    summary["inpatient_rows"] = int(df.shape[0])

    # Step 2 -- build the outcome. It is not in the raw extract, so it has to be derived.
    # Same definition as postpacu/data.py, so the two projects stay comparable.
    df["escalation"] = (
        (df["MET_Team"] == 1)
        | ((df["ICUafterPACU_Days"] == 1) & (df["ICU_Bed_Order"] == 0))
        | (df["ICU_AfterStepDown_NoOrderBeforePacuDepart_Days"] == 1)
        | df["StepDownUnitAfterGeneralCareTime_Days"].notna()
    )
    target_column = "escalation"
    counts = df[target_column].value_counts()
    rate = df[target_column].mean()
    print(f"escalation: {int(counts.get(True, 0)):,} positives ({rate:.4%})")
    summary["escalation_positives"] = int(counts.get(True, 0))
    summary["escalation_rate"] = float(rate)

    # Step 3 -- select by exclusion rather than by position, so a reordered extract
    # cannot silently change the feature set.
    feature_columns = [
        c for c in df.columns[21:] if c not in LEAKAGE_AND_ID and c != target_column
    ]
    print(f"features: {len(feature_columns)}")
    summary["n_features"] = len(feature_columns)
    summary["dropped_leakage_and_id"] = [c for c in LEAKAGE_AND_ID if c in df.columns]

    # Step 4 -- data quality report
    validator = DataValidator()
    issues = validator.check_data_quality(df, target_column, feature_columns)
    report = validator.report_data_issues(issues)
    print(report)
    (OUT / "data_quality_report.txt").write_text(report)

    # Step 5 -- keep NaN. Missingness is informative in this data; the binner should
    # handle it as its own category rather than dropping or imputing here.
    pre = DataPreprocessor()
    X, y = pre.prepare_modeling_data(
        df,
        target_column=target_column,
        feature_columns=feature_columns,
        drop_missing=False,
        validate_ranges=True,
        clip_negative=False,
    )
    y = pre.encode_target(y, binary=True)
    summary["X_shape"] = list(X.shape)

    # Record missingness before any encoding, for the binning step later.
    missing = (
        X.isna().sum().rename("n_missing").to_frame().assign(pct=lambda d: d.n_missing / len(X) * 100)
    )
    missing.sort_values("n_missing", ascending=False).to_csv(OUT / "missingness.csv")

    # Step 6 -- split BEFORE fitting any encoder, so test categories never inform the fit.
    X_train, X_test, y_train, y_test = DataSplitter.split_data(
        X, y, test_size=0.2, random_state=42, stratify=True
    )
    print(f"train {X_train.shape}  test {X_test.shape}")
    summary["train_shape"] = list(X_train.shape)
    summary["test_shape"] = list(X_test.shape)

    # Flag object columns holding more than one python type. These are columns that look
    # numeric but carry stray free text (MACE_Score has 'per Cardiology' in it), which is
    # what breaks both LabelEncoder and parquet. Worth cleaning before binning.
    mixed = {}
    for col in X.columns:
        if X[col].dtype == object:
            types = sorted({type(v).__name__ for v in X[col].dropna().unique()})
            if len(types) > 1:
                nonnum = [
                    v for v in X[col].dropna().unique() if not isinstance(v, (int, float))
                ]
                mixed[col] = {"types": types, "n_non_numeric_values": len(nonnum),
                              "examples": [str(v) for v in nonnum[:8]]}
    if mixed:
        print("\nmixed-type object columns (clean these before binning):")
        for col, info in mixed.items():
            print(f"  {col}: {info['types']} -- e.g. {info['examples'][:4]}")
    (OUT / "mixed_type_columns.json").write_text(json.dumps(mixed, indent=2))
    summary["mixed_type_columns"] = sorted(mixed)

    # Save the unencoded frames first -- these are what a WOE/optbinning step wants, with
    # real NaN intact, and writing them here means an encoding failure cannot discard the
    # expensive Excel load. CSV rather than parquet: the mixed-type columns above have no
    # single Arrow type, so parquet refuses them.
    X_train.to_csv(OUT / "X_train_raw.csv.gz", index=False, compression="gzip")
    X_test.to_csv(OUT / "X_test_raw.csv.gz", index=False, compression="gzip")
    y_train.to_frame("escalation").to_csv(OUT / "y_train.csv", index=False)
    y_test.to_frame("escalation").to_csv(OUT / "y_test.csv", index=False)

    # Step 7 -- LabelEncoder rejects a column holding both str and NaN ("input argument
    # must be uniformly strings or numbers"). Rather than drop or impute those rows, give
    # missing its own level: it carries signal in this data, and keeping it as a category
    # is what the binning step will want anyway. Numeric columns keep real NaN.
    MISSING_LEVEL = "__MISSING__"

    def fill_object_na(frame):
        # Cast to str as well as filling: several of these columns mix numbers and text
        # in the same object column, and LabelEncoder rejects that just as it rejects NaN.
        out = frame.copy()
        for col in out.select_dtypes(include=["object", "string"]).columns:
            out[col] = out[col].where(out[col].notna(), MISSING_LEVEL).astype(str)
        return out

    X_train_fill = fill_object_na(X_train)
    X_test_fill = fill_object_na(X_test)

    # Fit encoders on train, then reuse for test. DataPreprocessor caches fitted encoders
    # in self.label_encoders and maps unseen categories to -1 on the second call.
    X_train_enc = pre.encode_categorical_features(X_train_fill, method="label")
    X_test_enc = pre.encode_categorical_features(X_test_fill, method="label")
    summary["n_label_encoded"] = len(pre.label_encoders)
    summary["missing_level"] = MISSING_LEVEL

    # Save the encoded frames alongside the raw ones.
    X_train_enc.to_parquet(OUT / "X_train_encoded.parquet")
    X_test_enc.to_parquet(OUT / "X_test_encoded.parquet")
    (OUT / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nwrote outputs to {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.stat().st_size:>12,}  {p.name}")


if __name__ == "__main__":
    main()
