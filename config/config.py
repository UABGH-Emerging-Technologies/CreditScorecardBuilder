"""
CreditScore_config.config
Centralised configuration for the Clinical Model Explorer application.

Usage:
    from config import Config
    data_dir = Config.DATA_DIR
    features = Config.DEFAULT_FEATURES
    binning_cfg = Config.binning_args()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Final, Optional, Set

from llm_utils.aiweb_common.configurables.data_dir import (
    get_data_dir as get_data_dir_from_submodule,
)

logger: Final = logging.getLogger(__name__)


class Config:
    """
    Central container for all configuration, directories,
    and JSON-loading utilities for the application.
    """

    # ---- Directory setup ----
    BASE_DIR: Final[Path] = Path(__file__).parent.parent.resolve()
    CONFIG_DIR: Final[Path] = BASE_DIR / "config"
    LOGS_DIR: Final[Path] = BASE_DIR / "logs"
    DEVCONTAINER_CONFIG_PATH: Final[Path] = BASE_DIR / ".devcontainer" / "devcontainer.json"

    # ---- Data location resolution ----
    _USER_DATA_DIR: Optional[Path] = None  # Optionally set this manually if needed

    @classmethod
    def get_data_dir(cls) -> Optional[Path]:
        """Resolve the best available data directory for the environment."""
        docker_path = Path("/data")
        if docker_path.is_dir():
            logger.info(f"Docker environment detected. Using {docker_path}")
            return docker_path
        if cls.DEVCONTAINER_CONFIG_PATH.exists():
            try:
                with open(cls.DEVCONTAINER_CONFIG_PATH, "r") as f:
                    devcontainer_config = json.load(f)
                    for mount in devcontainer_config.get("mounts", []):
                        if mount.get("target") == "/data":
                            host = Path(mount["source"])
                            if host.is_dir():
                                logger.info(f"Using devcontainer host data dir: {host}")
                                return host
            except Exception as e:
                logger.error(f"Failed parsing {cls.DEVCONTAINER_CONFIG_PATH}: {e}")
        if cls._USER_DATA_DIR and cls._USER_DATA_DIR.is_dir():
            logger.info(f"Using manual user data dir: {cls._USER_DATA_DIR}")
            return cls._USER_DATA_DIR
        logger.warning("Data directory not found. Please specify Config._USER_DATA_DIR.")
        return None

    # These can be re-evaluated at runtime if ._USER_DATA_DIR changes
    @classmethod
    def data_dir(cls) -> Optional[Path]:
        """Return the resolved data directory (if any)."""
        return cls.get_data_dir()

    @classmethod
    def raw_data(cls) -> Optional[Path]:
        """Return the raw data subdirectory path."""
        data = cls.data_dir()
        return data / "raw" if data else None

    @classmethod
    def intermediate_dir(cls) -> Optional[Path]:
        """Return the intermediate data subdirectory path."""
        data = cls.data_dir()
        return data / "intermediate" if data else None

    @classmethod
    def results_dir(cls) -> Optional[Path]:
        """Return the results subdirectory path."""
        data = cls.data_dir()
        return data / "results" if data else None

    @classmethod
    def example_output(cls) -> Optional[Path]:
        """Return the example output file path, if available."""
        interm = cls.intermediate_dir()
        return interm / "Example_Output.csv" if interm else None

    # Backward compatibility (initialized after class definition)
    DATA_DIR: Optional[Path] = None
    RAW_DATA: Optional[Path] = None
    INTERMEDIATE_DIR: Optional[Path] = None
    RESULTS_DIR: Optional[Path] = None
    EXAMPLE_OUTPUT: Optional[Path] = None

    # ---- Default thresholds & features ----
    UNIQUE_CATEGORY_THRESHOLD: Final[int] = 6
    DEFAULT_FEATURES: Final[list[str]] = [
        "Gender",
        "Age",
        "BMI",
        # 'Race', 'Ethnicity',
        "AlbuminLevel",
        "EBL",
        "SugammadexAmount",
        "NeostigmineAmount",
        # 'RBCs', 'WholeBlood',
        "FFP",
        "Cryo",
        "Platelets",
        "CellSaver",
        "EpinephrineInfusion",
        "VasopressinInfusion",
        "MilrinoneInfusion",
        "DobutamineInfusion",
        "DopamineInfusion",
        "NorepinephrineInfusion",
        "PhenylephrineInfusion",
        "BloodProductsTotal",
        "CrystalloidsTotal",
        "EMERGENCY",
        "CVL",
        "ArtLine",
        "PreOpBicarb",
        "PreOpHemoglobin",
        "PreOpPlatelets",
        "ASAStatus",
        "anes_surgical_duration_diff",
    ]

    # ---- Streamlit headers ----
    LR_HEADER_MARKDOWN: Final[str] = """
    ---
    **Clinical Model Explorer – internal research tool**
    _Use of real patient data may require additional IRB / compliance approval.
    Contact the project maintainer for details._
    All uploads remain local to the browser – no data leave your machine.

    ---
    """
    SC_HEADER_MARKDOWN: Final[str] = """
    This application builds clinical risk scorecards using automated binning and machine learning.
    #### Credit Score Model-Type Features
    - Automatic binning of numerical and categorical features
    - Elastic-net regularized logistic regression
    - Scorecard scaling (0–1000)
    - Comprehensive visual analytics
    >ℹ️ Use the sidebar (click the arrow at top left) to toggle visibility of scorecard tables and diagnostic plots.
    ---
    """

    # ---- JSON configuration utilities ----
    _COMMENT_PREFIX: Final[str] = "//"

    @classmethod
    def _load_json_cfg(cls, stem: str) -> Dict[str, Any]:
        """Load a JSON config file from the config directory."""
        path: Path = cls.CONFIG_DIR / f"{stem}.json"
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        # Remove JSON comments (keys starting with //)
        cleaned = {k: v for k, v in raw_config.items() if not k.startswith(cls._COMMENT_PREFIX)}
        logger.info("Loaded configuration %s", path.name)
        return cleaned

    # ---- Public API for JSON configs ----
    @classmethod
    def binning_args(cls) -> Dict[str, Any]:
        """Get binning configuration optimized for medical data with class imbalance."""
        return cls._load_json_cfg("binning_args")

    @classmethod
    def scorecard_args(cls) -> Dict[str, Any]:
        """Get scorecard configuration for model initialization."""
        return cls._load_json_cfg("scorecard_args")

    @classmethod
    def cross_validation_args(cls) -> Dict[str, Any]:
        """Get cross-validation configuration for model fitting."""
        return cls._load_json_cfg("cross_validation_args")

    @classmethod
    def monitoring_args(cls) -> Dict[str, Any]:
        """Get monitoring configuration."""
        allowed: Set[str] = {"psi_method", "psi_n_bins", "psi_min_bin_size", "show_digits"}
        raw = cls._load_json_cfg("monitoring_args")
        return {k: v for k, v in raw.items() if k in allowed}

    @classmethod
    def variable_selection_args(cls) -> Dict[str, Any]:
        """Get variable selection configuration."""
        return cls._load_json_cfg("variable_selection_args")

    @classmethod
    def power_analysis_args(cls) -> Dict[str, Any]:
        """Get power analysis configuration."""
        return cls._load_json_cfg("power_analysis_args")

    # ---- Convenience methods for "fit configs" etc ----
    @classmethod
    def get_scorecard_fit_config(cls) -> Dict[str, Any]:
        """Get cross-validation configuration for model fitting, with legacy/extracted keys."""
        config = cls.cross_validation_args()
        # Extract fit-specific parameters
        return {
            "hyperparameter_search": config.get("hyperparameter_search", True),
            "param_grid": config.get(
                "param_grid", {"C": [0.01, 0.1, 1, 10], "l1_ratio": [0.1, 0.5, 0.9]}
            ),
            "cv": config.get("cv", 5),
        }

    @classmethod
    def get_power_analysis_args_strict(cls) -> Dict[str, Any]:
        """
        Extract backward-compatible legacy format for power analysis.
        """
        config = cls.power_analysis_args()
        return {
            "alpha": config.get("default_alpha", 0.05),
            "power": config.get("default_power", 0.80),
            "min_sample_size": config.get("simulation", {}).get("min_sample_size", 50),
            "max_sample_size": config.get("simulation", {}).get("max_sample_size", 500),
            "step_size": config.get("simulation", {}).get("step_size", 50),
            "n_simulations": config.get("simulation", {}).get("n_simulations", 1000),
        }


Config.DATA_DIR = Config.get_data_dir()
Config.RAW_DATA = Config.DATA_DIR / "raw" if Config.DATA_DIR else None
Config.INTERMEDIATE_DIR = Config.DATA_DIR / "intermediate" if Config.DATA_DIR else None
Config.RESULTS_DIR = Config.DATA_DIR / "results" if Config.DATA_DIR else None
Config.EXAMPLE_OUTPUT = (
    Config.INTERMEDIATE_DIR / "Example_Output.csv" if Config.INTERMEDIATE_DIR else None
)

default_target = "HospitalMortality30Day_Value_CD"

default_features = [
    "AgeInYears_Value",
    "AnesthesiaDuration_Value",
    "AsaStatusClassification_Value_Code",
    "PrimaryAnesthesiaCPT_MPOGbaseUnits",  # derived from cpt
    "AdmissionType_Value_Code",
    "ComplicationAHRQPulmonaryAll_Value_Code",
    "ComplicationMpogAcuteKidneyInjury_Value",
    "PONVReportedClassification_Value_Code",
    "PostopTroponinHighest_Value",
    "PreopTroponinHighest_Value",
    "BodyMassIndex_Value",
    "Race_Value_Code",
    "Sex_Value_Code",
    "CardioPulmonaryBypassDuration_Value",
    "SurgeryDuration_Value",
    "Non-Surgical Anesthesia time",
    "Cryo_PrimaryDocumentationCode",
    "ArterialLinePlaced_Value_Code",
    "FFP_VolumeInMLs",
    "Platelets_VolumeInMLs",
    "PRBC_TotalVolumeInMLs",
    "ColloidEquivalent_Value",
    "Crystalloids_TotalValue",
    "PreopBicarbonate_Value",
    "PreopCO2Arterial_Value",
    "PreopCO2MixedVenous_Value",
    "PreopCO2Venous_Value",
    "PreopEGFR60DayLowest_Value",
    "PreopEGFR60DayMostRecent_Value",
    "SurgicalServiceGroup_Value_Code",
    "EstimatedBloodLoss_Value",
    "UnitsTransfused_Value",
    "UrineOutput_Value",
    "PreopAlbumin_Days_Before",
    "PreopAlbumin_Value",
    "PreopAlkPhosphatase_Value",
    "PreopALT_Value",
    "PreopArterialpH_Value",
    "PreopAST_Value",
    "PreopBUN_Value",
    "PreopCalciumTotal_Value",
    "PreopChloride_Value",
    "PreopCreatinine_Value",
    "SmokingTobaccoClassification_Value_Code",
    "PreopHCG_Value",
    "PreopHematocrit_Value",
    "PreopHemoglobin_Value",
    "PreopINR_Value",
    "PreopHgbA1c_Value",
    "PreopLactate_Value",
    "PreopPlatelets_Value",
    "PreopPotassium_Value",
    "PreopProtein_Value",
    "PreopPT_Value",
    "PreopPTT_Value",
    "PreopSodium_Value",
    "PreopTotalBilirubin_Value",
    "PreopTroponinMostRecent_Value",
    "PreopWBC_Value",
    "BPFirstInRoom_BP_Dias",
    "BPFirstInRoom_BP_Sys",
    "Hypoxemia_Value",
    "TidalVolumeActualMedian_Median",
    "AnesthesiaTechniqueNeuraxial_Value",
    "OpioidsGiven_Intraop_Opioids",
    "OralMorphineEquivalentNormalized_Value",
    "ProcedureTypeHipArthroplasty_Value",
]
