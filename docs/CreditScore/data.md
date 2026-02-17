# CreditScore.data

Data loading, preprocessing, splitting, and validation utilities.

## Classes

### `DataLoader`

Handles data loading from various file formats.

#### Methods

- `load_file(file_path, file_obj)`
  - Load data from CSV or Excel file.

Args:
    file_path: Path to the file or filename if using file_obj
    file_obj: Optional file object (for Streamlit uploads)

Returns:
    pd.DataFrame: Loaded data

### `DataPreprocessor`

Handles common data preprocessing tasks.

#### Methods

- `handle_missing_values(self, df, drop_missing)`
  - Handle missing values in the dataframe.

Args:
    df: Input dataframe
    drop_missing: Whether to drop rows with missing values

Returns:
    pd.DataFrame: Cleaned dataframe

- `prepare_modeling_data(self, df, target_column, feature_columns, drop_missing, validate_ranges, clip_negative)`
  - Prepare data for modeling by selecting columns and handling missing values.

Args:
    df: Input dataframe
    target_column: Name of the target column
    feature_columns: List of feature column names
    drop_missing: Whether to drop rows with missing values
    validate_ranges: Whether to validate data ranges
    clip_negative: Whether to clip negative values to 0 for age/duration columns

Returns:
    Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)

- `encode_categorical_features(self, X, categorical_features, method)`
  - Encode categorical features.

Args:
    X: Feature dataframe
    categorical_features: List of categorical feature names (auto-detected if None)
    method: Encoding method ('label' or 'onehot')

Returns:
    pd.DataFrame: Encoded features

- `encode_target(self, y, binary)`
  - Encode target variable.

Args:
    y: Target series
    binary: Whether this is a binary classification task

Returns:
    pd.Series: Encoded target

### `DataSplitter`

Handles train/test splitting with various strategies.

#### Methods

- `split_data(X, y, test_size, random_state, stratify)`
  - Split data into train and test sets.

Args:
    X: Feature dataframe
    y: Target series
    test_size: Proportion of data for test set
    random_state: Random seed for reproducibility
    stratify: Whether to stratify split by target variable

Returns:
    Tuple of (X_train, X_test, y_train, y_test)

### `DataPipeline`

High-level pipeline for credit scoring data preparation.

#### Methods

- `prepare_data(self, df, target_column, feature_columns, encode_categoricals, encoding_method)`
  - Complete pipeline for data preparation.

Args:
    target_column: Name of target column
    feature_columns: List of feature columns
    test_size: Test set proportion
    random_state: Random seed
    encode_categoricals: Whether to encode categorical features
    encoding_method: Method for encoding ('label' or 'onehot')
    file_obj: Optional file object for Streamlit

Returns:
    Dictionary with X_train, X_test, y_train, y_test, and metadata

### `DataValidator`

Data validation utilities.

#### Methods

- `validate_binary_target(y)`
  - Validate that target is binary.

Args:
    y: Target series or array

Returns:
    True if binary, False otherwise

- `check_data_quality(df, target_column, feature_columns)`
  - Check data quality issues.

Args:
    df: DataFrame to check
    target_column: Name of target column
    feature_columns: List of feature column names

Returns:
    Dictionary with quality check results

- `report_data_issues(issues)`
  - Create readable report from data quality issues.

Args:
    issues: Dictionary of issues from check_data_quality

Returns:
    Formatted report string
