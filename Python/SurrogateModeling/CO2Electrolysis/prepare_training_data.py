import os
import pandas as pd


def prepare_training_data(
    input_data=None,
    matlab_folder=None,
    worker_files=None,
):
    """
    Prepare training data from either file paths or a pandas dataframe.

    Parameters:
    -----------
    input_data : pd.DataFrame, optional
        A pandas DataFrame to use directly. If provided, matlab_folder and
        worker_files are ignored.
    matlab_folder : str, optional
        Path to the folder containing worker files. Default is the Matlab folder.
    worker_files : list, optional
        List of worker file names. Default is ["worker1.txt", "worker2.txt", "worker3.txt", "worker4.txt"]

    Returns:
    --------
    X, Y, combined_df : tuple of (DataFrame, DataFrame, DataFrame)
        Feature matrix, target matrix, and the full processed dataframe.
    """

    # Set defaults if not provided
    if matlab_folder is None:
        matlab_folder = (
            r"c:\Users\Philippe\Documents\UniCours\Master\Masterarbescht\MasterCode\master-thesis-orlewski\Matlab"
        )
    if worker_files is None:
        worker_files = ["worker1.txt", "worker2.txt", "worker3.txt", "worker4.txt"]

    # Load data from either input_data or from files
    if input_data is not None:
        # Use provided dataframe
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("input_data must be a pandas DataFrame")
        combined_df = input_data.copy()
        print("Data loaded from provided DataFrame")
    else:
        # Read all worker files and combine them
        dfs = []
        for worker_file in worker_files:
            file_path = os.path.join(matlab_folder, worker_file)
            df = pd.read_csv(file_path)
            dfs.append(df)

        # Combine all dataframes into one
        if len(dfs) == 0:
            print("No dataframes were loaded. Check file paths.")
            return None, None, None  # Explicit exit for empty data

        # Combine all dataframes into one
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Data loaded from {len(worker_files)} worker files")
    combined_df.to_csv(
        os.path.join(matlab_folder, "combined_data.csv"), index=False
    )  # Save combined data for reference
    original_rows = len(combined_df)

    # Drop all rows with Error > 0
    combined_df = combined_df[combined_df["Error"] == 0]
    rows_after_error_filter = len(combined_df)

    # Drop rows where FE <= 0 or FE > 1
    combined_df = combined_df[(combined_df["FE"] > 0) & (combined_df["FE"] <= 1)]
    rows_after_fe_filter = len(combined_df)

    # Drop rows where CO_out == 0
    combined_df = combined_df[combined_df["CO_out"] != 0]
    filtered_rows_co = len(combined_df)

    # Drop rows where H2_out == 0
    combined_df = combined_df[combined_df["H2_out"] != 0]
    filtered_rows_h2 = len(combined_df)

    dropped_by_error = original_rows - rows_after_error_filter
    dropped_by_fe = rows_after_error_filter - rows_after_fe_filter
    dropped_by_co_out = rows_after_fe_filter - filtered_rows_co
    dropped_by_h2_out = filtered_rows_co - filtered_rows_h2
    dropped_rows = original_rows - filtered_rows_h2

    print("\n" + "=" * 80)
    print("DATA LOADING AND FILTERING SUMMARY")
    print("=" * 80)
    print(f"Original rows:                {original_rows}")
    print(f"Rows after Error filter:      {rows_after_error_filter} (dropped {dropped_by_error})")
    print(f"Rows after FE > 0 filter:     {rows_after_fe_filter} (dropped {dropped_by_fe})")
    print(f"Rows after CO_out != 0 filter:{filtered_rows_co} (dropped {dropped_by_co_out})")
    print(f"Rows after H2_out != 0 filter:{filtered_rows_h2} (dropped {dropped_by_h2_out})")
    print(f"Total rows dropped:           {dropped_rows}")
    print(f"Average time (final):         {combined_df['time'].mean():.4f}")

    # Check if delP has the same value in every row
    unique_delP = combined_df["delP"].nunique()
    delP_values = combined_df["delP"].unique()
    print(f"\ndelP Analysis:")
    print(f"  Unique values:              {unique_delP}")
    if unique_delP == 1:
        print(f"  Value (constant):           {delP_values[0]}")
    else:
        print(f"  Values:                     {delP_values}")

    print("\n" + "=" * 80)
    print("DATAFRAME OVERVIEW")
    print("=" * 80)
    print(f"\nDataframe shape: {combined_df.shape}")
    print(f"\nColumns: {', '.join(combined_df.columns)}")
    print(f"\nFirst few rows:")
    print(combined_df.head())
    print(f"\nDataframe info:")
    print(combined_df.info())

    # Check for duplicates
    print("\n" + "=" * 80)
    print("DUPLICATE ANALYSIS")
    print("=" * 80)
    total_duplicates = combined_df.duplicated().sum()
    print(f"Total duplicate rows: {total_duplicates}")

    X = combined_df[["v", "CO2share", "E_appl"]]
    Y = combined_df[["CO2_out", "CO_out", "H2_out", "Vcell"]]
    combined_df.to_csv(os.path.join(matlab_folder, "filtered_data.csv"), index=False)
    return X, Y, combined_df
