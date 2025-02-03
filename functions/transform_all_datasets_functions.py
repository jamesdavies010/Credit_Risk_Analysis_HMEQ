# 12 - transform_anomalies
# 40 - drop_irrelevant_columns
# 74 - create custom bins
# 236 - drop_columns_after_WoE
# 281 - one_hot_encode_bins


import numpy as np
import pandas as pd
from typing import List, Union

def transform_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms anomalies in a given DataFrame by setting the following conditions:
    - LTV RATIO > 1.15: PROPERTY VALUE and LTV RATIO set to -1
    - OLDEST CREDIT (MONTHS) > 720: OLDEST CREDIT (MONTHS) set to 720
    
    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame on which the anomalies will be transformed.

    Returns:
    -------
    pd.DataFrame
        A new DataFrame with the anomalies transformed (NaN values set where applicable).
    """
    df.loc[
        df["LTV RATIO"] > 1.15,
        ["PROPERTY VALUE", "LTV RATIO"]
    ] = -1  # Values set to -1. Avoid NaN because NaN can be a powerful predictor...
    
    df.loc[
        df["OLDEST CREDIT (MONTHS)"] > 720, 
        "OLDEST CREDIT (MONTHS)"
    ] = 720


def drop_irrelevant_columns(
    df: pd.DataFrame, 
    columns_to_drop: list = ['LTV RATIO', 'CREDIT LINES', 'JOB', 'JOB_encoded', 'OUTSTANDING MORTGAGE']
) -> pd.DataFrame:
    """
    Drops specified columns from a DataFrame, with a default list of columns to drop.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame from which columns will be dropped.
    columns_to_drop : list, optional
        A list of column names to drop. Defaults to list provided.

    Returns:
    -------
    pd.DataFrame
        A new DataFrame with the specified columns removed.
    """
    columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]
    
    new_df = df.drop(columns=columns_to_drop_existing, inplace=False)
    
    if columns_to_drop_existing:
        print(f"Dropped columns: {', '.join(columns_to_drop_existing)}")
    else:
        print("No columns were dropped.")
    
    return new_df


import numpy as np
import pandas as pd
from typing import Union, List

def create_custom_bins(
    df: pd.DataFrame,
    columns: Union[List[str], str] = "All",
    target_column: str = "DEFAULT",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply custom binning rules to specified columns in the DataFrame.
    If no columns are specified, all columns except the target are binned using default rules.
    
    Parameters
    ----------
    df : pd.DataFrame
        Your dataset.
    columns : List[str] or str, default="All"
        Which columns to bin. If "All", we bin every column except `target_column`.
    target_column : str, default="DEFAULT"
        Column to exclude from binning. By default, "DEFAULT".
    
    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        1) The original DataFrame with new binned columns (e.g. "<col>_bin").
        2) A summary DataFrame describing the bin edges used for each column.
    """

    bin_values = {
        "PROPERTY VALUE": [-1, 0, 50000, 80000, 110000, np.inf],
        "YEARS AT JOB": [0, 5, 10, 15, 20, np.inf],
        "DEROGATORY REPORTS": [0, 1, 2, np.inf],
        "DELINQUENT CREDIT LINES": [0, 1, 2, np.inf],
        "OLDEST CREDIT (MONTHS)": [0, 120, 180, 240, np.inf],
        "RECENT CREDIT ENQUIRIES": [0, 1, 2, 3, np.inf],
        "DEBTINC": [0, 20, 30, 40, 45, np.inf],
    }

    bin_summary_records = []

    def custom_binning(df: pd.DataFrame, column: str) -> pd.Series:
        """
        Apply custom binning rules based on the column name and return the new binned Series.
        """
        if column in bin_values:
            bins = bin_values[column]
            # borrowers with an LTV ratio > 1.15 will be put in bin -1
            if column == "PROPERTY VALUE":
                labels = [-1, 1, 2, 3, 4]
            else:
                labels = list(range(1, len(bins)))

            df_with_bins = pd.cut(
                df[column],
                bins=bins,
                labels=labels,
                include_lowest=True
            )

            # For each adjacent pair of bins, store a row describing that interval
            for i, (label, (lower, upper)) in enumerate(
                zip(labels, zip(bins, bins[1:]))
            ):
                left_bracket = "[" if i == 0 else "("
                lower_str = "-inf" if lower == -np.inf else str(lower)
                upper_str = "inf" if upper == np.inf else str(upper)
                interval_str = f"{left_bracket}{lower_str}, {upper_str}]"

                bin_summary_records.append(
                    {
                        "Column": column,
                        "Bin Label": label,
                        "Lower Bound": lower,
                        "Upper Bound": upper,
                        "Bin Interval": interval_str,
                    }
                )

        else:
            # For columns not in bin_values, use qcut with 5 quantiles
            df_with_bins = pd.qcut(
                df[column],
                q=5,
                duplicates="drop",
                precision=2,
                labels=False
            )
            df_with_bins = df_with_bins + 1

            # missing values go in bin 0
            df_with_bins = df_with_bins.fillna(0).astype(int)

            if hasattr(df_with_bins, "cat") and hasattr(df_with_bins.cat, "categories"):
                intervals = df_with_bins.cat.categories
                for i, interval in enumerate(intervals, start=1):
                    bin_summary_records.append(
                        {
                            "Column": column,
                            "Bin Label": i,
                            "Lower Bound": interval.left,
                            "Upper Bound": interval.right,
                            "Bin Interval": str(interval),
                        }
                    )
            else:
                # Fallback if categories are gone
                bin_summary_records.append(
                    {
                        "Column": column,
                        "Bin Label": "qcut binning",
                        "Lower Bound": "N/A",
                        "Upper Bound": "N/A",
                        "Bin Interval": "N/A",
                    }
                )

        if isinstance(df_with_bins.dtype, pd.CategoricalDtype):
            # Add category 0 for missing
            df_with_bins = df_with_bins.cat.add_categories([0]).fillna(0)
        else:
            df_with_bins = df_with_bins.fillna(0).astype(float)

        if df[column].isna().any():
            bin_summary_records.append(
                {
                    "Column": column,
                    "Bin Label": 0,
                    "Lower Bound": "NaN",
                    "Upper Bound": "NaN",
                    "Bin Interval": "NaN",
                }
            )

        return df_with_bins

    if columns == "All":
        columns_to_bin = [col for col in df.columns if col != target_column]
    else:
        if isinstance(columns, str):
            columns_to_bin = [columns]
        else:
            columns_to_bin = columns

    for column in columns_to_bin:
        # Skip if the column doesn't exist in df
        if column not in df.columns:
            continue

        df_with_bins = custom_binning(df, column)
        binned_column_name = f"{column}_bin"

        # Insert the new binned column immediately after the original
        col_index = df.columns.get_loc(column) + 1
        df.insert(col_index, binned_column_name, df_with_bins)

    bin_summary_df = pd.DataFrame(bin_summary_records)
    bin_summary_df.rename(columns={"Column": "Variable"}, inplace=True)
    bin_summary_df.set_index(["Variable", "Bin Label"], inplace=True)

    return df, bin_summary_df


import pandas as pd
from typing import Union, List

def drop_columns_after_WoE(
    df: pd.DataFrame, 
    columns_to_drop: Union[str, List[str]] = 'YEARS AT JOB_bin', 
    drop_na_columns: bool = True
) -> pd.DataFrame:
    """
    Drop specified columns and optionally drop columns with missing values from a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_drop (Union[str, List[str]]): Column(s) to drop. Default is 'YEARS AT JOB_bin'.
        drop_na_columns (bool): Whether to drop all columns with missing values. Default is True.

    Returns:
        pd.DataFrame: The updated DataFrame after dropping columns.
    """
    # Ensure columns_to_drop is a list for consistency
    if isinstance(columns_to_drop, str):
        columns_to_drop = [columns_to_drop]

    # Store the columns that will be dropped
    columns_dropped = []

    # Drop specified columns
    if columns_to_drop:
        for col in columns_to_drop:
            if col in df.columns:
                columns_dropped.append(col)
        df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    # Optionally drop columns with missing values--set to True to remove non-binned columns (but this design is not so pythonic)
    if drop_na_columns:
        columns_na = df.columns[df.isna().any()].tolist()
        df.dropna(axis=1, inplace=True)
        columns_dropped.extend(columns_na)

    if columns_dropped:
        print("Dropped columns:", ", ".join(columns_dropped))

    return df


import pandas as pd
from typing import List

def one_hot_encode_bins(df: pd.DataFrame, target_column: str = 'DEFAULT') -> pd.DataFrame:
    """
    One-hot encode all columns except the target column.

    Parameters:
        df (pd.DataFrame): The input DataFrame with bin values.
        target_column (str): The name of the target column to exclude from encoding.

    Returns:
        pd.DataFrame: The DataFrame with one-hot encoded columns.
    """
    target = df[target_column]

    # One-hot encode all columns except the target
    df = pd.get_dummies(df.drop(columns=[target_column]), prefix_sep='_', drop_first=True, dtype=int)
    
    # Add the target column back
    df[target_column] = target

    return df
