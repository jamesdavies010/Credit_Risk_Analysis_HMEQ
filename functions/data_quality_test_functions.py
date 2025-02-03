import pandas as pd
import numpy as np
from typing import List, Optional, Dict

def data_quality_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes a dataframe to assess data quality for each column.

    Parameters:
        df (pd.DataFrame): The dataframe to analyze.

    Returns:
        pd.DataFrame: A summary of data quality metrics for each column.
    """
    results: List[Dict[str, any]] = []
    
    for column in df.columns:
        col_data = df[column]
        col_name: str = column
        col_dtype: str = str(col_data.dtype)
        non_null_count: int = col_data.notnull().sum()
        missing_count: int = col_data.isnull().sum()
        missing_percent: float = (missing_count / len(df)) * 100
        unique_count: int = col_data.nunique()
        
        anomaly_message: str = "No anomalies detected"
        
        if col_dtype == 'object':
            if col_data.apply(lambda x: isinstance(x, (int, float))).any():
                anomaly_message = "Contains non-string values"
        
        elif col_dtype == 'category':
            if col_data.apply(lambda x: not isinstance(x, str) and pd.notnull(x)).any():
                anomaly_message = "Contains non-category (non-string) values"
        
        elif np.issubdtype(col_data.dtype, np.number):
            if col_data.apply(lambda x: not pd.api.types.is_number(x) and pd.notnull(x)).any():
                anomaly_message = "Contains non-numeric values"
        
        else: 
            anomaly_message = f"Unexpected datatype: {col_dtype}"
        
        results.append({
            "Column": col_name,
            "Data Type": col_dtype,
            "Non-Null Count": non_null_count,
            "Missing Values": missing_count,
            "% Missing Values": missing_percent,
            "Unique Values": unique_count,
            "Anomaly Check": anomaly_message
        })
    
    quality_summary: pd.DataFrame = pd.DataFrame(results).sort_values(by="% Missing Values", ascending=False)
    return quality_summary


def missing_values_by_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the distribution of missing values across rows in a DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame to analyze.
    
    Returns:
        DataFrame: A summary DataFrame with the number of missing values per row
                and the count of rows with that number of missing values, sorted
                in descending order.
    """
    temp_df = df.copy()
    temp_df['missing_values_per_row'] = temp_df.isnull().sum(axis=1)
    
    missing_values_summary = (
        temp_df['missing_values_per_row']
        .value_counts()
        .reset_index() # to convert series back to df
        .set_index('missing_values_per_row')
        .sort_index()
    )
    
    return missing_values_summary


def duplicate_check(df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = False) -> pd.DataFrame:
    """
    Identifies duplicate rows in a dataframe.

    Parameters:
        df (pd.DataFrame): The dataframe to analyze.
        subset (Optional[List[str]]): List of columns to check for duplicates. 
                                    If None, all columns are used.
        keep (str): Determines which duplicates (if any) to mark as True:
                    - "first": Mark duplicates except for the first occurrence.
                    - "last": Mark duplicates except for the last occurrence.
                    - False: Mark all duplicates as True (default).

    Returns:
        pd.DataFrame: A dataframe containing only the duplicate rows.
    """
    duplicate_rows = df[df.duplicated(subset=subset, keep=keep)]
    
    print(f"Found {len(duplicate_rows)} duplicate rows.")
    
    return duplicate_rows
