import pandas as pd
from typing import List, Optional

def convert_object_to_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all columns with 'object' dtype in a dataframe to 'category' dtype.

    Parameters:
        df (pd.DataFrame): The dataframe to modify.

    Returns:
        pd.DataFrame: A dataframe with 'object' columns converted to 'category'.
    """
    object_columns = df.select_dtypes(include=['object']).columns
    
    for col in object_columns:
        df[col] = df[col].astype('category')
        print(f"Converted column '{col}' from 'object' to 'category'")
