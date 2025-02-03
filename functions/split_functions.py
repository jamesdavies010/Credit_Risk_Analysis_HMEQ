from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Optional, Tuple

def training_val_test_split(
    df: pd.DataFrame,
    training: float = 65,
    validation: float = 10,
    test: float = 25,
    stratify: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into training, validation, and test sets with configurable proportions.

    Args:
        df (pd.DataFrame): The dataset to be split.
        training (float): Percentage of the data to allocate for training. Defaults to 65.
        validation (float): Percentage of the data to allocate for validation. Defaults to 10.
        test (float): Percentage of the data to allocate for testing. Defaults to 25.
        stratify (Optional[str]): Column to stratify splits by. Defaults to None (no stratification).
        random_state (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - training_df: Training set
            - validation_df: Validation set
            - test_df: Test set

    Raises:
        ValueError: If the sum of training, val, and test proportions is not 100.
    """

    total = training + validation + test
    if total != 100:
        raise ValueError(f"The proportions must sum up to 100. Currently, they sum up to {total}.")
    # Split the dataset into training/validation and test sets
    training_val_df, test_df = train_test_split(
        df,
        test_size=test / 100,
        stratify=df[stratify] if stratify else None,
        random_state=random_state,
    )

    # Split training/validation into training and validation sets
    training_df, validation_df = train_test_split(
        training_val_df,
        test_size=validation / (training + validation),
        stratify=training_val_df[stratify] if stratify else None,
        random_state=random_state,
    )

    print(
        "Datasets created:\n"
        f" - Training set: {len(training_df)} rows\n"
        f" - Validation set: {len(validation_df)} rows\n"
        f" - Test set: {len(test_df)} rows\n\n"
    )
    
    return training_df, validation_df, test_df


import pandas as pd
import numpy as np
from typing import List, Tuple

def calculate_target_percentage(datasets: List[Tuple[str, pd.DataFrame]], column: str) -> None:    
    """
    Calculate the percentage of 1s in the target column for multiple datasets and print the results.

    Args:
        datasets (List[Tuple[str, pd.DataFrame]]): A list of tuples where each tuple contains:
            - The dataset name (as a string).
            - The DataFrame itself.
        column (str): The name of the column to calculate percentages for.

    Raises:
        ValueError: If the specified column does not exist in any of the datasets.
    """
    results = {}
    for name, df in datasets:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame '{name}'.")
        
        total = len(df)
        ones = df[column].sum()
        percentage = (ones / total) * 100 if total > 0 else 0
        results[name] = percentage

    print("Percentage of 1s in target column:")
    for name, percentage in results.items():
        print(f" - {name}: {percentage:.2f}%")