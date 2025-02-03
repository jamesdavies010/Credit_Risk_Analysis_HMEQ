import pandas as pd
from sklearn.utils import resample


def balance_data_undersampling(
    df: pd.DataFrame, target_col: str = "DEFAULT"
) -> pd.DataFrame:
    """
    Balances the dataset via undersampling.

    This function identifies the majority class (assumed to be labeled '0')
    and minority class (assumed to be labeled '1') in the DataFrame, then
    undersamples the majority class to match the number of samples in the
    minority class. The undersampled majority class and the minority class
    are combined, and the resulting DataFrame is shuffled and returned.

    Args:
        df (pd.DataFrame):
            The input DataFrame containing the data.
        target_col (str, optional):
            The name of the target column containing the class labels.
            Defaults to 'DEFAULT'.

    Returns:
        pd.DataFrame:
            A balanced DataFrame after undersampling the majority class.
    """
    majority_class = df[df[target_col] == 0]  # Assuming 0 is the majority class
    minority_class = df[df[target_col] == 1]  

    majority_class_undersampled = resample(
        majority_class, replace=False, n_samples=len(minority_class), random_state=42
    )

    balanced_df = pd.concat([majority_class_undersampled, minority_class])

    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df
