# 14 - compare_results
# 96 - add_predicted_probabilities
# 130 - merge_datasets
# 180 - monotonicity_analysis

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from typing import Dict


def compare_results(coefficients: Dict[str, float], **datasets: pd.DataFrame) -> None:
    """
    Compares the ROC-AUC values of a model across multiple datasets
    by plotting their ROC curves on the same graph.

    Parameters:
    ----------
    coefficients: Dict[str, float]
        Model coefficients from the trained model.
    datasets: pd.DataFrame
        Keyword arguments for datasets to compare (e.g., training, validation, test).
        Each dataset should be passed as a named argument like `training=training_df`.

    Returns:
    -------
    None, but plots the ROC curves for each dataset.
    """

    def compute_roc_metrics(data: pd.DataFrame, dataset_label: str):
        # Ensure that the feature names in the data match the model's expected features
        feature_names = [key for key in coefficients if key != "const"]
        X = data[feature_names]

        if not all(feature in data.columns for feature in feature_names):
            missing_features = [
                feature for feature in feature_names if feature not in data.columns
            ]
            print(
                f"Error: Missing features in {dataset_label} data: {missing_features}"
            )
            return None, None, None

        X_with_intercept = sm.add_constant(X)

        # Compute linear combination
        intercept = coefficients["const"]
        coefficients_values = [
            coefficients[key] for key in coefficients if key != "const"
        ]
        linear_combination = np.dot(
            X_with_intercept, np.array([intercept] + coefficients_values)
        )

        # Predicted probabilities
        predicted_proba = 1 / (1 + np.exp(-linear_combination))

        # Calculate ROC-AUC score
        roc_auc = roc_auc_score(data["DEFAULT"], predicted_proba)
        fpr, tpr, _ = roc_curve(data["DEFAULT"], predicted_proba)

        return fpr, tpr, roc_auc

    # Calculate ROC metrics for each dataset
    metrics = {}
    for dataset_label, dataset in datasets.items():
        metrics[dataset_label] = compute_roc_metrics(dataset, dataset_label)

    # Plot ROC curves
    plt.figure(figsize=(10, 7))

    for dataset, (fpr, tpr, roc_auc) in metrics.items():
        if fpr is not None and tpr is not None:
            plt.plot(
                fpr, tpr, label=f"{dataset.capitalize()} (ROC-AUC = {roc_auc:.3f})"
            )

    plt.plot(
        [0, 1], [0, 1], color="gray", linestyle="--"
    )  # Diagonal line for reference
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Comparison of ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


import numpy as np
import pandas as pd
from typing import Dict
import statsmodels.api as sm

def add_predicted_probabilities(data: pd.DataFrame, coefficients: Dict[str, float]) -> pd.DataFrame:
    """
    Adds a column of predicted probabilities to the dataset based on the model coefficients.

    Parameters:
    - data: pd.DataFrame, the dataset to which probabilities will be added
    - coefficients: dict, the model coefficients (keys are feature names, values are coefficients)
    
    Returns:
    - pd.DataFrame: The dataset with a new column 'Predicted_Probability'
    """
    # Ensure feature names align with coefficients
    feature_names = [key for key in coefficients if key != 'const']
    missing_features = [feature for feature in feature_names if feature not in data.columns]

    if missing_features:
        raise ValueError(f"Missing features in the dataset: {missing_features}")

    X_data = data[feature_names]
    X_data_with_intercept = sm.add_constant(X_data)

    # Calculate linear combination
    intercept = coefficients['const']
    coefficient_values = [coefficients[key] for key in coefficients if key != 'const']
    linear_combination = np.dot(X_data_with_intercept, [intercept] + coefficient_values)

    # Convert log-odds to probabilities
    data['predicted_probability'] = 1 / (1 + np.exp(-linear_combination))
    return data


import pandas as pd
from typing import Optional

def merge_datasets(
    training_data: Optional[pd.DataFrame] = None,
    validation_data: Optional[pd.DataFrame] = None,
    test_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merges selected datasets (training, validation, test) into a single DataFrame.

    Parameters:
    - training_data: Optional[pd.DataFrame], the training dataset (default: None)
    - validation_data: Optional[pd.DataFrame], the validation dataset (default: None)
    - test_data: Optional[pd.DataFrame], the test dataset (default: None)

    Returns:
    - merged_data: pd.DataFrame, a single DataFrame with the selected datasets merged
    and an additional 'Dataset' column indicating the source of each row.
    """
    datasets = []

    if training_data is not None:
        training_data = training_data.copy()  # Avoid modifying the original DataFrame
        training_data["dataset"] = "Training"
        datasets.append(training_data)

    if validation_data is not None:
        validation_data = validation_data.copy()
        validation_data["dataset"] = "Validation"
        datasets.append(validation_data)

    if test_data is not None:
        test_data = test_data.copy()
        test_data["dataset"] = "Test"
        datasets.append(test_data)

    if not datasets:
        raise ValueError("At least one dataset must be provided for merging.")

    # Concatenate the datasets
    merged_data = pd.concat(datasets, ignore_index=False)

    return merged_data


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from IPython.display import display


def monotonicity_analysis(
    data: pd.DataFrame,
    prob_col: str = "predicted_probability",
    default_col: str = "DEFAULT",
    dataset_col: str = "dataset",
    num_groups: int = 10,
    group_by: Optional[str] = "quantile",
    analysis_mode: str = "all",
):
    """
    Performs monotonicity analysis by grouping borrowers and checking defaults in each group.

    Parameters:
    - data: pd.DataFrame, the dataset with predicted probabilities and default information
    - prob_col: str, the column name for predicted probabilities (default: 'predicted_probability')
    - default_col: str, the column name for the default status (default: 'DEFAULT')
    - dataset_col: str, the column name indicating the dataset split (e.g., 'Training', 'Validation', 'Test')
    - num_groups: int, the number of groups to divide the borrowers into (default: 10)
    - group_by: str, 'quantile' to group by quantiles, or 'cutoff' to group by probability cutoffs
    - analysis_mode: str, 'all' to analyze the whole dataset, or 'by_dataset' to analyze each dataset split

    Returns:
    - results: pd.DataFrame or dict, a summary table of groups and defaults
    If analysis_mode = 'all': returns a single DataFrame
    If analysis_mode = 'by_dataset': returns a dictionary of DataFrames for each dataset split
    """

    def analyze_subset(subset, label):
        if group_by == "quantile":
            subset["group"] = pd.qcut(
                subset[prob_col], num_groups, labels=range(1, num_groups + 1)
            )
        elif group_by == "cutoff":
            # group by custom cutoffs (e.g., 0-0.1, 0.1-0.2, ...)
            bins = np.linspace(0, 1, num_groups + 1)
            subset["group"] = pd.cut(
                subset[prob_col],
                bins,
                labels=range(1, num_groups + 1),
                include_lowest=True,
            )
        else:
            raise ValueError("group_by must be 'quantile' or 'cutoff'")

        # Aggregate defaults per group
        monotonicity_df = subset.groupby("group", observed=False).agg(
            defaults=(default_col, "sum"),
            total=("group", "size"),
            average_probability=(prob_col, "mean"),
        )
        monotonicity_df["default_rate"] = (
            monotonicity_df["defaults"] / monotonicity_df["total"]
        )

        # Add Trend column
        monotonicity_df["Trend"] = (
            monotonicity_df["default_rate"]
            .diff()
            .apply(
                lambda x: (
                    "Increasing" if x > 0 else ("Decreasing" if x < 0 else "No Change")
                )
            )
        )

        return monotonicity_df

    if analysis_mode == "all":
        monotonicity_df = analyze_subset(data, "All")

        plt.figure(figsize=(10, 6))
        plt.plot(
            monotonicity_df.index,
            monotonicity_df["default_rate"],
            marker="o",
            label="All Data",
        )
        plt.xlabel("Group")
        plt.ylabel("Default Rate")
        plt.title(f"Monotonicity Analysis ({group_by.capitalize()}-based groups)")
        plt.grid(True)
        plt.legend()
        plt.show()

        return monotonicity_df

    elif analysis_mode == "by_dataset":
        datasets = data[dataset_col].unique()
        results = {}

        plt.figure(figsize=(10, 6))
        for dataset in datasets:
            subset = data[data[dataset_col] == dataset].copy()
            monotonicity_df = analyze_subset(subset, dataset)
            results[dataset] = monotonicity_df

            plt.plot(
                monotonicity_df.index,
                monotonicity_df["default_rate"],
                marker="o",
                label=f"{dataset} Data",
            )

        plt.xlabel("Group")
        plt.ylabel("Default Rate")
        plt.title(
            f"Monotonicity Analysis by Dataset ({group_by.capitalize()}-based groups)"
        )
        plt.grid(True)
        plt.legend()
        plt.show()

        # Print the results for each dataset
        for dataset in datasets:
            subset = data[data[dataset_col] == dataset].copy()
            monotonicity_df = analyze_subset(subset, dataset)
            print(f"Results for {dataset} dataset:")
            display(monotonicity_df)
            print("\n")
