# 20 - calculate_single_predictor_metrics
# 105 - check_vif
# 159 - calculate_WoE_and_IV
# 271 - multiple_feature_selection_skl
# 396 - multiple_feature_selection_sm
# 528 - get_model_coefficients
# 608 - apply_model_and_evaluate


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import (
    roc_auc_score
)

def calculate_single_predictor_metrics(
    df: pd.DataFrame, 
    response_variable: str
) -> pd.DataFrame:
    """
    Calculates logistic regression metrics (ROC-AUC and p-value) for each feature 
    against the response variable, while ensuring only relevant NaNs are dropped 
    per feature analysis.
    
    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing features and the response variable.
    response_variable : str
        The name of the response variable column.
    
    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the metrics for each feature.
    """
    df_copy = df.copy()
    independent_variables = [col for col in df_copy.columns if col != response_variable]
    single_predictor_results = pd.DataFrame()
    
    for idx, variable in enumerate(independent_variables):
        valid_data = df_copy[[variable, response_variable]].dropna()
        
        print(f"Variable: {variable}, Non-Null Values: {valid_data[variable].notnull().sum()} (Feature), "
              f"{valid_data[response_variable].notnull().sum()} (Response)")
        
        if valid_data.empty:
            print(f"Skipping {variable} as it has no valid data.\n")
            continue
        
        x = valid_data[variable]
        y = valid_data[response_variable]
        constant = sm.add_constant(x)
        
        try:
            model = sm.Logit(y, constant).fit(disp=0)
        except Exception as e:
            print(f"Error fitting model for variable '{variable}': {e}")
            continue
        
        beta_coefficient = model.params.to_dict()
        p_value = {k: float("{:.2f}".format(v)) for k, v in model.pvalues.to_dict().items()}
        predictions = model.predict(constant)

        AUC = roc_auc_score(y, predictions)

        results = {
            "Variable": variable,
            "Beta": beta_coefficient[variable],
            "Intercept": beta_coefficient["const"],
            "AUC": AUC,
            "p-value": p_value[variable],
        }

        df_temp = pd.DataFrame(results, index=[idx])
        single_predictor_results = pd.concat([single_predictor_results, df_temp])

    single_predictor_results = single_predictor_results.sort_values(by="AUC", ascending=False)
    
    return single_predictor_results


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def check_vif(
    df: pd.DataFrame, 
    target_variable: str = 'DEFAULT'
) -> pd.DataFrame:
    """
    Calculates the Variance Inflation Factor (VIF) for each independent variable in the provided DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the independent variables and target variable.
    target_variable : str, optional
        The name of the target variable column. Defaults to 'DEFAULT'. If specified, the column will be 
        excluded from the independent variables when calculating VIF.
    
    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the Variance Inflation Factor (VIF) for each independent variable. 
        The DataFrame includes two columns: "VIF" and "feature", with the features sorted by VIF in descending order.
    
    Example:
    --------
    # Using the default target variable 'DEFAULT'
    vif_results = check_vif(training_df_no_na)
    
    # Using a custom target variable 'TARGET'
    vif_results_custom_target = check_vif(training_df_no_na, target_variable='TARGET')
    """
    df_temp = df.copy()
    df_temp = df_temp.dropna()

    independent_variables: list[str] = [col for col in df_temp.columns if col != target_variable]

    df_numeric = df_temp[independent_variables].select_dtypes(include=['number', 'float', 'int'])

    X_with_constant: pd.DataFrame = sm.add_constant(df_numeric)

    vif: pd.DataFrame = pd.DataFrame()
    vif["VIF"] = [
        variance_inflation_factor(X_with_constant.values, i)
        for i in range(X_with_constant.shape[1])
    ]
    vif["feature"] = ["const"] + df_numeric.columns.tolist()

    vif = vif.sort_values("VIF", ascending=False)

    return vif


import pandas as pd
import numpy as np


def calculate_WoE_and_IV(
    df: pd.DataFrame, bin_df: pd.DataFrame, target_column: str = "DEFAULT"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV) for each binned variable
    and join the results with bin interval details.

    This function requires a `bin_df` dataframe containing columns such as 'Column',
    'Bin Label', 'Lower Bound', 'Upper Bound', and 'Bin Interval'.

    Parameters:
        df (pd.DataFrame): The input dataframe containing binned variables and a target column.
        bin_df (pd.DataFrame): The dataframe containing bin interval details.
        target_column (str): The column name representing the binary target variable (default is 'DEFAULT').

    Returns:
        tuple: A dataframe summarizing WoE and IV calculations, and a series summarizing IV values per variable.
    """
    pd.set_option("display.float_format", lambda x: "%.3f" % x)

    # Identify columns with bins
    columns_with_bins = [
        col for col in df.columns if "_bin" in col and col != target_column
    ]
    summary_dfs = []
    information_values_by_column = {}

    for column_binned in columns_with_bins:
        # Group by bin and calculate WoE and IV components
        grouped_df = (
            df.groupby(column_binned, observed=False)[target_column]
            .value_counts()
            .unstack(fill_value=0)
        )
        grouped_df.columns = ["Good", "Bad"]

        total_good = grouped_df["Good"].sum()
        total_bad = grouped_df["Bad"].sum()

        grouped_df["No. of values"] = grouped_df["Good"] + grouped_df["Bad"]
        grouped_df["Good_%"] = (grouped_df["Good"] / total_good) * 100
        grouped_df["Bad_%"] = (grouped_df["Bad"] / total_bad) * 100
        grouped_df["WoE"] = np.log(
            (grouped_df["Good"] / total_good) / (grouped_df["Bad"] / total_bad)
        )
        grouped_df["Partial_IV"] = (
            grouped_df["Good_%"] / 100 - grouped_df["Bad_%"] / 100
        ) * grouped_df["WoE"]
        grouped_df["IV_sum"] = grouped_df["Partial_IV"].sum()

        grouped_df.reset_index(inplace=True)
        grouped_df.rename(columns={grouped_df.columns[0]: "Bin Label"}, inplace=True)
        grouped_df.insert(0, "Variable", column_binned)

        # Append to summary and collect IV per variable
        summary_dfs.append(grouped_df)
        information_values_by_column[column_binned] = grouped_df["Partial_IV"].sum()

    # Concatenate all variable summaries into one dataframe
    summary_of_bins_iv_df = pd.concat(summary_dfs, ignore_index=True)

    # Join with bin_df to add bin intervals
    summary_of_bins_iv_df["Variable"] = summary_of_bins_iv_df["Variable"].str.replace(
        "_bin", ""
    )
    summary_of_bins_iv_df = pd.merge(
        bin_df,
        summary_of_bins_iv_df,
        left_on=["Variable", "Bin Label"],
        right_on=["Variable", "Bin Label"],
        how="outer",
    )

    # Drop redundant columns and clean up
    summary_of_bins_iv_df.drop(
        columns=["Variable_y", "index", "Lower Bound", "Upper Bound"],
        inplace=True,
        errors="ignore",
    )
    summary_of_bins_iv_df.set_index(
        ["Variable", "IV_sum", "Bin Label", "Bin Interval"], inplace=True
    )

    # Calculate overall IV summary
    information_values_by_column_df = (
        pd.Series(information_values_by_column, name="Information Value")
        .sort_values(ascending=False)
        .reset_index()
    )
    information_values_by_column_df.columns = ["Variable", "Information Value"]
    information_values_by_column_df["Variable"] = information_values_by_column_df[
        "Variable"
    ].str.replace("_bin", "")

    return summary_of_bins_iv_df, information_values_by_column_df


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import pandas as pd
import numpy as np
from typing import List


def multiple_feature_selection_skl(
    df: pd.DataFrame,
    response_variable: str = "DEFAULT",
    scoring: List[str] = ["roc_auc", "f1"],
    threshold: float = 0.5,
    num_features: int = None,
    model=LogisticRegression(),
    is_validation_df: bool = False,
    top_5: bool = False,
) -> pd.DataFrame:
    """
    Performs feature selection using RFE, automatically determining the optimal number of features
    based on the elbow method, and allows setting a custom probability threshold for classification.

    If `is_validation_df` is True, the model will be trained on the whole dataset without cross-validation.

    Parameters:
    - df: pd.DataFrame, The dataset containing features and target variable
    - response_variable: str, The name of the target variable (default='DEFAULT')
    - scoring: List[str], The scoring metrics for evaluation (default=['roc_auc', 'f1'])
    - threshold: float, The threshold for converting predicted probabilities into class labels (default=0.5)
    - model: sklearn model (optional), If provided, this will be used instead of LogisticRegression
    - is_validation_df: bool, Whether the input dataframe is a validation set (default=False)
    - top_5: bool or str, Whether to return the top 5 feature sets based on a specific metric (default=False)
        - If 'roc_auc', it sorts the results by AUC.
        - If 'f1', it sorts the results by F1 score.

    Returns:
    - all_scores_df: pd.DataFrame, DataFrame with results for different feature counts
    - top_5_scores_df: pd.DataFrame, DataFrame with the top 5 results based on the specified metric
    """
    if response_variable not in df.columns:
        raise ValueError(
            f"Response variable '{response_variable}' not found in the DataFrame columns."
        )

    independent_variables = [col for col in df.columns if col != response_variable]
    y = df[response_variable]
    X = df[independent_variables]

    # Initialize storage for tracking scores
    all_scores = []

    # Test range of feature counts from 2 to total features
    feature_counts = range(2, len(independent_variables) + 1)

    metric_functions = {
        "f1": f1_score,
        "roc_auc": roc_auc_score,
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
    }

    for n_features in feature_counts:
        # RFE uses coefficient magnitude when selecting suitable features. This should be fine for binned variables, but be aware of this in future
        rfe = RFE(estimator=model, n_features_to_select=n_features, step=1)
        rfe.fit(X, y)

        # support_ produces a 'True'/'False' for whether column has been selected
        current_features = X.columns[rfe.support_].tolist()

        cv_scores = {metric: [] for metric in scoring}

        if is_validation_df:
            # If it's a validation dataset, train the model on the entire dataset
            model.fit(X[current_features], y)
            y_probs = model.predict_proba(X[current_features])[
                :, 1
            ]  # gives me just the probability of class = 1

            for metric in scoring:
                if metric in metric_functions:
                    y_pred = (
                        (y_probs >= threshold).astype(int)
                        if metric != "roc_auc"
                        else y_probs
                    )
                    score = metric_functions[metric](y, y_pred)
                    cv_scores[metric].append(score)
        else:
            # If it's not a validation dataset, perform cross-validation
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for train_idx, val_idx in kfold.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train[current_features], y_train)

                y_probs = model.predict_proba(X_val[current_features])[
                    :, 1
                ]  # gives me just the probability of class = 1

                for metric in scoring:
                    if metric in metric_functions:
                        y_pred = (
                            (y_probs >= threshold).astype(int)
                            if metric != "roc_auc"
                            else y_probs
                        )
                        score = metric_functions[metric](y_val, y_pred)
                        cv_scores[metric].append(score)

        row = {"n_features": n_features, "selected_features": current_features}
        for metric in scoring:
            row[f"{metric}_avg_score"] = np.mean(cv_scores[metric])
            row[f"{metric}_std_dev"] = np.std(cv_scores[metric])
            row[f"{metric}_cv_scores"] = [round(x, 3) for x in cv_scores[metric]]

        all_scores.append(row)

    all_scores_df = pd.DataFrame(all_scores)
    all_scores_df = all_scores_df.set_index("n_features")

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.float_format", "{:,.3f}".format)

    if top_5:
        if top_5 not in ["roc_auc", "f1"]:
            raise ValueError("top_5 must be either 'roc_auc' or 'f1'.")

        # Sort by the chosen metric and select the top 5 rows
        top_5_scores = all_scores_df.sort_values(
            by=f"{top_5}_avg_score", ascending=False
        ).head(5)

        return top_5_scores

    return all_scores_df


import pandas as pd
import numpy as np
from typing import List
import statsmodels.api as sm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

def multiple_feature_selection_sm(
    df: pd.DataFrame,
    response_variable: str = "DEFAULT",
    scoring: List[str] = ["roc_auc", "f1"],
    threshold: float = 0.5,
    num_features: int = None,
    model=None,  # No model here as we're using statsmodels directly
    is_validation_df: bool = False,
    top_5: bool = False,
    p_value_threshold: float = 0.05  # Add p-value threshold for feature selection
) -> pd.DataFrame:
    """
    Performs feature selection using p-value based selection, automatically determining the optimal number of features
    based on the p-value threshold, and allows setting a custom probability threshold for classification.

    If `is_validation_df` is True, the model will be trained on the whole dataset without cross-validation.

    Parameters:
    - df: pd.DataFrame, The dataset containing features and target variable
    - response_variable: str, The name of the target variable (default='DEFAULT')
    - scoring: List[str], The scoring metrics for evaluation (default=['roc_auc', 'f1'])
    - threshold: float, The threshold for converting predicted probabilities into class labels (default=0.5)
    - model: None (not used), as we're using statsmodels' Logit model directly
    - is_validation_df: bool, Whether the input dataframe is a validation set (default=False)
    - top_5: bool or str, Whether to return the top 5 feature sets based on a specific metric (default=False)
        - If 'roc_auc', it sorts the results by AUC.
        - If 'f1', it sorts the results by F1 score.
    - p_value_threshold: float, p-value threshold for feature selection (default=0.05)

    Returns:
    - all_scores_df: pd.DataFrame, DataFrame with results for different feature counts
    - top_5_scores_df: pd.DataFrame, DataFrame with the top 5 results based on the specified metric
    """
    if response_variable not in df.columns:
        raise ValueError(
            f"Response variable '{response_variable}' not found in the DataFrame columns."
        )

    independent_variables = [col for col in df.columns if col != response_variable]
    y = df[response_variable]
    X = df[independent_variables]

    all_scores = []

    # Test range of feature counts from 2 to total features
    feature_counts = range(2, len(independent_variables) + 1)

    metric_functions = {
        "f1": f1_score,
        "roc_auc": roc_auc_score,
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
    }

    for n_features in feature_counts:
        current_features = independent_variables[:n_features]  # Start with the first 'n_features'
        selected_features = current_features.copy()
        
        X_with_intercept = sm.add_constant(X[selected_features])
        
        model = sm.Logit(y, X_with_intercept).fit(disp=0)
        
        while True:
            p_values = model.pvalues[1:]  # Ignore intercept
            max_p_value = p_values.max()
            if max_p_value >= p_value_threshold:
                feature_to_remove = p_values.idxmax()
                selected_features.remove(feature_to_remove)
                X_with_intercept = sm.add_constant(X[selected_features])
                model = sm.Logit(y, X_with_intercept).fit(disp=0)  # Refit after removing feature
            else:
                break
        
        cv_scores = {metric: [] for metric in scoring}
        
        if is_validation_df:
            X_with_intercept = sm.add_constant(X[selected_features])
            model = sm.Logit(y, X_with_intercept).fit(disp=0)
            y_probs = model.predict(X_with_intercept)
            
            for metric in scoring:
                if metric in metric_functions:
                    y_pred = (y_probs >= threshold).astype(int) if metric != 'roc_auc' else y_probs
                    score = metric_functions[metric](y, y_pred)
                    cv_scores[metric].append(score)
        else:
            # If it's not a validation dataset, perform cross-validation
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for train_idx, val_idx in kfold.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                X_train_with_intercept = sm.add_constant(X_train[selected_features])
                X_val_with_intercept = sm.add_constant(X_val[selected_features])
                
                model = sm.Logit(y_train, X_train_with_intercept).fit(disp=0)
                
                y_probs = model.predict(X_val_with_intercept)
                
                for metric in scoring:
                    if metric in metric_functions:
                        y_pred = (y_probs >= threshold).astype(int) if metric != 'roc_auc' else y_probs
                        score = metric_functions[metric](y_val, y_pred)
                        cv_scores[metric].append(score)
        
        row = {'n_features': len(selected_features), 'selected_features': selected_features}
        for metric in scoring:
            row[f'{metric}_avg_score'] = np.mean(cv_scores[metric])
            row[f'{metric}_std_dev'] = np.std(cv_scores[metric])
            row[f'{metric}_cv_scores'] = [round(x, 3) for x in cv_scores[metric]]
            
        all_scores.append(row)
    
    all_scores_df = pd.DataFrame(all_scores)
    all_scores_df = all_scores_df.set_index('n_features')
    
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.float_format', '{:,.3f}'.format)

    if top_5:
        if top_5 not in ['roc_auc', 'f1']:
            raise ValueError("top_5 must be either 'roc_auc' or 'f1'.")
        
        # Sort by the chosen metric and select the top 5 rows
        top_5_scores = all_scores_df.sort_values(by=f'{top_5}_avg_score', ascending=False).head(5)
        
        return top_5_scores

    return all_scores_df


import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

def get_model_coefficients(
    top_5_scores_df: pd.DataFrame,  # The top 5 feature sets dataframe
    n_features_index: int,  # The index of the row from the top 5 results to choose
    model_type="statsmodels",  # Model type - "statsmodels" or "sklearn"
    data_source: pd.DataFrame = None  # DataFrame containing the features and target
    ) -> dict:
    """
    Extracts the model coefficients from the trained model without printing 
    the entire model summary or running full evaluation. 
    
    Parameters:
    - top_5_scores_df: pd.DataFrame, the top 5 feature sets dataframe
    - n_features_index: int, the row index from the top 5 results to choose
    - model_type: str, "statsmodels" or "sklearn" (default is "statsmodels")
    - data_source: pd.DataFrame, DataFrame containing features and target
    
    Returns:
    - Coefficients (intercept + features) as a dictionary.
    """
    if model_type == "statsmodels":
        selected_row = top_5_scores_df.iloc[n_features_index]
        selected_independent_variables = selected_row['selected_features']
        if isinstance(selected_independent_variables, str):
            selected_independent_variables = selected_independent_variables.split(", ")

        X_selected = data_source[selected_independent_variables]
        y = data_source["DEFAULT"]

        X_selected_with_intercept = sm.add_constant(X_selected)

        logit_model = sm.Logit(y, X_selected_with_intercept)
        result = logit_model.fit(disp=0)

        coefficients = result.params.to_dict()

        print("\nCoefficients and P-values:")
        print(pd.DataFrame({
            'Coefficient': result.params,
            'P-value': result.pvalues
        }).round(3))

        return coefficients

    elif model_type == "sklearn":
        selected_row = top_5_scores_df.iloc[n_features_index]
        selected_independent_variables = selected_row['selected_features']
        if isinstance(selected_independent_variables, str):
            selected_independent_variables = selected_independent_variables.split(", ")

        X_selected = data_source[selected_independent_variables]
        y = data_source["DEFAULT"]

        logit_model = LogisticRegression()
        logit_model.fit(X_selected, y)

        coefficients = logit_model.coef_.flatten()
        intercept = logit_model.intercept_[0]

        return {
            "coefficients": coefficients.tolist(),
            "intercept": intercept
        }

    else:
        print("Error: Unsupported model type. Choose 'statsmodels' or 'sklearn'.")
        return {}


import numpy as np
import pandas as pd
from typing import Dict
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def apply_model_and_evaluate(
    coefficients: Dict[str, float], new_data: pd.DataFrame
) -> None:
    """
    Applies the model coefficients to the validation data to make predictions
    and evaluates the model using the ROC-AUC score.

    Parameters:
    ----------
    coefficients: Dict[str, float]
        Model coefficients from the trained model.
    new_data: pd.DataFrame
        Validation dataset to apply the coefficients to.

    Returns:
    -------
    None, but prints evaluation metrics and plots the ROC curve.
    """
    # Ensure that the feature names in new data match the model's expected features
    feature_names = [key for key in coefficients if key != "const"]

    # Check if the validation data contains the required features
    missing_features = [
        feature for feature in feature_names if feature not in new_data.columns
    ]
    if missing_features:
        print(f"Error: Missing features in validation data: {missing_features}")
        return

    X_new = new_data[feature_names]

    X_new_with_intercept = sm.add_constant(X_new)

    intercept = coefficients["const"]
    coefficients_values = [coefficients[key] for key in coefficients if key != "const"]

    # Compute the linear combination for predictions
    linear_combination = np.dot(
        X_new_with_intercept, np.array([intercept] + coefficients_values)
    )

    # Calculate predicted probabilities
    predicted_proba = 1 / (1 + np.exp(-linear_combination))

    # Calculate and display the ROC-AUC score
    roc_auc = roc_auc_score(new_data["DEFAULT"], predicted_proba)
    print(f"ROC-AUC: {roc_auc:.3f}")

    # Plot the ROC curve
    fpr, tpr, _ = roc_curve(new_data["DEFAULT"], predicted_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()
