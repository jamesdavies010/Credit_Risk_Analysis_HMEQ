# 15 - calculate_single_predictor_metrics
# 114 - apply_model_and_evaluate

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

def calculate_single_predictor_metrics(
    df: pd.DataFrame, 
    response_variable: str, 
    prob_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Calculates logistic regression metrics for each feature against the response variable,
    while ensuring only relevant NaNs are dropped per feature analysis.
    
    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing features and the response variable.
    response_variable : str
        The name of the response variable column.
    prob_threshold : float
        The probability threshold for converting predictions into binary labels.
    
    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the metrics for each feature.
    """
    df_copy = df.copy()

    response_variable = 'DEFAULT'
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

        predicted_labels = (predictions >= prob_threshold).astype(int)

        AUC = roc_auc_score(y, predictions)
        F1 = f1_score(y, predicted_labels)
        Accuracy = accuracy_score(y, predicted_labels)
        Precision = precision_score(y, predicted_labels, zero_division=0)
        Recall = recall_score(y, predicted_labels, zero_division=0)

        results = {
            "Variable": variable,
            "Beta": beta_coefficient[variable],
            "Intercept": beta_coefficient["const"],
            "AUC": AUC,
            "F1 Score": F1,
            "Accuracy": Accuracy,
            "Precision": Precision,
            "Recall": Recall,
            "p-value": p_value[variable],
        }

        df_temp = pd.DataFrame(results, index=[idx])
        single_predictor_results = pd.concat([single_predictor_results, df_temp])

    single_predictor_results = single_predictor_results.sort_values(by="AUC", ascending=False)

    return single_predictor_results


import numpy as np
import pandas as pd
from typing import Dict
import statsmodels.api as sm
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def apply_model_and_evaluate(
    coefficients: Dict[str, float], new_data: pd.DataFrame, threshold: float = 0.3
) -> None:
    """
    Applies the model coefficients to the validation data to make predictions
    and evaluate the model.

    Parameters:
    - coefficients: Dict[str, float], model coefficients from the trained model
    - new_data: pd.DataFrame, validation dataset to apply the coefficients to
    - threshold: float, classification threshold for the predictions (default is 0.3)

    Returns:
    - None, but prints evaluation metrics and plots
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

    linear_combination = np.dot(
        X_new_with_intercept, np.array([intercept] + coefficients_values)
    )

    model = sm.Logit(new_data["DEFAULT"], X_new_with_intercept)
    result = model.fit(disp=False)
    p_values = result.pvalues
    coefficients_with_pvalues = pd.DataFrame(
        {"Coefficient": result.params, "P-value": p_values}
    )

    print(coefficients_with_pvalues.round(3))
    print("\n")

    predicted_proba = 1 / (1 + np.exp(-linear_combination))

    y_pred_class = (predicted_proba >= threshold).astype(int)

    print("Classification Report:")
    print(classification_report(new_data["DEFAULT"], y_pred_class))

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    metrics_data = []

    for thresh in thresholds:
        y_pred = (predicted_proba >= thresh).astype(int)

        accuracy = accuracy_score(new_data["DEFAULT"], y_pred)
        precision = precision_score(new_data["DEFAULT"], y_pred)
        recall = recall_score(new_data["DEFAULT"], y_pred)  # Sensitivity
        specificity = recall_score(new_data["DEFAULT"], y_pred, pos_label=0)
        f1 = f1_score(new_data["DEFAULT"], y_pred)

        metrics_data.append(
            {
                "Threshold": thresh,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall (Sensitivity)": recall,
                "Specificity": specificity,
                "F1 Score": f1,
            }
        )

    metrics_df = pd.DataFrame(metrics_data)

    # Plot sensitivity/specificity vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(
        metrics_df["Threshold"],
        metrics_df["Recall (Sensitivity)"],
        label="Sensitivity",
        marker="o",
    )
    plt.plot(
        metrics_df["Threshold"],
        metrics_df["Specificity"],
        label="Specificity",
        marker="o",
    )
    plt.plot(
        metrics_df["Threshold"], metrics_df["F1 Score"], label="F1 Score", marker="o"
    )
    plt.axvline(
        x=threshold, color="r", linestyle="--", label=f"Current Threshold ({threshold})"
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Sensitivity, Specificity and F1 Score vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nMetrics at Different Thresholds:")
    print(metrics_df.round(3))

    roc_auc = roc_auc_score(new_data["DEFAULT"], predicted_proba)
    print(f"ROC-AUC: {roc_auc:.3f}")

    fpr, tpr, _ = roc_curve(new_data["DEFAULT"], predicted_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

    cm = confusion_matrix(new_data["DEFAULT"], y_pred_class)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
