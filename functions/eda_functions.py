# 13 - describe_dataframe 
# 38 - create boxplots
# 92 - correlation_matrix
# 138 - calculate_bins_for_eda
# 190 - chart_visualisations
# 410 - show_defaults_in_missing_values


import pandas as pd
from typing import Optional, Union


def describe_dataframe(
    df: pd.DataFrame, decimals: int = 2, include: Optional[Union[str, list]] = None
) -> pd.DataFrame:
    """
    Returns a transposed summary of the DataFrame with rounded statistics.

    Parameters:
        df (pd.DataFrame): The DataFrame to describe.
        decimals (int): The number of decimal places to round to.
        include (Optional[Union[str, list]]): Determines the scope of the describe method.
                                            For example, use 'all' to include all columns (including non-numeric).

    Returns:
        pd.DataFrame: Transposed and rounded summary statistics.
    """
    return df.describe(include=include).round(decimals).transpose()


from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_boxplots(
    df: pd.DataFrame,
    target_column: str = "DEFAULT",
    columns_by_type: Optional[Dict[str, List[str]]] = None,
) -> None:
    """
    Creates boxplots of features in the DataFrame grouped by the target column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The column used to group the data in the boxplots. Defaults to "DEFAULT".
        columns_by_type (Optional[Dict[str, List[str]]]): A dictionary categorizing columns by type.
                                                        Columns categorized as 'Category' will be excluded
                                                        from the boxplots. Defaults to None.

    Returns:
        None: Displays the boxplots using Matplotlib and Seaborn.
    """
    df = df.copy()

    feature_columns = [col for col in df.columns if col != target_column]

    # Exclude categorical columns
    if columns_by_type is not None:
        feature_columns = [
            col
            for col in feature_columns
            if col not in columns_by_type.get("Category", [])
        ]

    # Calculate number of rows needed
    n_rows = int(np.ceil(len(feature_columns) / 2))

    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(20, 8 * n_rows))
    axes = axes.flatten()

    for i, var in enumerate(feature_columns):
        sns.boxplot(data=df, x=target_column, y=var, hue="DEFAULT", ax=axes[i])

        # Customize each subplot
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].set_title(f"{var} by {target_column}", pad=15)
        axes[i].set_xlabel(f"{target_column}")
        axes[i].set_ylabel("Value")

    # Hide any empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def correlation_matrix(
    df: pd.DataFrame, columns_to_exclude: Optional[List[str]] = None
) -> None:
    """
    Generates a correlation matrix heatmap for the given DataFrame, optionally excluding specified columns.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data to plot.
    columns_to_exclude (Optional[List[str]]): List of column names to exclude from the correlation calculation. Default is None.

    Returns:
    None: Displays the generated heatmap.
    """
    if columns_to_exclude is not None:
        df = df.drop(columns=columns_to_exclude)

    df = df.select_dtypes(include=["number"])

    corr_matrix = df.corr().round(2)

    fig = ff.create_annotated_heatmap(
        corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        colorscale="RdBu",
        reversescale=False,
        zmin=-1,
        zmax=1,
    )

    fig.update_layout(
        title="Correlation matrix",
        xaxis_tickangle=-45,
        coloraxis_colorbar=dict(title="Correlation"),
    )
    
    return fig


from typing import Tuple, Union, Optional, List, Dict
import pandas as pd
import numpy as np
import plotly.figure_factory as ff


# this is used in the chart_visualisations function
def calculate_bins_for_eda(
    data: pd.Series,
) -> Tuple[Union[float, int], Union[float, int], Union[float, int], int]:
    """
    Calculates bin parameters for numeric and categorical data.

    Parameters:
        data (pd.Series): A Pandas Series (column) to calculate bins for.

    Returns:
        Tuple[Union[float, int], Union[float, int], Union[float, int], int]:
            - bin_start (float or int): The start of the bins (always 0 for numeric data).
            - bin_end (float or int): The end of the bins.
            - bin_size (float or int): The size of each bin (always 1 for categorical data).
            - n_bins (int): The number of bins.
    """
    if isinstance(data.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(
        data
    ):
        unique_values: int = len(data.unique())
        bin_start: int = 0
        bin_end: int = unique_values
        bin_size: int = 1
        n_bins: int = unique_values
    else:
        data = data.dropna()
        if data.empty:
            return 0, 1, 1, 1

        data_min: float = max(0, data.min())
        data_max: float = data.max()
        data_range: float = data_max - data_min

        n_bins: int = min(20, len(data.unique()))  # maximum of 20 bins
        bin_size: float = max(1, round(data_range / n_bins))  # avoid zero-sized bins

        # Adjust bin size to a more rounded number (e.g., 1, 2, 5, 10, etc.)
        if bin_size >= 10:
            scale = 10 ** (len(str(int(bin_size))) - 1)
            bin_size = round(bin_size / scale) * scale

        bin_start: float = 0
        bin_end: float = np.ceil(data_max / bin_size) * bin_size

    return bin_start, bin_end, bin_size, n_bins


import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def chart_visualisations(
    df: pd.DataFrame,
    legend_column: str = "DEFAULT",
    n_cols: int = 2,
    columns_by_type: dict = None,  
    specific_columns: list = None,  
) -> None:

    """
    Creates a 2-column subplot visualization for each relevant column in the provided DataFrame.
    
    - For discrete or categorical columns, the left subplot shows a count bar plot (split by `legend_column`),
    and the right subplot shows the corresponding percentage of each category.
    
    - For continuous columns, the left subplot shows a histogram (split by `legend_column`),
    and the right subplot shows the corresponding percentage histogram in stacked form.
    
    The subplots are arranged in rows, one row per column in `relevant_columns`. The function uses
    Plotly for interactive visualizations.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to visualize. 
        A copy of this DataFrame is used internally, so the original remains unchanged.

    legend_column : str, optional
        The name of the column that differentiates the data groups (e.g., "DEFAULT", "target").
        Each bar or histogram series will be split according to the unique values in this column.
        Default is "DEFAULT".

    n_cols : int, optional
        The number of columns in the subplot layout. Currently fixed to 2 (count vs. percentage).
        Default is 2.

    columns_by_type : dict, optional
        A dictionary that specifies which columns are "Discrete", "Category", and "Continuous".
        If None, the function will infer these types from the DataFrame dtypes, categorizing
        object or categorical columns as "Category" and numeric columns as "Continuous".
        Default is None.

    specific_columns : list, optional
        A list of specific columns to visualize. If provided, only these columns are used.
        If None, the function looks to `columns_by_type` (or infers it) to determine which columns
        are "Discrete", "Category", or "Continuous". Default is None.

    Returns:
    -------
    None
        Displays the final Plotly figure inline (e.g., in a Jupyter notebook) or opens it in a browser
        depending on your environment. No object is returned.
    """

    df = df.copy()

    if specific_columns is not None:
        relevant_columns = [col for col in specific_columns if col in df.columns]
    else:
        if columns_by_type is None:
            columns_by_type = {"Discrete": [], "Category": [], "Continuous": []}

            for col in df.columns:
                if col != legend_column:
                    if isinstance(
                        df[col].dtype, pd.CategoricalDtype
                    ) or pd.api.types.is_object_dtype(df[col]):
                        columns_by_type["Category"].append(col)
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        columns_by_type["Continuous"].append(col)

        relevant_columns = [
            col
            for col in df.columns
            if col in [c for group in columns_by_type.values() for c in group]  
        ]

    n_rows = len(relevant_columns)

    subplot_titles = []
    for col in relevant_columns:
        subplot_titles.extend([f"{col}", f"{col}"])

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,  # Always 2 columns - left for count, right for percentage
        subplot_titles=subplot_titles,
    )

    colour_map = {0: "rgb(31, 119, 180)", 1: "rgb(255, 127, 14)"}  # Blue and orange

    for idx, col in enumerate(relevant_columns):
        row_num = idx + 1

        is_discrete_or_category = (
            col in columns_by_type["Discrete"] or col in columns_by_type["Category"]
        ) if specific_columns is None else False

        # Determine bar width based on column type
        bar_width = 0.7 if is_discrete_or_category else 1.0

        if is_discrete_or_category:
            data = df[[col, legend_column]].dropna()
            grouped_data = (
                data.groupby([col, legend_column], observed=False)
                .size()
                .unstack(fill_value=0)
            )
            
            percentages = grouped_data.apply(lambda row: row / row.sum() * 100, axis=1)

            for response_value in df[legend_column].unique():
                fig.add_trace(
                    go.Bar(
                        x=grouped_data.index.astype(str),
                        y=grouped_data[response_value],
                        name=f"{legend_column} = {response_value}",
                        marker_color=colour_map[response_value],
                        marker_line_color="black",
                        marker_line_width=0.25,
                        showlegend=False,
                        width=bar_width,  # Adjust bar width
                    ),
                    row=row_num,
                    col=1,
                )
                fig.add_trace(
                    go.Bar(
                        x=percentages.index.astype(str),
                        y=percentages[response_value],
                        name=f"{legend_column} = {response_value}",
                        marker_color=colour_map[response_value],
                        marker_line_color="black",
                        marker_line_width=0.25,
                        showlegend=False,
                        width=bar_width,  # Adjust bar width
                    ),
                    row=row_num,
                    col=2,
                )

        else:
            # Continuous column: Histogram
            data = df[col].dropna()
            bin_start, bin_end, bin_size, n_bins = calculate_bins_for_eda(data)

            for response_value in df[legend_column].unique():
                data_filtered = df[df[legend_column] == response_value][col].dropna()

                fig.add_trace(
                    go.Histogram(
                        x=data_filtered,
                        name=f"{legend_column} = {response_value}",
                        marker_color=colour_map[response_value],
                        marker_line_color="black",
                        marker_line_width=0.25,
                        xbins=dict(start=bin_start, end=bin_end, size=bin_size),
                        showlegend=False,
                    ),
                    row=row_num,
                    col=1,
                )

            # Right column: Percentage histogram (stacked)
            binned_data = pd.cut(
                data,
                bins=np.linspace(bin_start, bin_end, n_bins + 1),
                include_lowest=True,
            )

            grouped_data = (
                pd.DataFrame({"bin": binned_data, legend_column: df[legend_column]})
                .groupby(["bin", legend_column], observed=False)
                .size()
                .unstack(fill_value=0)
            )

            percentages = grouped_data.apply(lambda row: row / row.sum() * 100, axis=1)

            for response_value in df[legend_column].unique():
                fig.add_trace(
                    go.Bar(
                        x=percentages.index.astype(str),
                        y=percentages[response_value],
                        name=f"{legend_column} = {response_value}",
                        marker_color=colour_map[response_value],
                        marker_line_color="black",
                        marker_line_width=0.25,
                        showlegend=False,
                    ),
                    row=row_num,
                    col=2,
                )

        # Update y-axis titles for both subplots
        fig.update_yaxes(title_text="Count", row=row_num, col=1, showgrid=False)
        fig.update_yaxes(title_text="Percentage", row=row_num, col=2, showgrid=False)

    fig.update_layout(
        showlegend=True,
        height=400 * n_rows,
        width=1200, # consider adjusting this depending on the size of the screen
        title={
            "text": f"""
            <span style='color: {colour_map[0]};'>{legend_column}=0</span> <span style='color: {colour_map[1]};'>{legend_column}=1</span>
            """,
            "x": 0.5,
            "y": 0.995,
            "xanchor": "center",
            "yanchor": "top",
        },
        barmode="stack",
        template="plotly_white"
    )

    return fig


import pandas as pd
from typing import List

def show_defaults_in_missing_values(df: pd.DataFrame, target_column: str = 'DEFAULT') -> pd.DataFrame:
    """
    Creates a summary table for a DataFrame, displaying:
    - Column names
    - Percentage of defaults (overall)
    - Number of missing values per column
    - Percentage of defaults among rows with missing values for each column.

    Args:
        df (pd.DataFrame): The input DataFrame containing data to analyze.
        target_column (str): The name of the column representing default status (1 for DEFAULT, 0 for NOT DEFAULT).

    Returns:
        pd.DataFrame: A summary table with columns:
            - 'Column Name': Name of each column in the DataFrame (excluding the default column).
            - 'Percentage of Defaults': Overall percentage of defaults in the dataset.
            - 'Number of Missing Values': Count of NaN values in each column.
            - 'Percentage of Defaults in Missing': Default percentage for rows with NaN in the respective column.
    """
    table_data: List[List] = []

    for col in df.columns:
        if col == target_column:
            continue

        total_defaults = round(df[target_column].mean() * 100, 2)

        missing_count = df[col].isna().sum()

        missing_defaults = round(df[df[col].isna()][target_column].mean() * 100, 2)

        table_data.append([col, total_defaults, missing_count, missing_defaults])

    result_df = pd.DataFrame(
        table_data,
        columns=['Column Name', 'Percentage of Defaults', 'Number of Missing Values', 'Percentage of Defaults in Missing']
    )
    
    result_df = result_df.sort_values(by='Percentage of Defaults in Missing', ascending=False)
        
    return result_df
