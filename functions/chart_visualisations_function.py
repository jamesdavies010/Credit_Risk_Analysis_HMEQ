import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from functions import calculate_bins

# def chart_visualisations(
#     df: pd.DataFrame,
#     legend_column: str = "DEFAULT",
#     n_cols: int = 2,
#     specified_columns: dict = None,
# ) -> None:

#     df = df.copy()

#     if specified_columns is None:
#         specified_columns = {"Discrete": [], "Category": [], "Continuous": []}

#         for col in df.columns:
#             if col != legend_column:
#                 if isinstance(
#                     df[col].dtype, pd.CategoricalDtype
#                 ) or pd.api.types.is_object_dtype(df[col]):
#                     specified_columns["Category"].append(col)
#                 elif pd.api.types.is_numeric_dtype(df[col]):
#                     specified_columns["Continuous"].append(col)

#     relevant_columns = [
#         col
#         for col in df.columns
#         if col in [c for group in specified_columns.values() for c in group]
#     ]

#     n_rows = len(relevant_columns)

#     subplot_titles = []
#     for col in relevant_columns:
#         subplot_titles.extend([f"{col}", f"{col}"])

#     fig = make_subplots(
#         rows=n_rows,
#         cols=n_cols,  # Always 2 columns - left for count, right for percentage
#         subplot_titles=subplot_titles,
#     )

#     colour_map = {0: "rgb(31, 119, 180)", 1: "rgb(255, 127, 14)"}  # Blue and orange

#     for idx, col in enumerate(relevant_columns):
#         row_num = idx + 1

#         is_discrete_or_category = (
#             col in specified_columns["Discrete"] or col in specified_columns["Category"]
#         )

#         # Determine bar width based on column type
#         bar_width = 0.7 if is_discrete_or_category else 1.0

#         # Check the column type and plot accordingly
#         if is_discrete_or_category:
#             data = df[[col, legend_column]].dropna()
#             grouped_data = (
#                 data.groupby([col, legend_column], observed=False)
#                 .size()
#                 .unstack(fill_value=0)
#             )
#             percentages = grouped_data.apply(lambda row: row / row.sum() * 100, axis=1)

#             for response_value in df[legend_column].unique():
#                 fig.add_trace(
#                     go.Bar(
#                         x=grouped_data.index.astype(str),
#                         y=grouped_data[response_value],
#                         name=f"{legend_column} = {response_value}",
#                         marker_color=colour_map[response_value],
#                         marker_line_color="black",
#                         marker_line_width=0.25,
#                         showlegend=False,
#                         width=bar_width,  # Adjust bar width
#                     ),
#                     row=row_num,
#                     col=1,
#                 )
#                 fig.add_trace(
#                     go.Bar(
#                         x=percentages.index.astype(str),
#                         y=percentages[response_value],
#                         name=f"{legend_column} = {response_value}",
#                         marker_color=colour_map[response_value],
#                         marker_line_color="black",
#                         marker_line_width=0.25,
#                         showlegend=False,
#                         width=bar_width,  # Adjust bar width
#                     ),
#                     row=row_num,
#                     col=2,
#                 )

#         else:
#             # Continuous column: Histogram
#             data = df[col].dropna()
#             bin_start, bin_end, bin_size, n_bins = calculate_bins(data)

#             for response_value in df[legend_column].unique():
#                 data_filtered = df[df[legend_column] == response_value][col].dropna()

#                 fig.add_trace(
#                     go.Histogram(
#                         x=data_filtered,
#                         name=f"{legend_column} = {response_value}",
#                         marker_color=colour_map[response_value],
#                         marker_line_color="black",
#                         marker_line_width=0.25,
#                         xbins=dict(start=bin_start, end=bin_end, size=bin_size),
#                         showlegend=False,
#                     ),
#                     row=row_num,
#                     col=1,
#                 )

#             # Right column: Percentage histogram (stacked)
#             binned_data = pd.cut(
#                 data,
#                 bins=np.linspace(bin_start, bin_end, n_bins + 1),
#                 include_lowest=True,
#             )

#             grouped_data = (
#                 pd.DataFrame({"bin": binned_data, legend_column: df[legend_column]})
#                 .groupby(["bin", legend_column], observed=False)
#                 .size()
#                 .unstack(fill_value=0)
#             )

#             percentages = grouped_data.apply(lambda row: row / row.sum() * 100, axis=1)

#             for response_value in df[legend_column].unique():
#                 fig.add_trace(
#                     go.Bar(
#                         x=percentages.index.astype(str),
#                         y=percentages[response_value],
#                         name=f"{legend_column} = {response_value}",
#                         marker_color=colour_map[response_value],
#                         marker_line_color="black",
#                         marker_line_width=0.25,
#                         showlegend=False,
#                     ),
#                     row=row_num,
#                     col=2,
#                 )

#         # Update y-axis titles for both subplots
#         fig.update_yaxes(title_text="Count", row=row_num, col=1, showgrid=False)
#         fig.update_yaxes(title_text="Percentage", row=row_num, col=2, showgrid=False)

#     fig.update_layout(
#         showlegend=True,
#         height=500 * n_rows,  # Adjust height based on number of features
#         width=1500,
#         title={
#             "text": f"""
#             <span style='color: {colour_map[0]};'>{legend_column}=0</span> <span style='color: {colour_map[1]};'>{legend_column}=1</span>
#             """,
#             "x": 0.5,
#             "y": 0.995,
#             "xanchor": "center",
#             "yanchor": "top",
#         },
#         barmode="stack",
#         template="plotly_white"
#     )

#     fig.show()


# def chart_visualisations(
#     df: pd.DataFrame,
#     legend_column: str = "DEFAULT",
#     n_cols: int = 2,
#     columns_by_type: dict = None,  # Renamed parameter
#     specific_columns: list = None,  # Added parameter
# ) -> None:

#     df = df.copy()

#     if specific_columns is not None:
#         # If specific columns are provided, use them directly
#         relevant_columns = [col for col in specific_columns if col in df.columns]
#     else:
#         if columns_by_type is None:  # Updated name
#             columns_by_type = {"Discrete": [], "Category": [], "Continuous": []}  # Updated name

#             for col in df.columns:
#                 if col != legend_column:
#                     if isinstance(
#                         df[col].dtype, pd.CategoricalDtype
#                     ) or pd.api.types.is_object_dtype(df[col]):
#                         columns_by_type["Category"].append(col)  # Updated name
#                     elif pd.api.types.is_numeric_dtype(df[col]):
#                         columns_by_type["Continuous"].append(col)  # Updated name

#         # Derive relevant columns from columns_by_type
#         relevant_columns = [
#             col
#             for col in df.columns
#             if col in [c for group in columns_by_type.values() for c in group]  # Updated name
#         ]

#     n_rows = len(relevant_columns)

#     subplot_titles = []
#     for col in relevant_columns:
#         subplot_titles.extend([f"{col}", f"{col}"])

#     fig = make_subplots(
#         rows=n_rows,
#         cols=n_cols,  # Always 2 columns - left for count, right for percentage
#         subplot_titles=subplot_titles,
#     )

#     colour_map = {0: "rgb(31, 119, 180)", 1: "rgb(255, 127, 14)"}  # Blue and orange

#     for idx, col in enumerate(relevant_columns):
#         row_num = idx + 1

#         is_discrete_or_category = (
#             col in columns_by_type["Discrete"] or col in columns_by_type["Category"]  # Updated name
#         ) if specific_columns is None else False

#         # Determine bar width based on column type
#         bar_width = 0.7 if is_discrete_or_category else 1.0

#         # Check the column type and plot accordingly
#         if is_discrete_or_category:
#             data = df[[col, legend_column]].dropna()
#             grouped_data = (
#                 data.groupby([col, legend_column], observed=False)
#                 .size()
#                 .unstack(fill_value=0)
#             )
#             percentages = grouped_data.apply(lambda row: row / row.sum() * 100, axis=1)

#             for response_value in df[legend_column].unique():
#                 fig.add_trace(
#                     go.Bar(
#                         x=grouped_data.index.astype(str),
#                         y=grouped_data[response_value],
#                         name=f"{legend_column} = {response_value}",
#                         marker_color=colour_map[response_value],
#                         marker_line_color="black",
#                         marker_line_width=0.25,
#                         showlegend=False,
#                         width=bar_width,  # Adjust bar width
#                     ),
#                     row=row_num,
#                     col=1,
#                 )
#                 fig.add_trace(
#                     go.Bar(
#                         x=percentages.index.astype(str),
#                         y=percentages[response_value],
#                         name=f"{legend_column} = {response_value}",
#                         marker_color=colour_map[response_value],
#                         marker_line_color="black",
#                         marker_line_width=0.25,
#                         showlegend=False,
#                         width=bar_width,  # Adjust bar width
#                     ),
#                     row=row_num,
#                     col=2,
#                 )

#         else:
#             # Continuous column: Histogram
#             data = df[col].dropna()
#             bin_start, bin_end, bin_size, n_bins = calculate_bins(data)

#             for response_value in df[legend_column].unique():
#                 data_filtered = df[df[legend_column] == response_value][col].dropna()

#                 fig.add_trace(
#                     go.Histogram(
#                         x=data_filtered,
#                         name=f"{legend_column} = {response_value}",
#                         marker_color=colour_map[response_value],
#                         marker_line_color="black",
#                         marker_line_width=0.25,
#                         xbins=dict(start=bin_start, end=bin_end, size=bin_size),
#                         showlegend=False,
#                     ),
#                     row=row_num,
#                     col=1,
#                 )

#             # Right column: Percentage histogram (stacked)
#             binned_data = pd.cut(
#                 data,
#                 bins=np.linspace(bin_start, bin_end, n_bins + 1),
#                 include_lowest=True,
#             )

#             grouped_data = (
#                 pd.DataFrame({"bin": binned_data, legend_column: df[legend_column]})
#                 .groupby(["bin", legend_column], observed=False)
#                 .size()
#                 .unstack(fill_value=0)
#             )

#             percentages = grouped_data.apply(lambda row: row / row.sum() * 100, axis=1)

#             for response_value in df[legend_column].unique():
#                 fig.add_trace(
#                     go.Bar(
#                         x=percentages.index.astype(str),
#                         y=percentages[response_value],
#                         name=f"{legend_column} = {response_value}",
#                         marker_color=colour_map[response_value],
#                         marker_line_color="black",
#                         marker_line_width=0.25,
#                         showlegend=False,
#                     ),
#                     row=row_num,
#                     col=2,
#                 )

#         # Update y-axis titles for both subplots
#         fig.update_yaxes(title_text="Count", row=row_num, col=1, showgrid=False)
#         fig.update_yaxes(title_text="Percentage", row=row_num, col=2, showgrid=False)

#     fig.update_layout(
#         showlegend=True,
#         height=500 * n_rows,  # Adjust height based on number of features
#         width=1500,
#         title={
#             "text": f"""
#             <span style='color: {colour_map[0]};'>{legend_column}=0</span> <span style='color: {colour_map[1]};'>{legend_column}=1</span>
#             """,
#             "x": 0.5,
#             "y": 0.995,
#             "xanchor": "center",
#             "yanchor": "top",
#         },
#         barmode="stack",
#         template="plotly_white"
#     )

#     fig.show()
