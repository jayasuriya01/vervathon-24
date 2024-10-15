import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import torch.nn as nn

# Streamlit app
st.title("Smart Finance Tracking System")
st.write("This app helps track financial metrics and predict future economic outputs.")

# Load Excel Files
@st.cache_data
def load_data(file_paths):
    data_frames = []
    for file_path in file_paths:
        try:
            df = pd.read_excel(file_path)
            data_frames.append(df)
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
    return data_frames

# Sample Files - Update with actual file paths
file_paths = [
    "D:/dataset_xlsx/Economic_Output_Filled.xlsx",
    "D:/dataset_xlsx/Infrastructure_Asset_Investments.xls",
    "D:/dataset_xlsx/Operational_Costs.xlsx",
    "D:/dataset_xlsx/Internal_Tax_Revenue.xls",
    "D:/dataset_xlsx/Overall_Fiscal_Shortfall.xlsx",
    "D:/dataset_xlsx/Total_State_Spending.xlsx",
    "D:/dataset_xlsx/Public_Services_Funding.xlsx"
]

# Load the data
data_frames = load_data(file_paths)

# Processing each dataset
def process_file(df, name):
    if 'State' not in df.columns:
        
        return None
    df_melted = df.melt(id_vars=['State'], var_name='Year', value_name=name)
    df_melted['Year'] = df_melted['Year'].astype(str)  # Ensure year is treated as string for merging
    df_melted['Year'] = df_melted['Year'].apply(lambda x: int(x.split('-')[0]))  # Convert Year to int
    return df_melted

# Create a unique column name for each dataset
dataset_names = [
    'Economic_Output', 'Infrastructure_Asset_Investments', 'Operational_Costs',
    'Internal_Tax_Revenue', 'Overall_Fiscal_Shortfall', 'Total_State_Spending', 'Public_Services_Funding'
]

# Process all datasets, filter out None values
processed_data = [process_file(df, name) for df, name in zip(data_frames, dataset_names) if process_file(df, name) is not None]

# Merging all datasets on 'State' and 'Year'
def merge_datasets(processed_data):
    merged_df = processed_data[0]
    for df in processed_data[1:]:
        merged_df = pd.merge(merged_df, df, on=['State', 'Year'], how='outer', suffixes=('', '_dup'))
    return merged_df

# Merge all processed datasets
if processed_data:
    final_df = merge_datasets(processed_data)

    # Display a preview of the merged data
    st.write("Preview of Merged Data:")
    st.dataframe(final_df.head(10))

    # Group by state and year for high-level analysis
    def yearly_summary(final_df):
        summary_df = final_df.groupby(['State', 'Year']).sum().reset_index()
        return summary_df

    # Generate summary report
    summary_df = yearly_summary(final_df)
    st.write("Summary Data:")
    st.dataframe(summary_df)

    # Plotting a basic overview of financial health by state over time
    def plot_summary(summary_df):
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=summary_df, x='Year', y='Economic_Output', hue='State', marker='o')
        plt.title('Yearly Financial Summary by State')
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Visualize the summary
    st.write("Financial Overview Plot:")
    plot_summary(summary_df)

    # Linear Regression Model for Prediction
    def predict_future_values(df, state, target_column='Economic_Output', future_years=[2024, 2025,2026,2027,2028,2029,2030]):
        state_data = df[df['State'] == state][['Year', target_column]].dropna()

        # Features and Target
        X = state_data[['Year']]
        y = state_data[target_column]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=50)

        # Linear Regression Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)
        st.write(f"Model Performance for {state}:")
        st.write(f"R-squared: {r2_score(y_test, y_pred):.2f}")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

        # Predict future values
        future_years_df = pd.DataFrame(future_years, columns=['Year'])
        future_preds = model.predict(future_years_df)

        # Show predictions
        future_df = pd.DataFrame({
            'Year': future_years_df['Year'],
            f'{target_column}_Prediction': future_preds
        })
        st.write(f"\nPredictions for {state}:\n", future_df)

        # Plot predictions
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Year', y=target_column, data=state_data, label='Historical Data')
        sns.lineplot(x='Year', y=f'{target_column}_Prediction', data=future_df, label='Predictions', linestyle='--')
        plt.title(f"{target_column} Predictions for {state}")
        st.pyplot(plt)

    # Example: Predict Economic Output for a state
    state = st.selectbox('Select a state to predict', final_df['State'].unique())
    if st.button('Predict Economic Output'):
        predict_future_values(summary_df, state, 'Economic_Output', future_years=[2024, 2025,2026,2027,2028,2029,2030])

else:
    st.error("No valid data to process.")
