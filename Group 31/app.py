import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# Import your forecasting model libraries here, e.g., Prophet, ARIMA

# st.markdown("""
#     <style>
#         /* Adjust the container to minimize the margin */
#         .css-1d391kg {  /* Streamlit's default container class */
#             padding-left: 5px;
#             padding-right: 5px;
#         }
        
#         /* Adjust the columns to have a 1:2 ratio, making column1 smaller */
#         .block-container {
#             padding: 0 2rem;
#         }

#         /* Adjust the column width manually using flexbox */
#         .row-widget {
#             display: flex;
#             justify-content: space-between;
#         }
        
#         .css-1v0mbdj {
#             flex: 1;  /* Column 1 (smaller) */
#             margin-right: 1rem;  /* Add some spacing between columns */
#         }
        
#         .css-1tptv69 {
#             flex: 2;  /* Column 2 (larger) */
#         }

#         /* Remove unnecessary margins */
#         .css-16q8frb {
#             margin-left: 0px !important;
#             margin-right: 0px !important;
#         }
#     </style>
# """, unsafe_allow_html=True)

def change_style():
    """Returns a CSS style string for modernizing tabs and horizontal radio buttons."""
    return """
    <style>
        /* Style for tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 5px; /* Horizontal spacing between tabs */
        }
        .stTabs [data-baseweb="tab"] {
            height: 35px;
            border-radius: 7px; 
            padding: 10px 40px;
            font-size: 36px;
            cursor: pointer;
            color: gray;
            transition: color 0.2s ease-in-out;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: gray;
        }
        
        /* Style for horizontal radio buttons */
        .stRadio > div {
            display: flex;
            flex-direction: row;
        }
        .stRadio > div > label {
            margin-right: 20px;
        }
        .key3 .stSelectbox-container {
            width: 150px; /* Adjust the width as needed */
        }
    </style>
    """
st.set_page_config(
    page_title="Forecasting.AI",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="auto",
)

def main():
    st.title("Time Series Forecasting App with Model Evaluation")
    st.markdown(change_style(), unsafe_allow_html=True)
    # Define the tabs
    tabs = st.tabs(["Introduction", "Input", "Results", "Performance"])

    # Introduction Tab
    with tabs[0]:
        st.header("Introduction")
        st.write(
            """
            Welcome to the Time Series Forecasting App. This tool is designed to help you explore and analyze 
            time series data using various forecasting models, including ARIMA, Prophet, and LSTM. You can upload 
            your dataset, set forecasting parameters, and view model performance metrics.
            
            **Model Overview:**
            - **ARIMA**: Suitable for univariate time series data with trend and seasonality.
            - **Prophet**: A robust model developed by Facebook, particularly good for handling seasonality and holiday effects.
            - **LSTM**: A deep learning model effective for capturing long-term dependencies in time series data.
            
            Select a forecasting method and evaluate results based on key performance metrics.
            """
        )

    # Input Tab
    with tabs[1]:
        # st.set_page_config(layout="wide")
        st.header("Upload and Configure Forecasting Parameters")
        
        # Upload CSV file
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df['ds'] = pd.to_datetime(df['ds'])

            # st.write("Uploaded Data Preview:")
            # st.write(df)
            
            # # Data visualization
            # st.subheader("Data Visualization")
            # selected_id = st.selectbox("Select id to visualize", df.unique_id.unique().tolist())  
            # temp = df[df['unique_id']==selected_id]
            # fig = px.line(temp, x='ds', y='y', title="Time Series Plot")
            # st.plotly_chart(fig)
            selected_id = st.selectbox("Select unique ID to visualize", df['unique_id'].unique().tolist())
            if selected_id:
                col1, col2 = st.columns([1,2])
                with col1:
                    # Display data preview
                    st.write("Uploaded Data Preview:")
                    st.write(df)  # Display only the first few rows for a quick preview
                
                with col2:
                    # Visualization settings
                    st.subheader("Data Visualization")
                    
                    # Filter data and plot
                    temp = df[df['unique_id'] == selected_id]
                    fig = px.line(temp, x='ds', y='y', title=f"Time Series Plot for ID: {selected_id}")
                    st.plotly_chart(fig)

            # Forecasting parameters
            st.sidebar.subheader("Set Forecasting Parameters")
            forecast_horizon = st.sidebar.slider("Select Forecast Horizon", min_value=1, max_value=60, value=12)
            data_frequency = st.sidebar.selectbox("Select Data Frequency", ["Daily", "Weekly", "Monthly", "Yearly"])

            # Forecasting button
            if st.sidebar.button("Get Forecast"):
                forecast_results = generate_forecasts(df, forecast_horizon, data_frequency)
                st.session_state['forecast_results'] = forecast_results  # Store results in session state for Results tab

    # Results Tab
    with tabs[2]:
        st.header("Forecast Results")
        
        if 'forecast_results' in st.session_state:
            forecast_results = st.session_state['forecast_results']
            st.write(forecast_results)

            # Visualization of forecasts for each model
            st.subheader("Forecast Visualizations")
            visualize_forecasts(forecast_results, df)

            # Download option for forecast results
            st.subheader("Download Forecast Results")
            csv_data = convert_to_csv(forecast_results)
            st.download_button(label="Download as CSV", data=csv_data, file_name="forecast_results.csv", mime="text/csv")
        else:
            st.write("Please upload data and generate forecasts in the 'Input' tab.")

    # Performance Tab
    with tabs[3]:
        st.header("Model Performance Metrics")
        
        if 'forecast_results' in st.session_state:
            # Assuming the forecast_results DataFrame has columns with models' forecasts
            performance_metrics = calculate_performance_metrics(st.session_state['forecast_results'], df)
            st.write(performance_metrics)
        else:
            st.write("Please generate forecasts to view performance metrics.")

# Helper Functions
def generate_forecasts(df, forecast_horizon, data_frequency):
    """
    Placeholder function to generate forecasts.
    Replace this with your actual model training and forecasting.
    """
    models = ["Model_ARIMA", "Model_Prophet", "Model_LSTM"]  # Replace with actual model names
    forecast_results = pd.DataFrame()
    forecast_results[f'ds'] = pd.date_range(df['ds'].iloc[-1], periods=forecast_horizon+1, freq='MS')[1:]
    forecast_results[f'unique_id']= df['unique_id'][:forecast_horizon]
    for model in models:
        # # Generate a random forecast for demonstration; replace with actual model results
        # model_forecast = {
        #     f'unique_id_{model}': df['unique_id'][:forecast_horizon],
        #     # f'ds_{model}': df['ds'][:forecast_horizon],  #pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(1, data_frequency[0].upper()), periods=forecast_horizon, freq=data_frequency[0].upper()),
        #     f'ds_{model}': pd.date_range(df['ds'].iloc[-1], periods=forecast_horizon, freq='MS')[1:],
        #     model: np.random.randn(forecast_horizon)     #+ df[selected_column].mean()
        # }

        # forecast_results = pd.concat([forecast_results, pd.DataFrame(model_forecast)], axis=1)
        forecast_results[model] = (np.random.randn(forecast_horizon)*200).tolist()

    return forecast_results

# def visualize_forecasts(forecast_results):
#     """
#     Display line plots of forecasts in rank order.
#     """
#     print(forecast_results.head())
#     # models = forecast_results.columns[2:]  # Assuming first two columns are unique_id and ds
#     models = ["Model_ARIMA", "Model_Prophet", "Model_LSTM"]  # Replace with actual model names

#     for model in models:
#         st.write(f"Forecast Plot for {model}")
#         # temp2 = pd.concat
#         # fig = px.line(forecast_results[[f'ds_{model}', model]], x=f'ds_{model}', y=model, title=f"Forecast Plot - {model}")
#         fig = px.line(forecast_results[[f'ds', model]], x=f'ds', y=model, title=f"Forecast Plot - {model}")

#         st.plotly_chart(fig)

import pandas as pd
import plotly.express as px
import streamlit as st

def visualize_forecasts(forecast_results, df):
    """
    Display line plots of historical and forecasted values for multiple models.
    
    df: DataFrame containing historical data with 'ds' and 'y'
    forecast_results: DataFrame containing forecasted data with 'ds' and columns for model forecasts
    """
    
    # Display first few rows of forecast_results to understand its structure
    # st.write(forecast_results.head())
    
    # Models to visualize from forecast_results (replace with your actual models)
    models = ["Model_ARIMA", "Model_Prophet", "Model_LSTM"]

    for model in models:
        st.write(f"Forecast Plot for {model}")
        
        # Combine historical and forecast data for the specific model
        historical_data = df[['ds', 'y']].copy()
        historical_data.columns = ['ds', 'Historical']
        forecast_data = forecast_results[['ds', model]]
        
        # Rename the forecasted column to match the 'y' column for consistency
        forecast_data = forecast_data.rename(columns={model: 'Forecast'})
        
        # Merge the historical and forecasted data on the 'ds' (date) column
        combined_data = pd.concat([historical_data, forecast_data], ignore_index=True, sort=False)
        
        # Create the line plot with both historical and forecasted data
        fig = px.line(combined_data, x='ds', y=['Historical', 'Forecast'], 
                      labels={'ds': 'Date', 'Historical': 'Historical Data', 'Forecast': 'Forecasted Data'},
                    #   title=f"Forecast Plot - {model}"
                      )
        
        # Show the plot in Streamlit
        st.plotly_chart(fig)



def calculate_performance_metrics(forecast_results, actual_df):
    """
    Placeholder for calculating performance metrics for each model.
    """
    # Example metrics calculation; replace with actual metrics
    metrics = {'Model': [], 'MAE': [], 'RMSE': []}
    models = ["Model_ARIMA", "Model_Prophet", "Model_LSTM"]  # Replace with actual model names
    
    for model in models:
        mae = np.mean(np.abs(forecast_results[model] - actual_df['y']))  # Replace 'y' with actual values
        rmse = np.sqrt(np.mean((forecast_results[model] - actual_df['y']) ** 2))
        
        metrics['Model'].append(model)
        metrics['MAE'].append(mae)
        metrics['RMSE'].append(rmse)

    return pd.DataFrame(metrics)

def convert_to_csv(df):
    """
    Convert a DataFrame to CSV format and return it as bytes.
    """
    return df.to_csv(index=False).encode("utf-8")

if __name__ == "__main__":
    main()
