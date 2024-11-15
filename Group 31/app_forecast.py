import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import plotly.express as px
from statsForecastModels import build_forecast_model


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
            time series data using a variety of forecasting models. You can upload your dataset, set forecasting 
            parameters, and view model performance metrics.

            **Model Overview:**
            - **AutoARIMA**: A method for automatically selecting the best ARIMA model parameters (p, d, q) based on the data.
            - **AutoETS**: An automated version of Exponential Smoothing (ETS), which adapts to data trends and seasonality.
            - **AutoCES**: An automatic method for selecting the best model from a set of Candidate Exponential Smoothing (CES) models.
            - **AutoTheta**: A model combining exponential smoothing and decomposition, optimized for forecasting time series data with strong seasonality and trend.
            - **AutoRegressive**: A simple model that uses past values in a time series to predict future values, typically used for short-term forecasting.
            - **SeasonalExponentialSmoothingOptimized**: A version of Exponential Smoothing optimized for seasonal data, adapting to both trend and seasonal variations.
            - **HoltWinters**: A popular method for forecasting with trend and seasonality, often used in business applications for sales forecasting.
            - **SeasonalNaive**: A simple forecasting method that uses the value from the same season (e.g., month or quarter) in the previous year as the prediction for the current season.
            - **SeasonalWindowAverage**: A forecasting method based on averaging data points within a given seasonal window, ideal for smoother trends.
            - **RandomWalkWithDrift**: A model that assumes the forecast is simply a random walk with a constant drift, often used when trends are weak.
            - **Naive**: A basic forecasting method that uses the last observed value as the prediction for all future periods, suitable for stable, non-trending data.
            - **stats_ensemble**: An ensemble model that combines predictions from multiple individual models to improve forecast accuracy.

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
            df.sort_values(['unique_id','ds'], ignore_index=True,inplace=True)
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
                    temp.sort_values('ds',inplace=True)
                    fig = px.line(temp, x='ds', y='y', title=f"Time Series Plot for ID: {selected_id}")
                    st.plotly_chart(fig)

            # Forecasting parameters
            st.sidebar.subheader("Set Forecasting Parameters")
            forecast_horizon = st.sidebar.slider("Select Forecast Horizon", min_value=1, max_value=60, value=12)
            data_frequency   = st.sidebar.selectbox("Select Data Frequency", ["Daily", "Weekly", "Monthly"])
            date_formate     = st.sidebar.selectbox("Select Date Format",    ['DD/MM/YYYY', 'MM/DD/YYYY', 'YYYY/MM/DD', 'DD-MM-YYYY', 'MM-DD-YYYY', 'YYYY-MM-DD'])
            default_metric   = st.sidebar.selectbox("Select Performance Metrics", ['rmse', 'mape'], index=1)

            dmap = {"Daily":"D", "Weekly":"W", "Monthly":"M"}
            data_frequency = dmap[data_frequency]

            # Forecasting button
            if st.sidebar.button("Get Forecast"):
                with st.spinner('Generating forecast...'):
                    df_pred, df_performance, df_crossval, df_rank = build_forecast_model(df=df, model_type='statistical', date_format=date_formate, data_frequency=data_frequency, seasonal_length=12, forecast_horizon=forecast_horizon, ignore_neg_fcsts=False, error_metric=default_metric)
                
                st.sidebar.markdown(
                            """
                            <div style="background-color:#e0f7fa; padding: 10px; border-radius: 5px;">
                                <h3 style="color: #00695c;">Forecast Generated Successfully!</h3>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                
                st.session_state['forecast_results'] = df_pred.round()
                st.session_state['forecast_crossval']= df_crossval.round() 
                st.session_state['forecast_rank']    = df_rank  
                st.session_state['default_metric']   = default_metric  
                if default_metric == 'rmse':
                    st.session_state['forecast_perform'] = df_performance.round()
                else:
                    st.session_state['forecast_perform'] = df_performance.round(4)





    # Results Tab
    with tabs[2]:
        st.header("Forecast Results")
        
        if 'forecast_results' in st.session_state:
            selected_id = st.selectbox("Select unique ID to visualize", df['unique_id'].unique().tolist(),key='id viszual')
            if selected_id:
                forecast_results = st.session_state['forecast_results']
                temp_for = forecast_results[forecast_results['unique_id'] == selected_id]
                st.write(temp_for)

                # Visualization of forecasts for each model
                temp_ori = df[df['unique_id'] == selected_id]
                st.subheader("Forecast Visualizations")
                visualize_forecasts(temp_for, temp_ori)

                # Download option for forecast results
                st.subheader("Download Forecast Results")
                csv_data = convert_to_csv(forecast_results)
                st.download_button(label="Download as CSV", data=csv_data, file_name="forecast_results.csv", mime="text/csv")
        else:
            st.write("Please upload data and generate forecasts in the 'Input' tab.")

    # Performance Tab
    with tabs[3]:
        
        if 'forecast_results' in st.session_state:
            default_metric = st.session_state['default_metric'] 
            st.header(f"Model Performance Metrics : {default_metric}")
            # Assuming the forecast_results DataFrame has columns with models' forecasts
            performance_metrics = st.session_state['forecast_perform']
            st.write(performance_metrics)
        else:
            st.write("Please generate forecasts to view performance metrics.")





def visualize_forecasts(forecast_results, df):
    """
    Display line plots of historical and forecasted values for multiple models.
    
    df: DataFrame containing historical data with 'ds' and 'y'
    forecast_results: DataFrame containing forecasted data with 'ds' and columns for model forecasts
    """
    
    # Display first few rows of forecast_results to understand its structure
    # st.write(forecast_results.head())
    
    # Models to visualize from forecast_results (replace with your actual models)
    models = ['AutoARIMA', 'AutoETS', 'AutoCES', 'AutoTheta', 'AutoRegressive', 'SeasonalExponentialSmoothingOptimized', 'HoltWinters', 'SeasonalNaive', 'SeasonalWindowAverage', 'RandomWalkWithDrift', 'Naive', 'stats_ensemble']

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



def convert_to_csv(df):
    """
    Convert a DataFrame to CSV format and return it as bytes.
    """
    return df.to_csv(index=False).encode("utf-8")

if __name__ == "__main__":
    main()
