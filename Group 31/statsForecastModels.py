from statsforecast import StatsForecast
import matplotlib.pyplot as plt
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoCES,
    AutoTheta,
    AutoRegressive,
    SeasonalExponentialSmoothingOptimized,
    Holt,
    HoltWinters,
    SeasonalWindowAverage,
    RandomWalkWithDrift,
    SeasonalNaive,
    Naive
)

from tqdm.autonotebook import tqdm
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utilsforecast.losses import smape, mape, mse, rmse, mae, rmae, mase
from utilsforecast.evaluation import evaluate
from pandas import DataFrame
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


MDL_PARAMETERS = {
    "AutoARIMA": {"season_length": 1},
    "AutoETS": {"season_length": 1},
    "AutoCES": {"season_length": 1},
    "AutoTheta": {"season_length": 1},
    "AutoRegressive": {"lags": 1,"include_drift":True},
    "SeasonalExponentialSmoothingOptimized": {"season_length": 1},
    "Holt": {"season_length": 1, "alias":"Holt"},
    "HoltWinters": {"season_length": 1, "alias":'HoltWinters'},
    "SeasonalNaive": {"season_length": 1},
    "SeasonalWindowAverage": {"season_length": 1, 'window_size':1}, 
}


MDL_RENAME_MAPPING = {
    "RWD": "RandomWalkWithDrift",
    "CES": "AutoCES",
    "SeasWA": "SeasonalWindowAverage",
    "SES": "SimpleExponentialSmoothing",
    "SESOpt": "SimpleExponentialSmoothingOptimized",
    "SeasonalES":"SeasonalExponentialSmoothing",
    "SeasESOpt": "SeasonalExponentialSmoothingOptimized"
}


# List of statistical forecasting models supported in the module
STATS_MDL = [
    "AutoARIMA",
    "AutoETS",
    "AutoCES",
    "AutoTheta",
    "AutoRegressive",
    "SeasonalExponentialSmoothingOptimized",
    "Holt",
    "HoltWinters",
    "SeasonalNaive",
    "SeasonalWindowAverage",
    "RandomWalkWithDrift",
]

# Mapping ensemble model groups to their respective models
ENSEMBLE_MAPPING = {
    "stats_ensemble": STATS_MDL
}

# Model speed mapping categorizing models by performance characteristics
MDL_SPEED_MAPPING = {
    "all": STATS_MDL,
    "test": STATS_MDL
}

class StatsForecastModels:
    def __init__(self, model_type: list, seasonal_length: int, freq: str, date_format: str) -> None:
        """
        Initialize the StatsForecastModels class with the specified parameters.

        Args:
            model_type (list): List of model types to use.
            seasonal_length (int): Length of the seasonal cycle.
            freq (str): Frequency of the time series data.
            date_format (str): Date format for parsing the input data.
        """
        # Select models based on provided model_type and mapping
        self.models = set(
            [mdl for key in model_type for mdl in MDL_SPEED_MAPPING.get(key, [])]
        )
        print(self.models)

        # Adjust the model set based on the seasonal length
        if seasonal_length > 1:
            self.models.remove('Holt')  # Remove non-seasonal model
        else:
            self.models.remove('HoltWinters')  # Remove seasonal model
        print(self.models)

        # Validate that all selected models are supported
        if len(set(self.models) - set(STATS_MDL)) != 0:
            mdl_np = ",".join(list(set(model_type) - set(STATS_MDL)))
            raise Exception(mdl_np, "models are not present in module")

        # Store additional parameters
        self.seasonal_length = seasonal_length
        self.freq = freq
        self.date_format = date_format
        self.model_pos = None
        self.ignore_neg_fcsts = False

    def set_mdl_param(self):
        """
        Set parameters for each selected model based on seasonal length and training data characteristics.
        """
        if self.seasonal_length is not None:
            for mdl_name, mdl_params in MDL_PARAMETERS.items():
                # Update season_length parameter
                if "season_length" in mdl_params.keys():
                    mdl_params["season_length"] = self.seasonal_length
                
                # Update window_size parameter dynamically based on the training data size
                if "window_size" in mdl_params.keys():
                    window_size = int((self.train_df.ds.nunique() * 0.20) / self.seasonal_length)
                    mdl_params["window_size"] = max(1, window_size)
                
                # Update lags parameter for models using lagged features
                if "lags" in mdl_params.keys():
                    mdl_params["lags"] = self.seasonal_length
        
    def data_validation(self):
        """
        Validate the training data and ensure correct date format.
        """
        try:
            # Mapping of date format strings to pandas-compatible format codes
            date_mapping = {
                "DD/MM/YYYY": "%d/%m/%Y",
                "MM/DD/YYYY": "%m/%d/%Y",
                "YYYY/MM/DD": "%Y/%m/%d",
                "DD-MM-YYYY": "%d-%m-%Y",
                "MM-DD-YYYY": "%m-%d-%Y",
                "YYYY-MM-DD": "%Y-%m-%d",
            }
            
            # Convert date column to datetime format
            self.train_df["ds"] = pd.to_datetime(
                self.train_df["ds"], format=date_mapping[self.date_format]
            )
            print(date_mapping[self.date_format], "----------------------------")
        except KeyError as e:
            print(f"Error: Invalid date format: {self.date_format}")
            return
        
        # Validate the format of the training data
        self.format_validation()
        return None

    def format_validation(self):
        """
        Placeholder for additional format validation logic.
        """
        return None

    def fit(self, train_df: pd.DataFrame):
        """
        Fit the selected forecasting models to the provided training data.

        Args:
            train_df (pd.DataFrame): Training data with columns ['ds', 'y'].
        """
        self.train_df = train_df
        self.set_mdl_param()

        try:
            # Validate the training data
            self.data_validation()
            
            # Instantiate and store the selected models
            selected_models = []
            for model_name in self.models:
                if model_name in MDL_PARAMETERS.keys():
                    selected_models.append(
                        globals()[model_name](**MDL_PARAMETERS[model_name])
                    )
                else:
                    selected_models.append(globals()[model_name]())
            
            # Initialize and fit the StatsForecast object
            self.sf = StatsForecast(
                models=selected_models,
                freq=self.freq,
            )
            self.sf = self.sf.fit(self.train_df)
            return self
        
        except Exception as e:
            print(f"Error during fitting: {e}")
            return None

    def predict(self, horizon: int, ignore_neg_fcsts=False):
        """
        Generate predictions using the fitted models.

        Args:
            horizon (int): Forecast horizon.
            ignore_neg_fcsts (bool): Whether to ignore negative forecasts.
        """
        self.horizon = horizon
        self.ignore_neg_fcsts = ignore_neg_fcsts

        if self.sf is None:
            raise Exception("Model not fitted yet. Call fit method first.")
        
        # Generate forecasts
        self.pred__ = (
            self.sf.predict(horizon).reset_index().rename(columns=MDL_RENAME_MAPPING)
        )

        # Add ensemble predictions
        for key_ in ENSEMBLE_MAPPING.keys():
            mdl_present_ = [
                mdl_ for mdl_ in ENSEMBLE_MAPPING[key_] if mdl_ in self.models
            ]
            if len(mdl_present_):
                self.pred__[key_] = self.pred__[mdl_present_].mean(axis=1)

        # Identify positions of non-negative forecasts
        self.model_pos = list((self.pred__[self.pred__.iloc[:, 2:].columns] >= 0).all().index)

        if ignore_neg_fcsts:
            self.pred__ = self.pred__[['unique_id', 'ds'] + self.model_pos]

        return self.pred__

    def crossvalidation(self, error_metric: str, h: Optional[int] = None):
        """
        Perform cross-validation and compute error metrics.

        Args:
            error_metric (str): Error metric function name.
            h (int, optional): Forecast horizon for cross-validation.
        """
        h = self.horizon

        # Use the specified error metric function
        error_metric = globals()[error_metric]
        crossvalidation_df = self.sf.cross_validation(h=h)

        crossvalidation_df = crossvalidation_df.reset_index().rename(
            columns=MDL_RENAME_MAPPING
        )

        # Add ensemble predictions to cross-validation results
        for key_ in ENSEMBLE_MAPPING.keys():
            mdl_present_ = [
                mdl_ for mdl_ in ENSEMBLE_MAPPING[key_] if mdl_ in self.models
            ]
            if len(mdl_present_):
                crossvalidation_df[key_] = crossvalidation_df[mdl_present_].mean(axis=1)

        # Filter out negative forecasts if specified
        if self.ignore_neg_fcsts:
            crossvalidation_df = crossvalidation_df[['unique_id', 'ds', 'cutoff', 'y'] + self.model_pos]

        return crossvalidation_df



# Function to build a forecast model based on the specified model type
def build_forecast_model(df, model_type, date_format, data_frequency, seasonal_length, forecast_horizon, ignore_neg_fcsts, error_metric, eval=True):
    """
    Builds and trains a forecast model based on the specified type (statistical).
    
    Parameters:
    - df: pandas DataFrame, the input data containing historical time series data.
    - model_type: str, type of the model to be used ('statistical').
    - date_format: str, format of the date column in the dataset.
    - data_frequency: str, frequency of the data (e.g., 'D', 'H', 'M', etc.).
    - seasonal_length: int, the number of periods that make up a seasonal cycle.
    - forecast_horizon: int, the number of periods to forecast ahead.
    - ignore_neg_fcsts: bool, flag to ignore negative forecasts.
    - error_metric: str, the error metric to use for evaluating model performance.
    - eval: bool, whether to evaluate the model performance (default is True).
    
    Returns:
    - df_pred: pandas DataFrame, the predicted forecast values.
    - df_performance: pandas DataFrame, performance metrics for the forecast.
    - df_crossval: pandas DataFrame, cross-validation results.
    - df_rank: pandas DataFrame, ranked forecasts.
    """
    
    # Check if the model type is 'statistical'
    if model_type == 'statistical':
        print('Getting results for statistical model')
        print(df.columns)  # Print the columns of the dataframe for inspection
        
        # Train the statistical model
        df_pred, df_performance, df_crossval, df_rank = train_statistical_model(
            df, seasonal_length, date_format, data_frequency, forecast_horizon, 
            ignore_neg_fcsts, error_metric, eval
        )
    
    return df_pred, df_performance, df_crossval, df_rank


# Function to train a statistical forecasting model
def train_statistical_model(df, seasonal_length, date_format, data_frequency, forecast_horizon, 
                             ignore_neg_fcsts, error_metric, eval):
    """
    Trains a statistical forecasting model using the provided data and parameters.
    
    Parameters:
    - df: pandas DataFrame, the input data.
    - seasonal_length: int, number of periods in one season.
    - date_format: str, the format of the date column.
    - data_frequency: str, frequency of the data (e.g., 'D', 'H', etc.).
    - forecast_horizon: int, the number of steps ahead to forecast.
    - ignore_neg_fcsts: bool, whether to ignore negative forecasts.
    - error_metric: str, the error metric to evaluate model performance.
    - eval: bool, whether to perform evaluation or not.
    
    Returns:
    - df_pred: pandas DataFrame, the predicted forecast values.
    - df_performance: pandas DataFrame, performance metrics for the forecast.
    - df_crossval: pandas DataFrame, cross-validation results.
    - df_rank: pandas DataFrame, ranked forecasts.
    """
    df = df.copy()  # Create a copy of the data to avoid modifying the original dataframe
    print('Under statistical model', df.columns)  # Print columns for inspection
    
    # Initialize the statistical forecasting model
    stats = StatsForecastModels(
        model_type=['all'],
        seasonal_length=seasonal_length,
        freq=data_frequency,
        date_format=date_format,
    )
    
    # Fit the model to the data
    stats.fit(df)
    
    # Generate forecast predictions
    df_pred = stats.predict(horizon=forecast_horizon, ignore_neg_fcsts=ignore_neg_fcsts)

    # If evaluation is required, perform cross-validation and performance ranking
    if eval:
        df_crossval = stats.crossvalidation(error_metric=error_metric)
        df_performance = evaluate_cross_validation(df_crossval, error_metric)
    else:
        # If evaluation is not required, return empty DataFrames for performance and ranking
        df_performance, df_rank, df_crossval = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    return df_pred, df_performance, df_crossval, pd.DataFrame()


# Function to evaluate cross-validation results and compute performance metrics
def evaluate_cross_validation(df, metric):
    """
    Evaluates the cross-validation results and computes the performance metrics for each model.
    
    Parameters:
    - df: pandas DataFrame, the cross-validation results.
    - metric: str, the error metric to use (e.g., 'rmse', 'mae').
    
    Returns:
    - evaluated: pandas DataFrame, performance metrics for each model.
    """
    error_metric = globals()[metric]  # Retrieve the error metric function dynamically
    models = df.drop(columns=["unique_id", "ds", "cutoff", "y"]).columns.tolist()  # List of models to evaluate
    evals = []  # List to store evaluation results
    
    # Perform evaluation for each cutoff in the cross-validation
    for cutoff in df["cutoff"].unique():
        eval_ = evaluate(
            df[df["cutoff"] == cutoff], metrics=[error_metric], models=models
        )
        evals.append(eval_)  # Append the evaluation result for each cutoff
    
    # Combine the evaluation results into a single DataFrame
    evaluated = pd.concat(evals)
    evaluated = evaluated.groupby("unique_id").mean(numeric_only=True)  # Average the evaluation results by unique_id

    # Insert the best model for each unique_id based on the minimum error metric
    evaluated.insert(0, "best_model", evaluated.idxmin(axis=1))
    
    return evaluated

def find_data_frequency(data, date_format, seasonal_length):
    df = data.copy(deep=True)
    df['ds'] = pd.to_datetime(df['ds'], format=date_format)
    if seasonal_length==1:
        frequency = "D"
    elif seasonal_length==7:
        frequency = "W"
    elif seasonal_length==12:
        frequency = "M"

    # Check for month start or end
    if frequency == 'M':
        if df['ds'].dt.day.eq(1).any():
            frequency = 'MS'
        if df['ds'].dt.is_month_end.any():
            frequency = 'ME'
    return frequency

def get_first_sunday_of_year(year):
    # Start from the first day of the year
    first_day_of_year = datetime(year, 1, 1)
    # Find the first Sunday by checking the weekday and adjusting accordingly
    first_sunday = first_day_of_year + timedelta(days=(6 - first_day_of_year.weekday()))
    return first_sunday
# Generate a list of Sundays starting from the first Sunday of 2023
def generate_sundays(start_date, num_weeks):
    sundays = []
    current_sunday = start_date
    for _ in range(num_weeks):
        sundays.append(current_sunday)
        current_sunday += timedelta(weeks=1)  # Move to the next Sunday
    return sundays