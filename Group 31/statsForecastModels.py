from statsforecast import StatsForecast
import matplotlib.pyplot as plt
import argparse
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
from utilsforecast.losses import smape, mape, mse, rmse, mae, rmae, mase
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utilsforecast.losses import smape, mape, mse, rmse, mae, rmae, mase
from utilsforecast.evaluation import evaluate
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Optional
import warnings
import cProfile

warnings.filterwarnings("ignore")

from statsforecast.utils import ConformalIntervals



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

ENSEMBLE_MAPPING = {
    "stats_ensemble": STATS_MDL
}

MDL_SPEED_MAPPING = {
    
    "all": STATS_MDL,
    
    "test": STATS_MDL
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





class StatsForecastModels:
    def __init__(
        self, seasonal_length: int, freq: str, date_format: str
    ) -> None:

        self.models = [
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

        if( seasonal_length > 1):
            self.models.remove('Holt')
        else:
            self.models.remove('HoltWinters')


#         if len(set(self.models) - set(STATS_MDL)) != 0:
#             mdl_np = ",".join(list(set(model_type) - set(STATS_MDL)))
#             raise Exception(mdl_np, "models are not present in module")

        self.seasonal_length = seasonal_length
        self.freq = freq
        self.date_format = date_format
        self.model_pos = None
        self.ignore_neg_fcsts = False

    def set_mdl_param(self):
        if self.seasonal_length is not None:
            for mdl_name, mdl_params in MDL_PARAMETERS.items():
                if "season_length" in mdl_params.keys():
                    mdl_params["season_length"] = self.seasonal_length
                if "window_size" in mdl_params.keys():
                    window_size = int( (self.train_df.ds.nunique()*0.20) / self.seasonal_length )
                    if(window_size<=1):
                        mdl_params['window_size'] = 1
                    else:
                        mdl_params['window_size'] = window_size
                        
                if "lags" in mdl_params.keys():
                    mdl_params["lags"] = self.seasonal_length
        
    def data_validation(self):
        try:
#             date_mapping = {"MM/DD/YYYY": "%m-%d-%Y", "DD/MM/YYYY": "%d-%m-%Y", "DD-MM-YYYY": "%d-%m-%Y", "MM-DD-YYYY": "%m-%d-%Y", "YYYY/MM/DD": "%Y-%m-%d"}
            date_mapping = {"DD/MM/YYYY": "%d/%m/%Y",
                            "MM/DD/YYYY": "%m/%d/%Y",
                            "YYYY/MM/DD": "%Y/%m/%d",
                            "DD-MM-YYYY": "%d-%m-%Y",
                            "MM-DD-YYYY": "%m-%d-%Y",
                            "YYYY-MM-DD": "%Y-%m-%d",}
    
            self.train_df["ds"] = pd.to_datetime(
                self.train_df["ds"], format=date_mapping[self.date_format]
            )
        except KeyError as e:
            print(f"Error: Invalid date format: {self.date_format}")
            return
        
        self.format_validation()
        return None

    def format_validation(self):
        return None

    def fit(self, train_df: pd.DataFrame):
        self.train_df = train_df
        
        if((self.freq=='M') | (self.freq=='MS') | (self.freq=="D")):
            self.ids_bucket_1 = list(( train_df.groupby('unique_id')['ds'].count()[(train_df.groupby('unique_id')['ds'].count()<=27)] ).index)
            self.ids_bucket_2 = list(( train_df.groupby('unique_id')['ds'].count()[(train_df.groupby('unique_id')['ds'].count()<=35) & (train_df.groupby('unique_id')['ds'].count()>=28)] ).index)
            self.ids_bucket_3 = list(( train_df.groupby('unique_id')['ds'].count()[(train_df.groupby('unique_id')['ds'].count()>=36)] ).index)

        if(self.freq=='W'):
            self.ids_bucket_1 = list(( train_df.groupby('unique_id')['ds'].count()[(train_df.groupby('unique_id')['ds'].count()<=12) & (train_df.groupby('unique_id')['ds'].count()>=4)] ).index)
            self.ids_bucket_2 = list(( train_df.groupby('unique_id')['ds'].count()[ (train_df.groupby('unique_id')['ds'].count()>=13) ] ).index)
            self.ids_bucket_3 = list([])
            
        
        self.set_mdl_param()
        
        try:
            self.data_validation()
            selected_models = []
            for model_name in self.models:
                if model_name in MDL_PARAMETERS.keys():
                    selected_models.append(
                        globals()[model_name](**MDL_PARAMETERS[model_name])
                    )
                else:
                    selected_models.append(globals()[model_name]())
                                        
            sf_naive = StatsForecast( 
                models=[Naive()],
                freq=self.freq
                )

            self.sf_naive = sf_naive.fit(self.train_df)

            if(len(self.ids_bucket_1)>0):
                
                sf = StatsForecast(
                models=selected_models,
                freq=self.freq,
                fallback_model=Naive()
                )
                self.sf_bucket_1 = sf.fit(self.train_df[self.train_df['unique_id'].isin(self.ids_bucket_1)].reset_index(drop=True))
                
            if(len(self.ids_bucket_2)>0):
            
                sf = StatsForecast(
                models=selected_models,
                freq=self.freq,
                fallback_model=Naive()
                )
                self.sf_bucket_2 =sf.fit(self.train_df[self.train_df['unique_id'].isin(self.ids_bucket_2)].reset_index(drop=True))
            
            if(len(self.ids_bucket_3)>0):
                
                sf = StatsForecast(
                models=selected_models,
                freq=self.freq,
                fallback_model=Naive()
                )
                self.sf_bucket_3 = sf.fit(self.train_df[self.train_df['unique_id'].isin(self.ids_bucket_3)].reset_index(drop=True))

                            
            return self
        
        except Exception as e:
            print(f"Error during fitting: {e}")
            return None

    def predict(self, horizon: int,ignore_neg_fcsts = False):
        
        self.horizon = horizon
        self.ignore_neg_fcsts = ignore_neg_fcsts
        
        self.pred__ = pd.DataFrame()
        
        try:
            
            if(len(self.ids_bucket_1)>0):
                self.pred_bucket_1 = self.sf_bucket_1.predict(horizon).rename(columns=MDL_RENAME_MAPPING) 
                
                mdl_rename = list(set(self.models) - set(self.pred_bucket_1.columns))
                
                if(len(mdl_rename)==1):
                    self.pred_bucket_1 = self.pred_bucket_1.rename(columns={'Naive':mdl_rename[0]})
                                        
                self.pred__ = pd.concat([self.pred__,self.pred_bucket_1])
    
                
            if(len(self.ids_bucket_2)>0):
                self.pred_bucket_2 = self.sf_bucket_2.predict(horizon).rename(columns=MDL_RENAME_MAPPING) 
                
                mdl_rename = list(set(self.models) - set(self.pred_bucket_2.columns))
                
                if(len(mdl_rename)==1):
                    self.pred_bucket_2 = self.pred_bucket_2.rename(columns={'Naive':mdl_rename[0]})
                        
                self.pred__ = pd.concat([self.pred__,self.pred_bucket_2])
            
                        
            if(len(self.ids_bucket_3)>0):
                self.pred_bucket_3 = self.sf_bucket_3.predict(horizon).rename(columns=MDL_RENAME_MAPPING)
                
                mdl_rename = list(set(self.models) - set(self.pred_bucket_3.columns))
                
                if(len(mdl_rename)==1):
                    self.pred_bucket_3 = self.pred_bucket_3.rename(columns={'Naive':mdl_rename[0]})
                        
                self.pred__ = pd.concat([self.pred__,self.pred_bucket_3])
            

            self.pred_naive = self.sf_naive.predict(horizon)

            self.pred__ = self.pred__.merge(self.pred_naive,on=['unique_id','ds'],how='left').reset_index()


            # model_columns = self.pred__.columns[2:-1].values
            
            def replace_with_nan(group):
                for model in self.models:
                    if( (group[model]!=group['Naive']).sum() == 0 ):
                        group[model] = np.nan
                return group

            self.pred__ = self.pred__.groupby('unique_id').apply(replace_with_nan).reset_index(drop=True)

            
        except:
            raise Exception("Model not fitted yet. Call fit method first.")
            
        self.pred__['stats_ensemble'] = self.pred__[self.models].mean(axis=1).drop(columns='Naive')

        self.model_pos = list((self.pred__[self.pred__.iloc[:,2:].columns] >= 0).all().index)

        if(ignore_neg_fcsts == True):
            self.pred__= self.pred__[['unique_id','ds'] + self.model_pos]

        return self.pred__
    

    def crossvalidation(
        self, error_metric: str, h: Optional[int] = None
    ):
        
        crossvalidation_df = pd.DataFrame()

        def replace_with_nan(group):
                for model in self.models:
                    if( (group[model]!=group['Naive']).sum() == 0 ):
                        group[model] = np.nan
                return group
            
        if(len(self.ids_bucket_2)>0):
            
            df_f = self.train_df[self.train_df['unique_id'].isin(self.ids_bucket_2)].reset_index(drop=True)
            
            if((self.freq=='M') | (self.freq=='MS')):
                h = df_f.groupby('unique_id')['ds'].count().min() - 25
            if(self.freq=='W'):
                h = 4

            sf_naive_ = StatsForecast( 
                models=[Naive()],
                freq=self.freq,
                verbose=True)
            
            df_cross_naive = sf_naive_.cross_validation(df=self.train_df[self.train_df['unique_id'].isin(self.ids_bucket_2)].reset_index(drop=True) , h=h).reset_index()

            crossvalidation_df_bucket_2 = self.sf_bucket_2.cross_validation(h=h)

            crossvalidation_df_bucket_2 = crossvalidation_df_bucket_2.reset_index().rename(
                columns=MDL_RENAME_MAPPING
            )

            crossvalidation_df_bucket_2 = crossvalidation_df_bucket_2.merge(df_cross_naive[['unique_id','ds','Naive']],on=['unique_id','ds'],how='left')

            crossvalidation_df_bucket_2 = crossvalidation_df_bucket_2.groupby('unique_id').apply(replace_with_nan).reset_index(drop=True)
            
            crossvalidation_df = pd.concat([crossvalidation_df,crossvalidation_df_bucket_2])
            
        if(len(self.ids_bucket_3)>0):

            if((self.freq=='M') | (self.freq=='MS')): 
                h = 12
            if(self.freq=='W'): 
                h = 4

            sf_naive_ = StatsForecast( 
                models=[Naive()],
                freq=self.freq,
                verbose=True)

            df_cross_naive = sf_naive_.cross_validation(df=self.train_df[self.train_df['unique_id'].isin(self.ids_bucket_3)].reset_index(drop=True) ,h=h).reset_index()

            crossvalidation_df_bucket_3 = self.sf_bucket_3.cross_validation(h=h)

            crossvalidation_df_bucket_3 = crossvalidation_df_bucket_3.reset_index().rename(
                columns=MDL_RENAME_MAPPING
            )

            crossvalidation_df_bucket_3 = crossvalidation_df_bucket_3.merge(df_cross_naive[['unique_id','ds','Naive']],on=['unique_id','ds'],how='left')

            crossvalidation_df_bucket_3 = crossvalidation_df_bucket_3.groupby('unique_id').apply(replace_with_nan).reset_index(drop=True)
            
            crossvalidation_df = pd.concat([crossvalidation_df,crossvalidation_df_bucket_3])

        if( (len(self.ids_bucket_3)>0) | (len(self.ids_bucket_2)>0) ):
            
            crossvalidation_df['stats_ensemble'] = crossvalidation_df[self.models].mean(axis=1)

            if(self.ignore_neg_fcsts):
                crossvalidation_df = crossvalidation_df[['unique_id','ds','cutoff','y']+self.model_pos]

        return crossvalidation_df
 

def build_forecast_model(df, model_type, date_format, data_frequency, seasonal_length, forecast_horizon, ignore_neg_fcsts, error_metric, eval=True):
    
    if(model_type=='statistical'):
        print('getting resulst for statsitcal model')
        print(df.columns)
        df_pred, df_performance, df_crossval, df_rank = train_statistical_model(df, seasonal_length, date_format, data_frequency, forecast_horizon, ignore_neg_fcsts, error_metric, eval)
    return df_pred, df_performance, df_crossval, df_rank


def train_statistical_model(df, seasonal_length, date_format, data_frequency, forecast_horizon, 
                             ignore_neg_fcsts, error_metric, eval):
    df = df.copy()
    print('under stats model', df.columns)
    stats = StatsForecastModels(
        seasonal_length=seasonal_length,
        freq=data_frequency,
        date_format=date_format,
        )
    
    stats.fit(df)
    df_pred = stats.predict(horizon=forecast_horizon, ignore_neg_fcsts=ignore_neg_fcsts)

    if ( eval ):
        df_crossval = stats.crossvalidation(error_metric=error_metric)
        df_performance = evaluate_cross_validation(df_crossval, error_metric)
        df_rank = rank_forecast(df_pred[df_pred['unique_id'].isin(df_performance.index)].reset_index(drop=True), df_performance)
        
        # ids_bucket_1 = list(( df.groupby('unique_id')['ds'].count()[ (df.groupby('unique_id')['ds'].count()<=27) & (df.groupby('unique_id')['ds'].count()>=12) ] ).index)

        # if(len(ids_bucket_1)>0):
        #     df_rank_bucket_1 = df_pred[df_pred['unique_id'].isin(ids_bucket_1)][['unique_id','ds','AutoTheta']].reset_index(drop=True)
        #     df_rank_bucket_1.rename(columns={'AutoTheta':'forecast'},inplace=True)
        #     df_rank_bucket_1['model'] = 'AutoTheta'
        #     df_rank_bucket_1['rank'] = 1
            
        #     df_rank = pd.concat([df_rank_bucket_1,df_rank])
    
    else:
        df_performance,df_rank,df_crossval = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    return df_pred, df_performance, df_crossval, df_rank


def evaluate_cross_validation(df, metric):
    error_metric = globals()[metric]
    models = df.drop(columns=["unique_id", "ds", "cutoff", "y"]).columns.tolist()
    evals = []
    for cutoff in df["cutoff"].unique():
        eval_ = evaluate(
            df[df["cutoff"] == cutoff], metrics=[error_metric], models=models
        )
        evals.append(eval_)
    evaluated = pd.concat(evals)
    evaluated = evaluated.groupby("unique_id").mean(numeric_only=True)

    evaluated.insert(0, "best_model", evaluated.idxmin(axis=1))
    return evaluated


def rank_forecast(pred__, evaluate__):

    df_tmp_transform_ = evaluate__.reset_index().T.reset_index()
    headers = df_tmp_transform_.iloc[0]
    df_tmp_rank_ = pd.DataFrame(df_tmp_transform_.values[1:], columns=headers)
    df_tmp_rank_.rename(columns={"unique_id": "models"}, inplace=True)
    df_tmp_rank_ = df_tmp_rank_[df_tmp_rank_["models"] != "best_model"].reset_index(
        drop=True
    )
    df_int_ = pd.DataFrame()
    for id_ in evaluate__.index:
        df_tmp_rank_[id_] = df_tmp_rank_[id_].astype(float)
        df_tmp_ = df_tmp_rank_[["models", id_]].sort_values(by=id_)
        df_tmp_pred = pd.DataFrame()
        rank = 1
        for mdl_ in df_tmp_["models"]:
            df_tmp_pred = pred__[pred__["unique_id"] == id_].reset_index(drop=True)[
                ["unique_id", "ds"]
            ]
            df_tmp_pred["model"] = mdl_
            df_tmp_pred["forecast"] = pred__[mdl_]
            df_tmp_pred["rank"] = rank
            df_int_ = pd.concat([df_int_, df_tmp_pred], ignore_index=True)
            rank += 1
    
    df_int_ = df_int_[~df_int_['forecast'].isna()].reset_index(drop=True)

    return df_int_


def run_forecast(
    seasonal_length,
    data_frequency,
    error_metric,
    ignore_neg_fcsts,
    date_format,
    forecast_horizon,
    df,
    model_type,
):
    
    return train_statistical_model(df, model_type, seasonal_length, date_format, data_frequency, forecast_horizon, 
                                           ignore_neg_fcsts, error_metric, eval=eval)
