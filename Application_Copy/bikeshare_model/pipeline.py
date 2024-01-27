import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer
from bikeshare_model.processing.features import WeathersitImputer
from bikeshare_model.processing.features import OutlierHandler
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import WeekdayOneHotEncoder
from bikeshare_model.processing.features import ColumnDropper

columns_to_drop = ['dteday', 'weekday']
bikeshare_pipe = Pipeline([
    # Data Imputation
    ('weekday_imputer', WeekdayImputer(day_column=config.model_config.day_column, weekday_column=config.model_config.weekday_column)),
    ('weather_imputer', WeathersitImputer(weathersit_column=config.model_config.weathersit_column)),
    
    # Outlier Handling
    ('outlier_handler', OutlierHandler(iqr_multiplier=config.model_config.iqr_multiplier)),
    
    # Categorical Mapping
    ('map_yr', Mapper(config.model_config.year_column, config.model_config.yr_mappings)),
    ('map_mnth', Mapper(config.model_config.month_column, config.model_config.mnth_mappings)),
    ('map_season', Mapper(config.model_config.season_column, config.model_config.season_mappings)),
    ('map_weathersit', Mapper(config.model_config.weathersit_column, config.model_config.weathersit_mappings)),
    ('map_holiday', Mapper(config.model_config.holiday_column, config.model_config.holiday_mappings)),
    ('map_workingday', Mapper(config.model_config.workingday_column, config.model_config.workingday_mappings)),
    ('map_hr', Mapper(config.model_config.hour_column, config.model_config.hr_mappings)),
    
    # One-Hot Encoding
    ('weekday_encoder', WeekdayOneHotEncoder(weekday_column=config.model_config.weekday_column)),

    # Drop Unused Columns
    ('column_dropper', ColumnDropper(columns_to_drop=columns_to_drop)),
    
    # Feature Scaling (optional)
    ('scaler', StandardScaler()),

    # Regression Model
    ('regressor', linear_model.LinearRegression())
])