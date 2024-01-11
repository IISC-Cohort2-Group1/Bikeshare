from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from bikeshare_model.config.core import config

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, day_column='dteday', weekday_column='weekday'):
        # YOUR CODE HERE
        self.day_column = day_column
        self.weekday_column = weekday_column
        self.imputer = SimpleImputer(strategy='constant', fill_value='Unknown')

    def fit(self, X:pd.DataFrame, y:pd.Series = None):
        # YOUR CODE HERE
        # Extracting day names from the 'dteday' column
        day_names = pd.to_datetime(X[self.day_column]).dt.day_name()

        # Mapping full day names to first three letters
        day_names_mapped = day_names.str[:3]

        # Fitting the SimpleImputer with the mode of 'weekday' column
        self.imputer.fit(X[[self.weekday_column]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Finding NaN entries and their indices in the 'weekday' column
        nan_indices = X[X[self.weekday_column].isna()].index

        # Extracting day names from the 'dteday' column
        day_names = pd.to_datetime(X[self.day_column]).dt.day_name()

        # Mapping full day names to first three letters
        day_names_mapped = day_names.str[:3]

        # Imputing values for the missing row indices in 'weekday' column
        X.loc[nan_indices, self.weekday_column] = day_names_mapped.loc[nan_indices]

        # Applying the SimpleImputer to handle any remaining missing values
        X[self.weekday_column] = self.imputer.transform(X[[self.weekday_column]])
        return X
    
class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, weathersit_column='weathersit'):
        # YOUR CODE HERE
        self.weathersit_column = weathersit_column
        self.imputer = SimpleImputer(strategy='most_frequent')

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        # YOUR CODE HERE
        # Fitting the SimpleImputer with the most frequent category of 'weathersit' column
        self.imputer.fit(X[[self.weathersit_column]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        # Applying the SimpleImputer to fill missing values in 'weathersit' column
        X[self.weathersit_column] = self.imputer.transform(X[[self.weathersit_column]])
        return X
    
class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variables: str, mappings: dict):
        # YOUR CODE HERE
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        # YOUR CODE HERE
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X_copy = X.copy()
        X_copy[self.variables] = X_copy[self.variables].map(self.mappings).astype(int)
        return X_copy
    
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """
    def __init__(self, numerical_columns=None, iqr_multiplier=1.5):
        # YOUR CODE HERE
        self.numerical_columns = numerical_columns if numerical_columns else config.model_config.numerical_columns
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        # YOUR CODE HERE
        self.iqr = X[self.numerical_columns].quantile(0.75) - X[self.numerical_columns].quantile(0.25)
        return self

    def transform(self, X):
        # YOUR CODE HERE
        X_copy = X.copy()

        for column in self.numerical_columns:
           # Define upper and lower bounds based on IQR
            upper_bound = X_copy[column].quantile(0.75) + self.iqr_multiplier * self.iqr[column]
            lower_bound = X_copy[column].quantile(0.25) - self.iqr_multiplier * self.iqr[column]

            # Cap values to upper and lower bounds
            X_copy[column] = X_copy[column].clip(lower=lower_bound, upper=upper_bound)
        return X_copy
    
class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, weekday_column='weekday'):
        # YOUR CODE HERE
        self.weekday_column = weekday_column
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X, y=None):
        # YOUR CODE HERE
        # Extract the 'weekday' column for one-hot encoding
        weekdays = X[[self.weekday_column]]

        # Fit the OneHotEncoder
        self.one_hot_encoder.fit(weekdays)
        return self

    def transform(self, X):
        # YOUR CODE HERE
        X_copy = X.copy()

        # Extract the 'weekday' column for one-hot encoding
        weekdays = X_copy[[self.weekday_column]]

        # Transform and append the one-hot encoded features
        encoded_weekday = self.one_hot_encoder.transform(weekdays)
        enc_wkday_features = self.one_hot_encoder.get_feature_names_out([self.weekday_column])
        X_copy[enc_wkday_features] = encoded_weekday
        return X_copy
    
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop if columns_to_drop else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.drop(columns=self.columns_to_drop, inplace=True)
        return X_copy