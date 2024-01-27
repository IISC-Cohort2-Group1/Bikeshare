
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
import math
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer, WeathersitImputer, WeekdayOneHotEncoder, OutlierHandler, Mapper


def test_weekday_imputer(sample_input_data):
    # Given
    transformer = WeekdayImputer(day_column=config.model_config.day_column, weekday_column=config.model_config.weekday_column)
    assert math.isnan(sample_input_data.loc[7046,config.model_config.weekday_column])

    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    # Then
    assert sample_input_data.loc[7046,config.model_config.weekday_column] == 'Wed'


def test_weathersit_imputer(sample_input_data):
    # Given
    transformer = WeathersitImputer(
        weathersit_column=config.model_config.weathersit_column,  # cabin
    )
    assert np.isnan(sample_input_data.loc[7046,config.model_config.weathersit_column])

    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    # Then
    assert subject.loc[7046,config.model_config.weathersit_column] == 'Clear'

def test_mapper(sample_input_data):
    # Given
    transformer = Mapper(
        config.model_config.season_column, config.model_config.season_mappings
    )
    assert sample_input_data[config.model_config.season_column].dtype == 'O'

    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    # Then
    assert subject[config.model_config.season_column].dtype == 'int32'

def test_outlier_handler(sample_input_data):
    # Given
    transformer = OutlierHandler(
        iqr_multiplier = config.model_config.iqr_multiplier,  # cabin
    )
    assert sample_input_data['windspeed'].max() > 31.992

    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    # Then
    assert subject['windspeed'].max().round(3) == 31.992

def test_weekday_onehot_encoder(sample_input_data):
    # Given
    transformer = WeekdayOneHotEncoder(
        weekday_column=config.model_config.weekday_column,  # cabin
    )
    assert len(sample_input_data.columns) == 14

    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    # Then
    assert len(subject.columns) == 22