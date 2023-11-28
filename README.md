# Forecasting Wildfires in California

## Introduction

With trends set in motion by Climate Change, wildfires are becoming an increasingly devastating force for populations around the world. In some regions, such as California in the United States, they are a leading cause of property loss, habitat destruction, and GHG emissions. In the 2020 fire season, wildfires in California caused over 12 Billion USD in damages. There is a high demand for prediction capabilities in regions such as this. Even small predictive ability on the order of minutes or hours could avert billions in damages by allowing first responders the ability to mitigate blazes before they spread, erect preventative measures, mobilize resources proactively to blaze sites, and begin evacuation efforts before residents are in peril. This project attempts to predict and forecast wildfires, using machine learning classifier algorithms, in California.

## Data

This analysis makes use of several datasets in order to extract disparate features that are important for determining if an area is at risk of wildfire. These include, but are not limited to, elevation, previous fires, surface reflectance and emissivity, land surface temperature, soil temperature and water content, and wind. The datasets used in this analysis are as follows:

* [USGS_SRTMGL1_003](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003)
* [MODIS_061_MOD14A1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD14A1)
* [MODIS_061_MOD11A1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A1)
* [MODIS_061_MOD09GA](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09GA)
* [ECMWF_ERA5_LAND_DAILY_AGGR](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR)

## Libraries

`import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm.sklearn import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.inspection import permutation_importance
import eli5
import shap
from sklearn.metrics import (
    average_precision_score, 
    precision_recall_curve, 
    roc_auc_score, 
    roc_curve, 
    ConfusionMatrixDisplay
)`