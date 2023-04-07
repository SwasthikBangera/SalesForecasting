''' Create a Forecasting model for sales '''

# Importing Dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

''' Exploratory Data analysis '''

# Impport the dataset into pandas  dataframe
sales_dataset = pd.read_csv('/Users/yennamac/Downloads/SalesForecasting/Train.csv')

# Print the first five rows
print(sales_dataset.head())

# Names of all features
print(sales_dataset.head(0))

# Alternatively get the names of all features
print(sales_dataset.columns.tolist())

# No. of datapoints in dataset
print(sales_dataset.shape)

# Information about the dataset
print(sales_dataset.info())

# Statistical data of dataset
print(sales_dataset.describe())

# Type of values in 'object' column - Item_Identifier
print(sales_dataset['Item_Identifier'].unique())
print(sales_dataset['Item_Identifier'].value_counts())

# Type of values in 'object' column - Item_Fat_Content
print(sales_dataset['Item_Fat_Content'].unique())
print(sales_dataset['Item_Fat_Content'].value_counts())

# Type of values in 'object' column - Item_Type
print(sales_dataset['Item_Type'].unique())
print(sales_dataset['Item_Type'].value_counts())

# Type of values in 'object' column - Outlet_Identifier
print(sales_dataset['Outlet_Identifier'].unique())
print(sales_dataset['Outlet_Identifier'].value_counts())

# Type of values in 'object' column - Outlet_Size
print(sales_dataset['Outlet_Size'].unique())
print(sales_dataset['Outlet_Size'].value_counts())

# Type of values in 'object' column - Outlet_Location_Type
print(sales_dataset['Outlet_Location_Type'].unique())
print(sales_dataset['Outlet_Location_Type'].value_counts())

# Type of values in 'object' column - Outlet_Type
print(sales_dataset['Outlet_Type'].unique())
print(sales_dataset['Outlet_Type'].value_counts())

# Maintain consistent values in Item_Fat_Content 
sales_dataset = sales_dataset.replace({'Item_Fat_Content':{'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'}})
print(sales_dataset['Item_Fat_Content'].unique())
print(sales_dataset['Item_Fat_Content'].value_counts())

''' Replace missing values via imputation '''

# No. of missing values
print(sales_dataset.isnull().sum())

# Mean value of Item_Weight
print(sales_dataset['Item_Weight'].mean())

# Replacing numerical missing values with mean 
sales_dataset['Item_Weight'].fillna(sales_dataset['Item_Weight'].mean(), inplace=True)
print(sales_dataset.isnull().sum())

# Mode value of Outlet_Size


# Replacing categorical missing values with mode

 



