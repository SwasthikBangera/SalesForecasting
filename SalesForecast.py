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
outlet_size_mode = sales_dataset.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
print(outlet_size_mode)

# Replacing categorical missing values with mode
missing_values = sales_dataset['Outlet_Size'].isnull()
print(missing_values)

sales_dataset.loc[missing_values,'Outlet_Size'] = sales_dataset.loc[missing_values, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
print(sales_dataset.isnull().sum())


# Statistical data of dataset
print(sales_dataset.describe())

''' Numerical Features '''

# Set aspects of seaborn and matplotlib
sns.set()

# Item weight distribution
plt.figure(figsize=(6,6))
sns.displot(sales_dataset['Item_Weight'])
plt.show()

# Item Visibility distribution
plt.figure(figsize=(6,6))
sns.displot(sales_dataset['Item_Visibility'])
plt.show()

# Item MRP distribution
plt.figure(figsize=(6,6))
sns.displot(sales_dataset['Item_MRP'])
plt.show()

# Outlet Establishment Year distribution
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=sales_dataset)
plt.show()

# Item Outlet Sales distribution
plt.figure(figsize=(6,6))
sns.displot(sales_dataset['Item_Outlet_Sales'])
plt.show()

'''  Categorical Features '''

print(list(sales_dataset.select_dtypes(['object']).columns))

# Item_Fat_Content distribution
plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=sales_dataset)
plt.show()

# Item_Type distribution
plt.figure(figsize=(35,6))
sns.countplot(x='Item_Type', data=sales_dataset)
plt.show()

# Outlet_Size distribution - displot
plt.figure(figsize=(6,6))
sns.displot(sales_dataset['Outlet_Size'])
plt.show()

# Outlet_Size distribution - countplot
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Size', data=sales_dataset)
plt.show

''' Data pre-processing using Label encoding '''

#Calling the encoder
encoder = LabelEncoder()

# Convert Categorical data to Numerical data using Label Encoding
sales_dataset['Item_Identifier'] = encoder.fit_transform(sales_dataset['Item_Identifier'])
print(sales_dataset['Item_Identifier'])

sales_dataset['Item_Fat_Content'] = encoder.fit_transform(sales_dataset['Item_Fat_Content'])
print(sales_dataset['Item_Fat_Content'])

sales_dataset['Item_Type'] = encoder.fit_transform(sales_dataset['Item_Type'])
print(sales_dataset['Item_Type'])

sales_dataset['Outlet_Identifier'] = encoder.fit_transform(sales_dataset['Outlet_Identifier'])
print(sales_dataset['Outlet_Identifier'])

sales_dataset['Outlet_Size'] = encoder.fit_transform(sales_dataset['Outlet_Size'])
print(sales_dataset['Outlet_Size'])

sales_dataset['Outlet_Location_Type'] = encoder.fit_transform(sales_dataset['Outlet_Location_Type'])
print(sales_dataset['Outlet_Location_Type'])

sales_dataset['Outlet_Type'] = encoder.fit_transform(sales_dataset['Outlet_Type'])
print(sales_dataset['Outlet_Type'])

print(sales_dataset.head())

''' Split the datta set to Features and Target '''

X = sales_dataset.drop(columns='Item_Outlet_Sales',axis=1)
y = sales_dataset['Item_Outlet_Sales']

print(X.head(), y.head())
print(X.shape, y.shape)

''' Split data into Training and Test data '''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

