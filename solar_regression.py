#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline


# We will now retrieve and process the CSV file containing actual weather data for the city of Birmingham, Alabama. In order to later aggregate the electricity generated daily, we divide the Date and Time data into additional columns.

df = pd.read_csv('Actual_33.55_-86.85_2006_DPV_39MW_5_Min.csv')
df.head()
df['LocalTime'] = pd.to_datetime(df['LocalTime'], format="%m/%d/%y %H:%M")
df['Date'] = df['LocalTime'].dt.date
df['Time'] = df['LocalTime'].dt.time
print(df.head())


# To designate that total value as the dependent variable later on, we will want to take the total MW generated across all of the entire days in the year. After that, we'll correlate it with a number of independent variables, including local meteorological characteristics from another dataset.

days_df = df.groupby('Date').sum('Power(MW)').reset_index()
days_df.plot(kind='bar',x='Date',y='Power(MW)',xlabel='Days of the Year 2006',ylabel='Total Power Generated (MW)')


# In the data above, there appears to be a bell-shaped distribution among the higher values, reaching its peak during the summer months, with numerous lower values scattered throughout. To develop a predictive model for the total daily power generation, we'll import a distinct CSV file containing essential weather variables for the specified time and location.

features = pd.read_csv('Birmingham_Weather_2006.csv')
new_df = days_df.merge(features,on='Date',how='right')
new_df['Power(MW)'] = days_df['Power(MW)']
new_df = new_df.drop('Unnamed: 0',inplace=False,axis=1)
print(new_df.head())


# Create feature variables for regression

X = new_df.drop('Power(MW)',axis=1)
X.drop('Date',inplace=True,axis=1)
y = new_df['Power(MW)'].to_numpy().reshape(-1,1)



# To understand model performance, dividing the dataset into training data and testing data is a brilliant strategy. Split the data into 90% training data and 10% testing.
# 
# Let's split the dataset by using the function train_test_split(). You need to pass 3 parameters: features, target, and test_set size. Additionally, you can use random_state to select records randomly.
# 
# Then, fit the linear regression model on the train set using fit() and perform prediction on the test set using predict(). 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=40)

model = LinearRegression()
model.fit(X_train,y_train)
print(model.coef_)
coefs = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(model.coef_))],axis=1, ignore_index=True)
coefs.rename(columns={0:'Features',1:'Coefficients'},inplace=True)
print(coefs)


# Unexpectedly, Temperature seems to be have a negative correlation with the amount solar power generated, while Hours of Daylight is positively. The negative correlation could be explained by the fact that a certain point or temperature, the efficiency of solar panels starts decreasing as the temperature increases . 
# Let's make predictions about the solar power generated based on our weather data

y_pred = model.predict(X_test)

predict_df = pd.DataFrame(y_test)
predict_df.columns = ['Actual Value (MW)']

predict_df['Predicted Value (MW)'] = pd.DataFrame(y_pred).apply(lambda value:round(value,1))

predict_df['Difference'] = y_pred-y_test
predict_df['Difference'] = predict_df['Difference'].apply(lambda value:round(value,1))

predict_df['% Error'] = round(100*(predict_df['Difference']/predict_df['Actual Value (MW)']), 2)
print(predict_df)
print('\nAverage Difference')
print(round(predict_df['Difference'].sum()/37,2))






