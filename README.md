# Ex-06-Feature-Transformation

# AIM

## STEP 1

Read the given Data

## STEP 2

Clean the Data Set using Data Cleaning Process

## STEP 3

Apply Feature Transformation techniques to all the feature of the data set

## STEP 4

Save the data to the file

# CODE

import pandas as pd

df=pd.read_csv('/content/Data_to_Transform.csv')

df.head()

df.isnull().sum()

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')

plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')

plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')

plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')

plt.show()

df['Highly Positive Skew']=np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')

plt.show()

df['Highly Positive Skew']=1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')

plt.show()

df['Highly Positive Skew']=np.sqrt(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')

plt.show()

from sklearn.preprocessing import PowerTransformer pt=PowerTransformer("yeo-johnson")

df['Moderate Negative Skew']=pd.DataFrame(pt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')

plt.show()

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df['Moderate Negative Skew']=pd.DataFrame(pt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')

plt.show()

# OUTPUT

![image](https://github.com/vidhyasrikachapalayam/Ex-06-Feature-Transformation/assets/119477817/cd22a663-5778-47fe-b016-38170df1b236)

![image](https://github.com/vidhyasrikachapalayam/Ex-06-Feature-Transformation/assets/119477817/582e334d-2c7e-497f-85ed-d217a7e031b2)

![image](https://github.com/vidhyasrikachapalayam/Ex-06-Feature-Transformation/assets/119477817/4d9627b2-ded8-4847-9ee0-bd9e8547e265)

![image](https://github.com/vidhyasrikachapalayam/Ex-06-Feature-Transformation/assets/119477817/35775249-c6c5-48a7-862d-78f547c14947)

![image](https://github.com/vidhyasrikachapalayam/Ex-06-Feature-Transformation/assets/119477817/4df79385-80a5-4cdc-ae70-9ac213b00c7b)

![image](https://github.com/vidhyasrikachapalayam/Ex-06-Feature-Transformation/assets/119477817/0e453b9b-5081-4a89-96f9-ae49d0ab01b9)

![image](https://github.com/vidhyasrikachapalayam/Ex-06-Feature-Transformation/assets/119477817/6355e80b-8c46-48bc-85f7-c3b494d256fb)

![image](https://github.com/vidhyasrikachapalayam/Ex-06-Feature-Transformation/assets/119477817/21d1e727-f86d-4d55-9741-5ce68585d21c)

![image](https://github.com/vidhyasrikachapalayam/Ex-06-Feature-Transformation/assets/119477817/078b61a2-8be7-4b39-a872-dbec2af768a9)

![image](https://github.com/vidhyasrikachapalayam/Ex-06-Feature-Transformation/assets/119477817/61621502-c51c-4de5-9025-9f41c3281785)

![image](https://github.com/vidhyasrikachapalayam/Ex-06-Feature-Transformation/assets/119477817/376b1e7c-976b-41b7-8cf4-66c85a4e201a)

![image](https://github.com/vidhyasrikachapalayam/Ex-06-Feature-Transformation/assets/119477817/2f2f0aa4-3c6a-4a1f-95a8-e359753e9ecf)

# RESULT

Thus feature transformation is done for the given dataset.

