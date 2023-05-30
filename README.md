# Ex-06-Feature-Transformation
Ex-06-Feature-Transformation
AIM
To read the given data and perform Feature Transformation process and save the data to a file.

ALGORITHM
STEP 1
Read the given Data

STEP 2
Clean the Data Set using Data Cleaning Process

STEP 3
Apply Feature Transformation techniques to all the feature of the data set

STEP 4
Save the data to the file

CODE
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

OUTPUT

![image](https://github.com/vidhyasrikachapalayam/Ex-06-Feature-Transformation/assets/119477817/cd22a663-5778-47fe-b016-38170df1b236)

![image](https://github.com/vidhyasrikachapalayam/Ex-06-Feature-Transformation/assets/119477817/582e334d-2c7e-497f-85ed-d217a7e031b2)

![image](https://github.com/vidhyasrikachapalayam/Ex-06-Feature-Transformation/assets/119477817/4d9627b2-ded8-4847-9ee0-bd9e8547e265)

