import quandl
import pandas as pd
import numpy as np
import datetime

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

df = quandl.get("WIKI/AMZN")

#print(df.tail())

df = df[['Adj. Close']]

forecast_out = int(30) # predicting 30 days into future
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out) #  label column with data shifted 30 units up
print(df.tail())

X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:] # set X_forecast equal to last 30
X = X[:-forecast_out] # remove last 30 from X

y = np.array(df['Prediction'])
y = y[:-forecast_out]

# Linear Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Training
clf = LinearRegression()
clf.fit(X_train,y_train)
# Testing
confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)

forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)

# not sure how to plot
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color="#467ddd", label="trained data")
plt.scatter(X_test, y_test, color="#dd4646", label="test data")
plt.plot(X_train, clf, color="000000", label="regression line")
plt.title("Prediction of Amazon Stock Price in Next 30 Days")
plt.legend(loc=4)


