"""
AUTHOR: HARIKRISHNA MUDIDUDDI
PLACE: HYDERABAD
DESIGNATION: Jr DATA SCIENTIST
PROJECT: CANDIDATE SELECTION
"""

import pandas as pd
import numpy as np
import pickle

data = pd.read_csv('hiring.csv')
data.head()

data['experience'].fillna(0, inplace=True)

data.head()

data['test_score'].fillna(data['test_score'].mean(), inplace=True)

X = data.iloc[:, :3]

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0:0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

Y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear = linear.fit(X_train, Y_train)

"""from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))"""

from sklearn.preprocessing import PolynomialFeatures

polyR = PolynomialFeatures(degree = 4)
x_poly = polyR.fit_transform(X)

polyR.fit(x_poly, Y)
polymodel = linear.predict(X_test)

pickle.dump(linear, open('model.pk1', 'wb'))

model = pickle.load(open('model.pk1','rb'))



