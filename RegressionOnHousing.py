import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from UsefulFunctions import lin_regplot

# Loading the data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# • CRIM: This is the per capita crime rate by town
# • ZN: This is the proportion of residential land zoned for lots larger than
# 25,000 sq.ft.
# • INDUS: This is the proportion of non-retail business acres per town
# • CHAS: This is the Charles River dummy variable (this is equal to 1 if tract
# bounds river; 0 otherwise)
# • NOX: This is the nitric oxides concentration (parts per 10 million)
# • RM: This is the average number of rooms per dwelling
# • AGE: This is the proportion of owner-occupied units built prior to 1940
# • DIS: This is the weighted distances to five Boston employment centers
# • RAD: This is the index of accessibility to radial highways
# • TAX: This is the full-value property-tax rate per $10,000
# • PTRATIO: This is the pupil-teacher ratio by town
# • B: This is calculated as 1000(Bk - 0.63)^2, where Bk is the proportion of
# people of African American descent by town
# • LSTAT: This is the percentage lower status of the population
# • MEDV: This is the median value of owner-occupied homes in $1000s

print(df.head())

#Visualizing the important characteristics of the housing dataset, we will use the scatterplot mattrix in seaborn to visualize the pair-wise correlations between features

# sns.set(style='whitegrid', context='notebook')
# sns.pairplot(df[df.columns], size=1);
# plt.show()

# Relatively strong relationships (DIS, NOX) (DIS, INDUS) ( CRIM, MEDV) (DIS, CRIM)

#Let's try the Pearson product-moment correlation coefficients !

# cols = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# cm = np.corrcoef(df[cols].values.T)
# sns.set(font_scale=1.5)
# hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols, xticklabels=cols)
# plt.show()



#Highest correlations, MEDV as a target : (MEDV, RM )=0.70 , (MEDV, LSTAT) = -0.74

# X = df['RM'].values
# X = X.reshape(len(X),1)
# Y = df['MEDV'].values

#Let's use Scikit-learn' linear regression model.

# lr = LinearRegression()
# lr.fit(X,Y)
# print('Slope : %.3f' % lr.coef_[0])
# print('Intercept : %.3f' % lr.intercept_)
# lin_regplot(X, Y , lr)
# plt.xlabel('avg number of rooms std')
# plt.ylabel('Price in $1000 MEDV std')
# plt.show()



#Fit the regression model to a subset of inliers ( using RASNAC )

# ransac = RANSACRegressor(LinearRegression(),max_trials=100,min_samples=50,residual_metric=lambda x: np.sum(np.abs(x), axis=1),residual_threshold=5.0,random_state=0)
# ransac.fit(X, Y)
#
# print('Slope: %.3f' % ransac.estimator_.coef_[0])
# print('Intercept: %.3f' % ransac.estimator_.intercept_)
#
# inlier_mask = ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)
# line_X = np.arange(3, 10, 1)
# line_y_ransac = ransac.predict(line_X[:, np.newaxis])
# plt.scatter(X[inlier_mask], Y[inlier_mask],c='blue', marker='o', label='Inliers')
# plt.scatter(X[outlier_mask], Y[outlier_mask],c='lightgreen', marker='s', label='Outliers')
# plt.plot(line_X, line_y_ransac, color='red')
# plt.xlabel('Average number of rooms [RM]')
# plt.ylabel('Price in $1000\'s [MEDV]')
# plt.legend(loc='upper left')
# plt.show()


X= df.iloc[:, :-1].values
Y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

# Coef of determination : R^2 = 1 - SSE/SST   , SST(total sum of squares) ~ Variance , SSE ( sum of squared errors )
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

#Plotting the Residuals between predicted train and test sets :

# plt.scatter(y_train_pred, y_train_pred - y_train,c='blue', marker='o', label='Training data')
# plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
# plt.xlabel('Predicted values')
# plt.ylabel('Residuals')
# plt.legend(loc='upper left')
# plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
# plt.xlim([-10, 50])
# plt.show()