#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 11:54:59 2019

@author: xiaolu
"""
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


### load boston data 
boston = load_boston()

### assign dependent variable to y
y = boston.target

### assign independent variables to x 
x = boston.data

### assign feature names of boston dataset to name 
names = boston.feature_names

### fit linear model between x & y 
regression = LinearRegression().fit(x, y)

### print linear model coefficients and intercept
regression.coef_

regression.intercept_


### combine variable names with fitted regression coefficients into dictionary
pair = dict(zip(names, regression.coef_))

### convert all regression coefficients to their absolute value 
new_pair = {key: abs(value) for key, value in pair.items()}

new_pair

### as we can see the variable with highest absolute value coefficients is NOX
### - NOX      nitric oxides concentration (parts per 10 million)
