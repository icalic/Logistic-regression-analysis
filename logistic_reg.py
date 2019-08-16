#!usr/bin/python
#Linear regression model
#Code source: Irina Calic

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.formula.api as sm
import seaborn as sns
import patsy

from pandas import DataFrame, Series
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.graphics.regressionplots import *
from statsmodels.iolib.summary2 import summary_col
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

sns.set(color_codes=True)

df1 = pd.read_csv('/path/path/filename.csv')

df1.dropna()
y=len(df1.columns)


for x in range(0, y-1):
	a=df1.columns[x]
	logit = smf.logit(formula="fitness ~ "+a, data=df1).fit()
	print(logit.summary()) 
	logitreg = smf.logit(formula="fitness ~ "+a+" + np.power("+a+", 2)", data=df1).fit()
	print(logitreg.summary())
	