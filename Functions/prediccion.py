# se creara  un modelos de  prediccion de  ventas de acuerdo a diferentes variables 

# importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# importar dataset
df = pd.read_csv('data/Advertising.csv')
df.head()

# explorar dataset
