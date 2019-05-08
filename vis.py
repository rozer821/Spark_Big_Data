from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('results-20190505-160439.csv')
df['rain'].replace(df['rain'].unique()[0],0,inplace=True)
df['rain'].replace(df['rain'].unique()[1],1,inplace=True)
print(df.head(10))