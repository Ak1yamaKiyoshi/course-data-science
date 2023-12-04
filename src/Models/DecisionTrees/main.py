import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

def load_data():
    df = pd.read_csv('./Employee Attrition.csv')
    return df

df = load_data()
"""
1) View the data and obtain statistical summaries. Ensure data types are appropriate and
there is no missing data. Determine the outcome and input variables.
"""
#print(df.info())
#print(df.isnull().sum())


sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))

work_accident_counts = df['Work_accident'].value_counts().reset_index()
work_accident_counts.columns = ['Work Accident', 'Count']

# Create the bar plot
barplot = sns.barplot(x='Work Accident', y='Count', data=work_accident_counts, palette='pastel')

# Customize the plot
plt.xlabel('Work Accident')
plt.ylabel('Number of Employees')
plt.title('Work Accidents')
barplot.set_xticklabels(['No', 'Yes'])
plt.show()