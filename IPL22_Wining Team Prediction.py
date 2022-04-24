import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

ipldata = pd.read_csv('IPL_Matches_2008_2021.csv')

print(ipldata)

print(ipldata.info())

iplnewdata=ipldata.drop(columns=['ID','MatchNumber','SuperOver','City','Date','Season','Player_of_Match','Team1Players','Team2Players',
                                 'Umpire1','Umpire2','method','Margin','WonBy'])
print(iplnewdata.info())
