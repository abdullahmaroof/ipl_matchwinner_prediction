import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

ipldata = pd.read_csv('IPL_Matches_2008_2021.csv')

print(ipldata)

print(ipldata.info())

iplnewdata=ipldata.drop(columns=['ID','MatchNumber','SuperOver','City','Date','Season','Player_of_Match','Team1Players','Team2Players',
                                 'Umpire1','Umpire2','method','Margin','WonBy'])
print(iplnewdata.info())

iplnewdata=iplnewdata.dropna()

print("Counting Team1 values in dataset")
print(iplnewdata.Team1.value_counts())

print("Counting Team2 values in dataset")
print(iplnewdata.Team2.value_counts())

print("Counting Venue values in dataset")
print(iplnewdata.Venue.value_counts())

print("Counting TossWinner values in dataset")
print(iplnewdata.TossWinner.value_counts())

print("Counting TossDecision values in dataset")
print(iplnewdata.TossDecision.value_counts())

print("Counting WinningTeam values in dataset")
print(iplnewdata.WinningTeam.value_counts())

iplnewdata=iplnewdata.replace({'WinningTeam':{'Mumbai Indians':0,'Chennai Super Kings':1,'Kolkata Knight Riders':2,
                                   'Royal Challengers Bangalore':3,'Kings XI Punjab':4,'Rajasthan Royals':5,
                                   'Sunrisers Hyderabad':6,'Delhi Daredevils':7,'Delhi Capitals':8,'Deccan Chargers':9,
                                   'Gujarat Lions':10,'Rising Pune Supergiant':11,'Pune Warriors':12,'Punjab Kings':13,
                                   'Kochi Tuskers Kerala':14,'Rising Pune Supergiants':15}})

iplnewdata=iplnewdata.replace({'TossDecision':{'field':0,'bat':1}})

iplnewdata=iplnewdata.replace({'TossWinner':{'Mumbai Indians':0,'Chennai Super Kings':1,'Kolkata Knight Riders':2,
                                   'Royal Challengers Bangalore':3,'Kings XI Punjab':4,'Rajasthan Royals':5,
                                   'Sunrisers Hyderabad':6,'Delhi Daredevils':7,'Delhi Capitals':8,'Deccan Chargers':9,
                                   'Gujarat Lions':10,'Rising Pune Supergiant':11,'Pune Warriors':12,'Punjab Kings':13,
                                   'Kochi Tuskers Kerala':14,'Rising Pune Supergiants':15}})

iplnewdata=iplnewdata.replace({'Team1':{'Mumbai Indians':0,'Chennai Super Kings':1,'Kolkata Knight Riders':2,
                                   'Royal Challengers Bangalore':3,'Kings XI Punjab':4,'Rajasthan Royals':5,
                                   'Sunrisers Hyderabad':6,'Delhi Daredevils':7,'Delhi Capitals':8,'Deccan Chargers':9,
                                   'Gujarat Lions':10,'Rising Pune Supergiant':11,'Pune Warriors':12,'Punjab Kings':13,
                                   'Kochi Tuskers Kerala':14,'Rising Pune Supergiants':15}})

iplnewdata=iplnewdata.replace({'Team2':{'Mumbai Indians':0,'Chennai Super Kings':1,'Kolkata Knight Riders':2,
                                   'Royal Challengers Bangalore':3,'Kings XI Punjab':4,'Rajasthan Royals':5,
                                   'Sunrisers Hyderabad':6,'Delhi Daredevils':7,'Delhi Capitals':8,'Deccan Chargers':9,
                                   'Gujarat Lions':10,'Rising Pune Supergiant':11,'Pune Warriors':12,'Punjab Kings':13,
                                   'Kochi Tuskers Kerala':14,'Rising Pune Supergiants':15}})

iplnewdata=iplnewdata.replace({'Venue':{'Eden Gardens':0,'Wankhede Stadium':1,'M Chinnaswamy Stadium':2, 'Feroz Shah Kotla':3,
                                        'Rajiv Gandhi International Stadium, Uppal':4,'MA Chidambaram Stadium, Chepauk':5,
                                        'Sawai Mansingh Stadium':6,'Dubai International Cricket Stadium':7,
                                        'Punjab Cricket Association Stadium, Mohali':8,'Sheikh Zayed Stadium':9,
                                        'Sharjah Cricket Stadium':10,'Maharashtra Cricket Association Stadium':11,
                                        'Dr DY Patil Sports Academy':12,'Subrata Roy Sahara Stadium':13,
                                        'Rajiv Gandhi International Stadium':14,'M.Chinnaswamy Stadium':15,
                                        'Punjab Cricket Association IS Bindra Stadium, Mohali':16, 'Kingsmead':17,
                                        'OUTsurance Oval':18,'Buffalo Park':19, 'De Beers Diamond Oval':20,
                                        'Brabourne Stadium, Mumbai':21,'Arun Jaitley Stadium, Delhi':22, 'Newlands':23,
                                        'Vidarbha Cricket Association Stadium, Jamtha':24,'Green Park':25, 'Nehru Stadium':26,
                                        'St George\'s Park':27,'Barabati Stadium':28,'JSCA International Stadium Complex':29,
                                        'Narendra Modi Stadium, Ahmedabad':30,'Zayed Cricket Stadium, Abu Dhabi':31,
                                        'Shaheed Veer Narayan Singh International Stadium':32,'New Wanderers Stadium':33,
                                        'MA Chidambaram Stadium':34, 'Himachal Pradesh Cricket Association Stadium':35,
                                        'Holkar Cricket Stadium':36, 'Arun Jaitley Stadium':37, 'SuperSport Park':38,
                                        'Punjab Cricket Association IS Bindra Stadium':39, 'Wankhede Stadium, Mumbai':40,
                                        'Saurashtra Cricket Association Stadium':41, 'Brabourne Stadium':42, 
                                        'Sardar Patel Stadium, Motera':43, 'MA Chidambaram Stadium, Chepauk, Chennai':44,
                                        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium':45}})

print(iplnewdata)

iplnewdata = iplnewdata.sort_values('TossWinner',ascending=False)

print(iplnewdata)

X = iplnewdata.drop(columns=['WinningTeam'])
Y = iplnewdata['WinningTeam']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

lin_reg_model = DecisionTreeRegressor()

lin_reg_model.fit(X_train, Y_train)

X_train_prediction = lin_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train, X_train_prediction)
print("R squared Error : ", error_score)



X_test_prediction = lin_reg_model.predict(X_test)
test_data_accuracy = metrics.r2_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)


input_data = (3,6,42,6,0)

input_data_as_numpy_array = np.asarray(input_data)

std_data = input_data_as_numpy_array.reshape(1,-1)

print(std_data)

prediction = lin_reg_model.predict(std_data)
print(prediction)

if (prediction[0]==0):
    print('Team 1 Winner')
    print(input_data[0])
else:
    print('Team 2 Winner')
    print(input_data[1])
