#!/usr/bin/env python
# coding: utf-8

# ###### <h1 align='center'><u>IPL 2022 Match Winner Prediction System</u></h1>

# ![IPLLogo.png](attachment:IPLLogo.png)

# ## Coder Tech Team
# ##### Team Leader: Abdullah Maroof  - BAIM-F19-007
# ##### Team Member: Abdullah Mujtaba - BDSM-S20-003
# ##### Team Member:    Zubair Ali    - BAIM-S20-009

# ###### <br>
# ###### <hr>
# ###### Import Libraries
# ###### Numpy for matrices & Mathematical Functions
# ###### Pandas for importing Dataset & perform different functions on it
# ###### Sklearn for training & testing our model & predict diabetes

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# ###### <hr>
# ###### Importing Dataset

# In[2]:


ipldata = pd.read_csv('IPL_Matches_2008_2021.csv')


# In[3]:


ipldata


# ###### <hr>
# ###### Checking Null Values

# In[4]:


ipldata.isnull().sum()


# ###### <hr>
# ###### Checking Cities in IPL Data of Matches

# In[5]:


print('Matches played in which cities')
ipldata['City'].unique()


# ###### <hr>
# ###### Checking all time Teams of IPL

# In[6]:


print('Teams in IPL')
ipldata['Team1'].unique()


# ###### <hr>
# ###### Winning Teams of All Seasons

# In[7]:


WiningTeams=[]
y = 0 
for i in ipldata['WinningTeam']:
    if ipldata['MatchNumber'][y] == "Final":
        WiningTeams.append(i)
    y+=1
WiningTeams


# In[8]:


plt.figure(figsize=(12,5))
plt.scatter(WiningTeams,ipldata['Season'].unique())
plt.xlabel("Teams", fontsize=10,fontweight="bold")
plt.ylabel("Season",fontsize=10,fontweight="bold")
plt.title("IPL Winners", fontsize=12,fontweight="bold")
plt.show()


# ###### <hr>
# ###### Total Matches Win by each Team

# In[9]:


eachteamwinmatches=ipldata.groupby(["WinningTeam"])["ID"].count().reset_index().rename(columns={"ID":"Matches Won"})
eachteamwinmatches


# ###### <hr>
# ###### Each Season Final's Man of the Match

# In[10]:


manofthematch=[]
y = 0 
for i in ipldata['Player_of_Match']:
    if ipldata['MatchNumber'][y] == "Final":
        manofthematch.append(i)
    y+=1
manofthematch
plt.figure(figsize=(15,5))
plt.scatter(manofthematch,ipldata['Season'].unique())
plt.xlabel("Season", fontsize=10,fontweight="bold")
plt.ylabel("Man of the match",fontsize=10,fontweight="bold")
plt.title("IPL Final Man of the Match", fontsize=12,fontweight="bold")
plt.show()


# ###### <hr>
# #### Checking Dataset on the bases of Venue

# In[11]:


venue=ipldata[["Venue","Team1","Team2","TossWinner","TossDecision","WinningTeam"]]
venue["TossLosser"]=ipldata["Team1"]
venue.loc[venue["TossWinner"]!=venue["Team1"],"TossLosser"]=venue["Team1"]
venue


# ###### <hr>
# ###### Only taking data from Venue Dataset to check Match lose & Win on the bases of Toss Result

# In[12]:


venue.loc[venue["TossDecision"]=="bat","BattingTeam"]=venue["TossWinner"]
venue.loc[venue["TossDecision"]=="field","BattingTeam"]=venue["TossLosser"]
venue=venue.drop(columns=["Team1","Team2","TossDecision","TossWinner","TossLosser"])
venue.loc[venue["WinningTeam"]==venue["BattingTeam"],"BattingWin"]=1
venue.loc[venue["WinningTeam"]!=venue["BattingTeam"],"BattingWin"]=0
venue


# In[13]:


batting_win=venue[venue["BattingWin"]==1]
batting_win=batting_win.drop(columns=["BattingWin"])
batting_losses=venue[venue["BattingWin"]==0]
batting_losses=batting_losses.drop(columns=["BattingWin"])
print("Batting First & Win")
batting_win


# In[14]:


print("Batting First & Lose")
batting_losses


# In[15]:


x=batting_win.groupby("Venue")["Venue"].count()
y=batting_losses.groupby("Venue")["Venue"].count()
plt.figure(figsize=(20,10))
batting_win.groupby("Venue")["Venue"].count().plot(kind='bar')
plt.xticks(rotation='vertical')

plt.show()


# ###### <hr>
# ###### Venue where Batting First 100% chance of winning

# In[16]:


print("Stadium where win 100% by batting first")
ipldata.Venue[ipldata.WonBy != "Wickets"].mode()


# ###### <hr>
# ###### Venue where Bowling First 100% chance of winning

# In[17]:


print("Stadium where win 100% by bowling first")
ipldata.Venue[ipldata.WonBy == "Wickets"].mode()


# ###### <hr>
# ###### Man of Match of all IPL Matches

# In[18]:


player=ipldata[["Team1","Team2","Player_of_Match","Team1Players","Team2Players"]]
player


# ###### <hr>
# ###### Top 10 players who got most of Man of Match

# In[19]:


player_of_match=player.groupby("Player_of_Match")["Player_of_Match"].count().sort_values(ascending=False)
player_of_match[:10].plot(kind='bar')
plt.xticks(rotation='vertical')

plt.show()


# ###### <hr>
# ###### Checking whetter winning a toss means wins the game

# In[20]:


toss=(ipldata["TossWinner"]==ipldata["WinningTeam"])
toss.mode()


# ###### <hr>
# ###### Checking whetter winning a toss means wins the game (By Graphical Representation)

# In[21]:


plt.figure(figsize=(5,5))
sns.countplot(toss)
plt.suptitle("Toss Winner", fontsize=12,fontweight="bold")
plt.title("Match Win=True -- Match Lose=False", fontsize=10,fontweight="bold")
plt.show()


# ###### <hr>
# ###### Checking whetter losing a toss means wins the game (By Graphical Representation)

# In[22]:


tossloss=(ipldata["TossWinner"]!=ipldata["WinningTeam"])
plt.figure(figsize=(5,5))
sns.countplot(tossloss)
plt.suptitle("Toss Looser", fontsize=12,fontweight="bold")
plt.title("Match Win=True -- Match Lose=False", fontsize=10,fontweight="bold")
plt.show()


# ###### <hr>
# ###### Checking whetter Choosing bowl or bat first means wins the game (By Graphical Representation)

# In[23]:


plt.figure(figsize=(5,4))
sns.countplot(ipldata.TossDecision[ipldata.TossWinner==ipldata.WinningTeam])
plt.suptitle("Toss Winner", fontsize=12,fontweight="bold")
plt.title("Batting First & Bowling First Wining Graph", fontsize=10,fontweight="bold")
plt.show()


# ###### <hr>
# ###### Checking whetter losing a toss and put it into bowl or bat first means wins the game (By Graphical Representation)

# In[24]:


plt.figure(figsize=(5,4))
sns.countplot(ipldata.TossDecision[ipldata.TossWinner!=ipldata.WinningTeam])
plt.suptitle("Toss Losser", fontsize=12,fontweight="bold")
plt.title("Batting First & Bowling First Wining Graph", fontsize=10,fontweight="bold")
plt.show()


# ###### <hr>
# ###### Checking Chennai Super Kings Toss Won & Win the Match

# In[25]:


CSKtossdata = ipldata[ipldata['TossWinner']=='Chennai Super Kings']
CSKtossdata


# In[26]:


plt.figure(figsize=(5,5))
sns.countplot(CSKtossdata.TossWinner==CSKtossdata.WinningTeam)
plt.suptitle("Chennai Super Kings Toss Winner", fontsize=12,fontweight="bold")
plt.title("Match Win=True -- Match Lose=False", fontsize=10,fontweight="bold")
plt.show()


# ###### <hr>
# ###### Checking toss win ratio of Cheenai Super Kings when they won a match

# In[27]:


CSKdata = ipldata[ipldata['WinningTeam']=='Chennai Super Kings']
CSKdata


# In[28]:


plt.figure(figsize=(5,5))
sns.countplot(CSKdata.TossWinner==CSKdata.WinningTeam)
plt.suptitle("Chennai Super Kings Toss Graph in a winning Match", fontsize=12,fontweight="bold")
plt.title("Toss Winner=True -- Toss Looser=False", fontsize=10,fontweight="bold")
plt.show()


# ###### <hr>
# ###### Checking Kolkata Knight Riders Toss Won & Win the Match

# In[29]:


KKRdata = ipldata[ipldata['WinningTeam']=='Kolkata Knight Riders']
KKRdata


# In[30]:


plt.figure(figsize=(5,5))
sns.countplot(KKRdata.TossWinner==KKRdata.WinningTeam)
plt.suptitle("Kolkata Knight Riders Wining Graph", fontsize=12,fontweight="bold")
plt.title("Toss Winner=True -- Toss Looser=False", fontsize=10,fontweight="bold")
plt.show()


# ###### <hr>
# ###### Checking Dataset Information

# In[31]:


ipldata.info()


# ###### <hr>
# ###### Dropping Unusefull Feature for Prediction

# In[32]:


iplnewdata=ipldata.drop(columns=['ID','SuperOver','City','Date','Season','Player_of_Match','Team1Players','Team2Players',
                                 'Umpire1','Umpire2','method','Margin','WonBy'])


# In[33]:


iplnewdata.info()


# ###### <hr>
# ###### Removing Row which have empty entities

# In[34]:


iplnewdata=iplnewdata.dropna()


# ###### <hr>
# ###### Checking Match Number Type & Counting their reptition

# In[35]:


print("Counting Match Number values in dataset")
print(iplnewdata.MatchNumber.value_counts())


# ###### <hr>
# ###### Checking Team1 names & Counting their reptition

# In[36]:


print("Counting Team1 values in dataset")
print(iplnewdata.Team1.value_counts())


# ###### <hr>
# ###### Checking Team2 names & Counting their reptition

# In[37]:


print("Counting Team2 values in dataset")
print(iplnewdata.Team2.value_counts())


# ###### <hr>
# ###### Checking Venues Name & Counting their reptition

# In[38]:


print("Counting Venue values in dataset")
print(iplnewdata.Venue.value_counts())


# ###### <hr>
# ###### Checking Toss Winner Teams Name & Counting their reptition

# In[39]:


print("Counting TossWinner values in dataset")
print(iplnewdata.TossWinner.value_counts())


# ###### <hr>
# ###### Checking Toss Decision & Counting their reptition

# In[40]:


print("Counting TossDecision values in dataset")
print(iplnewdata.TossDecision.value_counts())


# ###### <hr>
# ###### Checking Winning Teams Name & Counting their reptition

# In[41]:


print("Counting WinningTeam values in dataset")
print(iplnewdata.WinningTeam.value_counts())


# ###### <hr>
# ###### Converting string Values of all features into integer for Machine Learning

# In[42]:


iplnewdata=iplnewdata.replace({'WinningTeam':{'Mumbai Indians':0,'Chennai Super Kings':1,'Kolkata Knight Riders':2,
                                   'Royal Challengers Bangalore':3,'Kings XI Punjab':4,'Rajasthan Royals':5,
                                   'Sunrisers Hyderabad':6,'Delhi Daredevils':7,'Delhi Capitals':8,'Deccan Chargers':9,
                                   'Gujarat Lions':10,'Rising Pune Supergiant':11,'Pune Warriors':12,'Punjab Kings':13,
                                   'Kochi Tuskers Kerala':14,'Rising Pune Supergiants':15}})


# In[43]:


iplnewdata=iplnewdata.replace({'TossDecision':{'field':0,'bat':1}})


# In[44]:


iplnewdata=iplnewdata.replace({'TossWinner':{'Mumbai Indians':0,'Chennai Super Kings':1,'Kolkata Knight Riders':2,
                                   'Royal Challengers Bangalore':3,'Kings XI Punjab':4,'Rajasthan Royals':5,
                                   'Sunrisers Hyderabad':6,'Delhi Daredevils':7,'Delhi Capitals':8,'Deccan Chargers':9,
                                   'Gujarat Lions':10,'Rising Pune Supergiant':11,'Pune Warriors':12,'Punjab Kings':13,
                                   'Kochi Tuskers Kerala':14,'Rising Pune Supergiants':15}})


# In[45]:


iplnewdata=iplnewdata.replace({'Team1':{'Mumbai Indians':0,'Chennai Super Kings':1,'Kolkata Knight Riders':2,
                                   'Royal Challengers Bangalore':3,'Kings XI Punjab':4,'Rajasthan Royals':5,
                                   'Sunrisers Hyderabad':6,'Delhi Daredevils':7,'Delhi Capitals':8,'Deccan Chargers':9,
                                   'Gujarat Lions':10,'Rising Pune Supergiant':11,'Pune Warriors':12,'Punjab Kings':13,
                                   'Kochi Tuskers Kerala':14,'Rising Pune Supergiants':15}})


# In[46]:


iplnewdata=iplnewdata.replace({'Team2':{'Mumbai Indians':0,'Chennai Super Kings':1,'Kolkata Knight Riders':2,
                                   'Royal Challengers Bangalore':3,'Kings XI Punjab':4,'Rajasthan Royals':5,
                                   'Sunrisers Hyderabad':6,'Delhi Daredevils':7,'Delhi Capitals':8,'Deccan Chargers':9,
                                   'Gujarat Lions':10,'Rising Pune Supergiant':11,'Pune Warriors':12,'Punjab Kings':13,
                                   'Kochi Tuskers Kerala':14,'Rising Pune Supergiants':15}})


# In[47]:


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


# In[48]:


iplnewdata=iplnewdata.replace({'MatchNumber':{'Final':0,'Qualifier 2':1,'Eliminator':3,'Qualifier 1':4,'3rd Place Play-Off':5,
                                            '1':6, '2':7, '3':8, '4':9, '5':10, '6':11, '7':12, '8':13, '9':14, '10':15, 
                                            '11':16, '12':17, '13':18, '14':19, '15':20, '16':21, '17':22, '18':23, '19':24, 
                                            '20':25, '21':26, '22':27, '23':28, '24':29, '25':30, '26':31, '27':32, '28':33, 
                                            '29':34, '30':35, '31':36, '32':37, '33':38, '34':39, '35':40, '36':41, '37':42, 
                                            '38':43, '39':44, '40':45, '41':46, '42':47, '43':48, '44':49, '45':50, '46':51, 
                                            '47':52, '48':53, '49':54, '50':55, '51':56, '52':57, '53':58, '54':59, '55':60, 
                                            '56':61, '57':62, '58':63, '59':64, '60':65, '61':66, '62':67, '63':68, '64':69, 
                                            '65':70, '66':71, '67':72, '68':73, '69':74, '70':75, '71':76, 'Semi Final': 77, 
                                            'Qualifier':78, 'Elimination Final':79}})


# ###### <hr>
# ###### New Dataset in Integer Form

# In[49]:


iplnewdata


# ###### <hr>
# ###### Sorting Dataset in Random Sequence for better training

# In[50]:


iplnewdata = iplnewdata.sort_values('TossWinner',ascending=False)


# In[51]:


iplnewdata


# ###### <hr>
# ###### Spliting Features into Two dataset & Converting X dataset in Scaler Values to get a linear data

# In[52]:


X = iplnewdata.drop(columns=['WinningTeam'])


# In[53]:


X


# In[54]:


scaler = StandardScaler()


# In[55]:


scaler.fit(X)


# In[56]:


standardized_data = scaler.transform(X)


# In[57]:


standardized_data


# In[58]:


X = standardized_data
Y = iplnewdata['WinningTeam']


# ###### <hr>
# ###### Spliting Data into Testing & Training data

# In[59]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[60]:


print(X.shape, X_train.shape, X_test.shape)


# ###### <hr>
# ###### SVM Model for Predictions 

# In[61]:


classifier = svm.SVC(kernel='linear')


# In[62]:


classifier.fit(X_train, Y_train)


# In[63]:


X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[64]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[65]:


X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[66]:


print('Accuracy score of the test data : ', test_data_accuracy)


# ###### <hr>
# ###### Make prediction of IPL 22 matches

# In[74]:


input_data = (22,1,3,12,3,0)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('Team 1 Winner')
else:
  print('Team 2 Winner')


# In[ ]:




