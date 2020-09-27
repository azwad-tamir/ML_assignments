# importing required packages:
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from functools import reduce
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import csv
import random
import math


########################################################################################################################
## Task 01:
########################################################################################################################
# Importing the toy dataset:
train_df = pd.read_csv('./football_game_dataset/train.csv')
test_df = pd.read_csv('./football_game_dataset/test.csv')
combine = [train_df, test_df]
unique_values1 = pd.unique(test_df['Media'])
unique_values2 = pd.unique(train_df['Media'])
# Converting to numerical data
for dataset in combine:
    dataset['Opponent'] = dataset['Opponent'].map( {'Texas':1, 'Virginia':2, 'GeorgiaTech':3, 'UMass':4, 'Clemson':5, 'Navy':6,
                                                   'USC':7, 'Temple':8, 'PITT':9, 'WakeForest':10, 'BostonCollege':11, 'Stanford':12,
                                                   'Nevada':13, 'MichiganState':14, 'Duke':15, 'Syracuse':16, 'NorthCarolinaState':17,
                                                   'MiamiFlorida':18, 'Army':19, 'VirginiaTech':20, 'Georgia':21, 'MiamiOhio':22,
                                                   'NorthCarolina':23} ).astype(int)

for dataset in combine:
    dataset['Is_Home_or_Away'] = dataset['Is_Home_or_Away'].map( {'Home': 0, 'Away': 1} ).astype(int)

for dataset in combine:
    dataset['Is_Opponent_in_AP25_Preseason'] = dataset['Is_Opponent_in_AP25_Preseason'].map( {'In': 0, 'Out': 1} ).astype(int)

for dataset in combine:
    dataset['Media'] = dataset['Media'].map( {'1-NBC':0, '2-ESPN':1, '3-FOX':2, '4-ABC':3, '5-CBS':4} ).astype(int)

for dataset in combine:
    dataset['Label'] = dataset['Label'].map( {'Win':1, 'Lose':0} ).astype(int)

# Selecting the features to put into the classifier:
X_train = train_df[['Is_Home_or_Away', 'Is_Opponent_in_AP25_Preseason', 'Media']]
y_train = train_df[['Label']]
X_test = test_df[['Is_Home_or_Away', 'Is_Opponent_in_AP25_Preseason', 'Media']]
y_test = test_df[['Label']]

# Creating Naive_Bayes Classifier:
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
sum =0
y_pred = y_pred.reshape(12, 1)
y_test1 = y_test.to_numpy()
fp = 0
fn = 0
tp = 0
tn = 0
labels_pred = []
for i in range(0,12):
    sum += y_pred[i,0] == y_test1[i,0]
    if y_pred[i,0] == 0:
        labels_pred.append('Lose')
    else:
        labels_pred.append('Win')

    if (y_pred[i,0] == 1) and (y_test1[i,0] == 1):
        tp += 1
    elif (y_pred[i,0] == 1) and (y_test1[i,0] == 0):
        fp += 1
    elif (y_pred[i,0] == 0) and (y_test1[i,0] == 1):
        fn += 1
    elif (y_pred[i,0] == 0) and (y_test1[i,0] == 0):
        tn += 1
    else:
        print("Fetal Error: Check accuracy and recall")

print("Naive_Bayes Classifier:\nAccuracy: ", sum/12*100, "%")
Precision = tp / (tp+fp)
Recall = tp / (tp+fn)
f1_score = 2*(Recall * Precision) / (Recall + Precision)
print("Precision: ", Precision, "\nRecall: ", Recall, "\nf1_score: ", f1_score)
print(labels_pred)

# Creating K-nearest neighbors classifier:
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
predicted = model.predict(X_test)
sum =0
predicted = predicted.reshape(12, 1)
fp = 0
fn = 0
tp = 0
tn = 0
labels_pred = []
for i in range(0,12):
    sum += predicted[i,0] == y_test1[i,0]

    if predicted[i,0] == 0:
        labels_pred.append('Lose')
    else:
        labels_pred.append('Win')

    if (predicted[i,0] == 1) and (y_test1[i,0] == 1):
        tp += 1
    elif (predicted[i,0] == 1) and (y_test1[i,0] == 0):
        fp += 1
    elif (predicted[i,0] == 0) and (y_test1[i,0] == 1):
        fn += 1
    elif (predicted[i,0] == 0) and (y_test1[i,0] == 0):
        tn += 1
    else:
        print("Fetal Error: Check accuracy and recall")

print("K-nearest neighbor Classifier:\nAccuracy: ", sum/12*100, "%")
Precision = tp / (tp+fp)
Recall = tp / (tp+fn)
f1_score = 2*(Recall * Precision) / (Recall + Precision)
print("Precision: ", Precision, "\nRecall: ", Recall, "\nf1_score: ", f1_score)
print(labels_pred)

########################################################################################################################
## Task 02:
########################################################################################################################
# Reading in the dataset from csv file:
passenger = pd.read_csv ('./titanic/gender_submission.csv')
train_df = pd.read_csv('./titanic/train.csv')
test_df = pd.read_csv('./titanic/test.csv')
combine = [train_df, test_df]
print(train_df.columns.values)
print(train_df.Embarked.hasnans)
print(test_df.Cabin.isna().sum())

# Converting Sex to numerical data
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# Filling in missing age values using KNN inputer
X = train_df['Age'].values
X = X.reshape(-1,1)
imputer = KNNImputer(n_neighbors=5)
Xtrans = imputer.fit_transform(X)
train_df = pd.concat([pd.DataFrame(Xtrans),train_df],axis=1)
del train_df['Age']
train_df.rename(columns={0:'Age'},inplace=True)

X = test_df['Age'].values
X = X.reshape(-1,1)
imputer = KNNImputer(n_neighbors=5)
Xtrans = imputer.fit_transform(X)
test_df = pd.concat([pd.DataFrame(Xtrans),test_df],axis=1)
del test_df['Age']
test_df.rename(columns={0:'Age'},inplace=True)

print(train_df.Age.head())
print(train_df.Age.hasnans)
print(test_df.Age.hasnans)

# Filling in Embarked values with mode
freq_port = train_df.Embarked.dropna().mode()[0]
train_df = train_df.fillna(freq_port)
print(train_df.Embarked.hasnans)
print(test_df.Embarked.hasnans)

# Converting Embarked to numerical data
combine = [train_df, test_df]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'C': 0, 'Q': 1, 'S' : 2} ).astype(int)
print(train_df.Embarked.head())

# Filling in fare values in test data
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.Fare.head()
print(train_df.Fare.hasnans)
print(test_df.Fare.hasnans)

# Grouping Age values in test and train data:
train_df['AgeBand'] = pd.qcut(train_df['Age'], 6)
print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))
print(train_df.AgeBand.head())
def convert_num1(x):
    if x >= -0.001 and x <= 4:
        return 0
    elif x > 4 and x <= 10:
        return 1
    elif x > 10 and x <= 25:
        return 2
    elif x > 25 and x <= 60:
        return 3
    else:
        return 4

train_df.Age = train_df["Age"].apply(convert_num1)
del train_df['AgeBand']
print(test_df.Fare.head())
print(test_df.Age.max())
test_df.Age = test_df["Age"].apply(convert_num1)
print(train_df.Age.hasnans)
print(test_df.Age.hasnans)

# grouping fare values in train and test
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))
print(train_df.FareBand.head())
def convert_num(x):
    if x >= -0.001 and x <= 7.91:
        return 0
    elif x > 7.91 and x <= 14.454:
        return 1
    elif x > 14.454 and x <= 31.0:
        return 2
    elif x > 31 and x <= 513:
        return 3

train_df.Fare = train_df["Fare"].apply(convert_num)
del train_df['FareBand']
print(train_df.Fare.head())
print(test_df.Fare.max())
test_df.Fare = test_df["Fare"].apply(convert_num)

# Selecting the features to put into the classifier:
A =train_df.head()
X_all = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y_all = train_df[['Survived']]

# Applying five-fold cross validation split:
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, random_state=1, test_size=0.2)

# Creating Naive_Bayes Classifier:
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
sum =0
y_pred = y_pred.reshape(179, 1)
y_test1 = y_test.to_numpy()
fp = 0
fn = 0
tp = 0
tn = 0
for i in range(0,179):
    sum += y_pred[i,0] == y_test1[i,0]
    if (y_pred[i,0] == 1) and (y_test1[i,0] == 1):
        tp += 1
    elif (y_pred[i,0] == 1) and (y_test1[i,0] == 0):
        fp += 1
    elif (y_pred[i,0] == 0) and (y_test1[i,0] == 1):
        fn += 1
    elif (y_pred[i,0] == 0) and (y_test1[i,0] == 0):
        tn += 1
    else:
        print("Fetal Error: Check accuracy and recall")

print("Naive_Bayes Classifier:\nAccuracy: ", sum/179*100, "%")
Precision = tp / (tp+fp)
Recall = tp / (tp+fn)
f1_score = 2*(Recall * Precision) / (Recall + Precision)
print("Precision: ", Precision, "\nRecall: ", Recall, "\nf1_score: ", f1_score)

# Creating K-nearest neighbors classifier from scratch:
X_train_raw = X_train.to_numpy()
y_train_raw = y_train.to_numpy()
X_test_raw = X_test.to_numpy()
y_test_raw = y_test.to_numpy()

X_train_array = np.zeros(X_train_raw.shape)
y_train_array = y_train.to_numpy()
X_test_array = np.zeros(X_test_raw.shape)
y_test_array = y_test.to_numpy()
# Normalizing the Dataset:
for i in range(0, X_train_raw.shape[1]):
    X_train_array[:,i] = (X_train_raw[:,i])/np.max(X_train_raw[:,i])
    X_test_array[:,i] = (X_test_raw[:,i])/np.max(X_test_raw[:,i])
num=0
accuracy = []
precision = []
recall = []
f1_score = []

for k in range(1, 500, 2):
    predicted_labels = []
    for sample_test in X_test_array:
        vote = np.array([0, 0])
        euclidean_distance = []
        for sample_train in X_train_array:
            distance = pow(sample_test - sample_train, 2)
            total_distance = np.sum(distance)
            euclidean_distance.append(np.sqrt(total_distance))

        sorted_distance = sorted(((v, i) for i, v in enumerate(euclidean_distance)))
        for i in range(0,k):
            vote[y_train_array[sorted_distance[i][1]]] += 1

        if vote[0] > vote[1]:
            predicted_labels.append(0)
        elif vote[1] > vote[0]:
            predicted_labels.append(1)
        else:
            print("Fetal Error! Check voting algorithm.")

    sum =0
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for i in range(0,179):
        sum += predicted_labels[i] == y_test_array[i,0]
        if (predicted_labels[i] == 1) and (y_test_array[i,0] == 1):
            tp += 1
        elif (predicted_labels[i] == 1) and (y_test_array[i,0] == 0):
            fp += 1
        elif (predicted_labels[i] == 0) and (y_test_array[i,0] == 1):
            fn += 1
        elif (predicted_labels[i] == 0) and (y_test_array[i,0] == 0):
            tn += 1
        else:
            print("Fetal Error: Check accuracy and recall")

    Accuracy = sum/179*100
    #print("K-nearest neighbor Classifier:\nAccuracy: ", sum/179*100, "%")
    if tp+fp ==0:
        Precision = 0
    else:
        Precision = tp / (tp + fp)
    if tp+fn == 0:
        Recall = 0
    else:
        Recall = tp / (tp+fn)
    if Precision+Recall == 0:
        F1_score = 0
    else:
        F1_score = 2*(Recall * Precision) / (Recall + Precision)
    accuracy.append(Accuracy)
    precision.append(Precision)
    recall.append(Recall)
    f1_score.append(F1_score)
    #print("Precision: ", Precision, "\nRecall: ", Recall, "\nf1_score: ", f1_score)
    print('K = ', k, ' : ', accuracy[num])
    num+=1

plt.plot(range(1,500,2), accuracy)
plt.xlabel('k-values')
plt.ylabel('Average accuracy in %')

print("K-nearest neighbor Algorithm:\nAccuracy: ",accuracy[2], "\nPrecision: ", precision[2], "\nRecall: ", recall[2], "\nF1_score: ", f1_score[2])
