# importing required packages:
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import numpy as np


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

# Creating SVM classifier:
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
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
